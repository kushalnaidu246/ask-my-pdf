"""Steps 2-3 of RAG: split text into chunks, then search them (BM25).

BM25 is a classic keyword-ranking formula (used by search engines).
It needs no model download, so the project works right after `pip install`.
"""
import math
import re
from collections import Counter
from dataclasses import dataclass

STOPWORDS = set(
    """a an the and or but if of to in on at by for with about as is are was were be
    been being it its this that these those from into than then so such not no do
    does did have has had can could should would will what which who whom whose
    when where why how i you he she we they me my your our their there here""".split()
)

_WORD = re.compile(r"[a-z0-9]+")


def tokenize(text: str) -> list[str]:
    """Lowercase words, no stopwords, tiny plural-stemming ('cats' -> 'cat')."""
    out = []
    for w in _WORD.findall(text.lower()):
        if w in STOPWORDS:
            continue
        if len(w) > 4 and w.endswith("ies"):
            w = w[:-3] + "y"
        elif len(w) > 3 and w.endswith("s") and not w.endswith("ss"):
            w = w[:-1]
        out.append(w)
    return out


@dataclass
class Chunk:
    id: int
    page: int  # 1-based page number
    text: str


def split_sentences(text: str) -> list[str]:
    parts = re.split(r"(?<=[.!?])\s+|\n{2,}", text)
    return [" ".join(p.split()) for p in parts if p.strip()]


def chunk_pages(pages: list[str], max_chars: int = 900, overlap_sents: int = 1) -> list[Chunk]:
    """Cut every page into ~900-character pieces made of whole sentences.

    A little overlap (the last sentence repeats) keeps ideas from being cut in half.
    """
    chunks: list[Chunk] = []
    for page_no, text in enumerate(pages, start=1):
        sents = split_sentences(text)
        buf: list[str] = []
        size = 0
        for s in sents:
            # A single giant "sentence" (e.g. a table) gets hard-split.
            while len(s) > max_chars:
                if buf:
                    chunks.append(Chunk(len(chunks), page_no, " ".join(buf)))
                    buf, size = [], 0
                chunks.append(Chunk(len(chunks), page_no, s[:max_chars]))
                s = s[max_chars:]
            if size + len(s) > max_chars and buf:
                chunks.append(Chunk(len(chunks), page_no, " ".join(buf)))
                buf = buf[-overlap_sents:] if overlap_sents else []
                size = sum(len(x) for x in buf)
            buf.append(s)
            size += len(s)
        if buf:
            chunks.append(Chunk(len(chunks), page_no, " ".join(buf)))
    return chunks


class BM25Index:
    def __init__(self, chunks: list[Chunk], k1: float = 1.5, b: float = 0.75):
        self.chunks = chunks
        self.k1, self.b = k1, b
        self.tfs = [Counter(tokenize(c.text)) for c in chunks]
        self.lens = [sum(tf.values()) for tf in self.tfs]
        self.avg_len = (sum(self.lens) / len(self.lens)) if self.lens else 0.0
        df: Counter = Counter()
        for tf in self.tfs:
            df.update(tf.keys())
        n = len(chunks)
        self.idf = {t: math.log(1 + (n - d + 0.5) / (d + 0.5)) for t, d in df.items()}

    def search(self, query: str, top_k: int = 4) -> list[tuple[Chunk, float]]:
        q = tokenize(query)
        scored = []
        for i, tf in enumerate(self.tfs):
            score = 0.0
            for t in q:
                if t not in tf:
                    continue
                f = tf[t]
                denom = f + self.k1 * (1 - self.b + self.b * self.lens[i] / (self.avg_len or 1))
                score += self.idf[t] * f * (self.k1 + 1) / denom
            if score > 0:
                scored.append((self.chunks[i], score))
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:top_k]
