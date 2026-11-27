"""Step 4 of RAG: write the answer from the retrieved chunks.

Two modes (chosen automatically):
  * ASKPDF_API_KEY and ASKPDF_MODEL are set -> an AI model writes the answer.
  * otherwise -> "extractive" mode: show the best matching
                 sentences from the PDF (works fully offline).
"""
import os

from .rag import Chunk, split_sentences, tokenize


SYSTEM_PROMPT = (
    "You answer questions about a PDF document. Use ONLY the excerpts provided. "
    "If the excerpts do not contain the answer, say you could not find it in the "
    "document. Mention page numbers like (p. 3) when you use an excerpt. Be concise."
)


def llm_available() -> bool:
    return bool(os.environ.get("ASKPDF_API_KEY") and os.environ.get("ASKPDF_MODEL"))


def mode() -> str:
    return "ai" if llm_available() else "extractive"


def build_context(hits: list[tuple[Chunk, float]]) -> str:
    return "\n\n".join(f"[Page {c.page}]\n{c.text}" for c, _ in hits)


def answer_with_ai(question: str, hits: list[tuple[Chunk, float]]) -> str:
    import anthropic

    client = anthropic.Anthropic(api_key=os.environ["ASKPDF_API_KEY"])
    msg = client.messages.create(
        model=os.environ["ASKPDF_MODEL"],
        max_tokens=800,
        system=SYSTEM_PROMPT,
        messages=[
            {
                "role": "user",
                "content": f"Excerpts:\n{build_context(hits)}\n\nQuestion: {question}",
            }
        ],
    )
    return "".join(b.text for b in msg.content if getattr(b, "type", "") == "text").strip()


def answer_extractive(question: str, hits: list[tuple[Chunk, float]], n_sents: int = 3) -> str:
    """Pick the sentences that share the most words with the question."""
    q = set(tokenize(question))
    candidates = []
    for chunk, score in hits:
        for s in split_sentences(chunk.text):
            overlap = len(q & set(tokenize(s)))
            if overlap:
                candidates.append((overlap, score, chunk.page, s))
    if not candidates:
        return "I couldn't find an answer to that in the document."
    candidates.sort(key=lambda x: (x[0], x[1]), reverse=True)
    seen, picked = set(), []
    for _, _, page, s in candidates:
        if s in seen:
            continue
        seen.add(s)
        picked.append((page, s))
        if len(picked) == n_sents:
            break
    return " ".join(f"{s} (p. {page})" for page, s in picked)


def answer(question: str, hits: list[tuple[Chunk, float]]) -> tuple[str, str]:
    """Return (answer_text, mode_used). Falls back to extractive if the AI call fails."""
    if not hits:
        return "I couldn't find anything about that in the document.", mode()
    if llm_available():
        try:
            return answer_with_ai(question, hits), "ai"
        except Exception as exc:  # network/auth/etc: never crash the app
            text = answer_extractive(question, hits)
            return f"{text}\n\n(The AI model was unavailable: {type(exc).__name__}. Showing extracted text instead.)", "extractive"
    return answer_extractive(question, hits), "extractive"
