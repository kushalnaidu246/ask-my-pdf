# 📄 Ask My PDF

Upload a PDF, ask a question, get an answer. A small **RAG** (Retrieval-Augmented Generation) app built with Python + FastAPI.

## Run it (3 commands)

```bash
python3 -m venv .venv && source .venv/bin/activate     # Windows: .venv\Scripts\activate
pip install -r requirements.txt
uvicorn app.main:app --reload
```

Open <http://127.0.0.1:8000>, drop in a PDF (try `samples/sample.pdf`), and ask, e.g. *"Which planet is the hottest?"*

## Two answer modes

| Mode | When | What you get |
|---|---|---|
| **AI** | `ASKPDF_API_KEY` and `ASKPDF_MODEL` are set | A natural-language answer written by an AI model from the best excerpts |
| **Offline** | no key | The best-matching sentences from your PDF, with page numbers |

```bash
export ASKPDF_API_KEY=your-key          # optional
export ASKPDF_MODEL=your-model-name     # optional, needed together with the key
```

## How RAG works here (step by step)

1. **Extract** – `pypdf` reads the text of each page (`app/pdf_utils.py`).
2. **Chunk** – text is cut into ~900-character pieces of whole sentences, remembering the page (`app/rag.py`).
3. **Retrieve** – your question is matched against the chunks with **BM25** keyword ranking; the top 4 win.
4. **Generate** – the winning chunks + your question go to the AI model (or the offline extractor) to write the answer (`app/llm.py`).

Memory trick: **E-C-R-G** — *Extract, Chunk, Retrieve, Generate*.

## API

| Endpoint | What it does |
|---|---|
| `POST /upload` (form file) | Index a PDF → `{doc_id, pages, chunks}` |
| `POST /ask` `{doc_id, question, top_k}` | Answer + source chunks |
| `GET /documents`, `DELETE /documents/{id}` | List / remove |
| `GET /health` | Status and answer mode |

Interactive docs: <http://127.0.0.1:8000/docs>

## Test

```bash
python -m pytest -q
```

## Limits

* Documents live in memory (lost on restart). 25 MB upload limit.
* Scanned PDFs (images) have no text and need OCR first.
* BM25 matches words, not meaning; swapping in embeddings is the natural next upgrade.
