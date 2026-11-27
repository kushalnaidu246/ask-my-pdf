"""FastAPI server: upload a PDF, ask questions, get answers."""
import uuid
from dataclasses import dataclass
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from . import llm
from .pdf_utils import PDFError, extract_pages
from .rag import BM25Index, chunk_pages

MAX_UPLOAD_BYTES = 25 * 1024 * 1024
STATIC = Path(__file__).parent / "static"

app = FastAPI(title="Ask My PDF", version="1.0.0")


@dataclass
class Document:
    id: str
    filename: str
    pages: int
    index: BM25Index


DOCS: dict[str, Document] = {}  # in-memory store (cleared when the server restarts)


class AskRequest(BaseModel):
    doc_id: str
    question: str = Field(min_length=1, max_length=2000)
    top_k: int = Field(4, ge=1, le=10)


@app.get("/health")
def health():
    return {"status": "ok", "answer_mode": llm.mode(), "documents": len(DOCS)}


@app.post("/upload")
async def upload(file: UploadFile = File(...)):
    data = await file.read()
    if not data:
        raise HTTPException(400, "The file is empty.")
    if len(data) > MAX_UPLOAD_BYTES:
        raise HTTPException(413, "File too large (max 25 MB).")
    if not data.lstrip().startswith(b"%PDF"):
        raise HTTPException(400, "That doesn't look like a PDF file.")
    try:
        pages = extract_pages(data)
    except PDFError as exc:
        raise HTTPException(422, str(exc))
    chunks = chunk_pages(pages)
    doc = Document(uuid.uuid4().hex[:12], file.filename or "document.pdf", len(pages), BM25Index(chunks))
    DOCS[doc.id] = doc
    return {
        "doc_id": doc.id,
        "filename": doc.filename,
        "pages": doc.pages,
        "chunks": len(chunks),
        "answer_mode": llm.mode(),
    }


@app.post("/ask")
def ask(req: AskRequest):
    doc = DOCS.get(req.doc_id)
    if not doc:
        raise HTTPException(404, "Unknown document. Please upload the PDF again.")
    hits = doc.index.search(req.question, top_k=req.top_k)
    text, used = llm.answer(req.question, hits)
    return {
        "answer": text,
        "answer_mode": used,
        "sources": [
            {"page": c.page, "score": round(s, 3), "text": c.text} for c, s in hits
        ],
    }


@app.get("/documents")
def documents():
    return [{"doc_id": d.id, "filename": d.filename, "pages": d.pages} for d in DOCS.values()]


@app.delete("/documents/{doc_id}")
def delete_document(doc_id: str):
    if DOCS.pop(doc_id, None) is None:
        raise HTTPException(404, "Unknown document.")
    return {"deleted": doc_id}


@app.get("/")
def home():
    return FileResponse(STATIC / "index.html")


app.mount("/static", StaticFiles(directory=STATIC), name="static")
