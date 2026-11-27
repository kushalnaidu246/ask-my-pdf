import os
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

os.environ.pop("ASKPDF_API_KEY", None)  # tests run offline

from app.main import DOCS, app  # noqa: E402
from app.rag import BM25Index, chunk_pages, tokenize  # noqa: E402

SAMPLE = Path(__file__).parent.parent / "samples" / "sample.pdf"
client = TestClient(app)


@pytest.fixture(autouse=True)
def _clean():
    DOCS.clear()


@pytest.fixture(scope="session", autouse=True)
def _sample():
    if not SAMPLE.exists():
        import make_sample_pdf
        make_sample_pdf.main()


def upload():
    r = client.post("/upload", files={"file": ("sample.pdf", SAMPLE.read_bytes(), "application/pdf")})
    assert r.status_code == 200, r.text
    return r.json()


def test_tokenize_drops_stopwords_and_plurals():
    assert tokenize("The planets are big") == ["planet", "big"]


def test_search_finds_right_chunk():
    chunks = chunk_pages(["Cats purr softly. Dogs bark loudly.", "The moon orbits the Earth."])
    top = BM25Index(chunks).search("what does the moon orbit?", 1)[0][0]
    assert top.page == 2


def test_upload_and_ask():
    d = upload()
    assert d["pages"] == 3
    r = client.post("/ask", json={"doc_id": d["doc_id"], "question": "Which planet is the hottest?"})
    assert r.status_code == 200
    body = r.json()
    assert "Venus" in body["answer"]
    assert body["sources"][0]["page"] == 2


def test_largest_moon():
    d = upload()
    body = client.post("/ask", json={"doc_id": d["doc_id"], "question": "What is the largest moon?"}).json()
    assert "Ganymede" in body["answer"]


def test_unanswerable_question():
    d = upload()
    body = client.post("/ask", json={"doc_id": d["doc_id"], "question": "zebra bicycle sandwich"}).json()
    assert "couldn't find" in body["answer"].lower()


def test_rejects_non_pdf():
    r = client.post("/upload", files={"file": ("a.pdf", b"hello world", "application/pdf")})
    assert r.status_code == 400


def test_rejects_empty_and_corrupt():
    assert client.post("/upload", files={"file": ("a.pdf", b"", "application/pdf")}).status_code == 400
    assert client.post("/upload", files={"file": ("a.pdf", b"%PDF-1.4 garbage", "application/pdf")}).status_code == 422


def test_unknown_doc_and_delete():
    assert client.post("/ask", json={"doc_id": "nope", "question": "hi"}).status_code == 404
    d = upload()
    assert client.get("/documents").json()[0]["doc_id"] == d["doc_id"]
    assert client.delete(f"/documents/{d['doc_id']}").status_code == 200
    assert client.delete(f"/documents/{d['doc_id']}").status_code == 404


def test_homepage_and_health():
    assert "Ask My PDF" in client.get("/").text
    assert client.get("/health").json()["status"] == "ok"
