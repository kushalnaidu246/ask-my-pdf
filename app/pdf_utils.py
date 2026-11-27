"""Step 1 of RAG: get the text out of the PDF, page by page."""
from io import BytesIO

from pypdf import PdfReader


class PDFError(ValueError):
    """Raised when the PDF is broken or has no readable text."""


def extract_pages(data: bytes) -> list[str]:
    """Return a list of page texts (index 0 = page 1)."""
    try:
        reader = PdfReader(BytesIO(data))
        if reader.is_encrypted:
            # Many "encrypted" PDFs just use an empty password.
            if not reader.decrypt(""):
                raise PDFError("This PDF is password protected.")
        pages = [(page.extract_text() or "").strip() for page in reader.pages]
    except PDFError:
        raise
    except Exception as exc:  # pypdf raises many different error types
        raise PDFError(f"Could not read this file as a PDF: {exc}") from exc

    if not any(pages):
        raise PDFError(
            "No text found in this PDF. It may be a scanned image (needs OCR)."
        )
    return pages
