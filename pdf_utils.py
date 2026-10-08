"""PDF reading + text cleaning. No Streamlit code here, so it is easy to test."""
import re
from io import BytesIO

from pypdf import PdfReader

MAX_FILE_MB = 5


class PDFError(Exception):
    """Raised with a user-friendly message when a PDF can't be used."""


def clean_text(text: str) -> str:
    """Collapse extra spaces and blank lines."""
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n\s*\n+", "\n", text)
    return text.strip()


def extract_text_from_pdf(uploaded_file) -> str:
    """Return clean text from a Streamlit UploadedFile, or raise PDFError."""
    data = uploaded_file.getvalue()

    if len(data) > MAX_FILE_MB * 1024 * 1024:
        raise PDFError(f"'{uploaded_file.name}' is larger than {MAX_FILE_MB} MB.")
    if not data.startswith(b"%PDF"):  # every real PDF starts with these bytes
        raise PDFError(f"'{uploaded_file.name}' is not a valid PDF file.")

    try:
        reader = PdfReader(BytesIO(data))
        if reader.is_encrypted and reader.decrypt("") == 0:
            raise PDFError(f"'{uploaded_file.name}' is password-protected.")
        pages = [page.extract_text() or "" for page in reader.pages]
    except PDFError:
        raise
    except Exception as exc:  # corrupt file, unsupported structure, etc.
        raise PDFError(f"Could not read '{uploaded_file.name}'. The file may be corrupted.") from exc

    text = clean_text("\n".join(pages))
    if not text:
        raise PDFError(
            f"No text found in '{uploaded_file.name}'. It looks like a scanned/image PDF. "
            "Please upload a text-based PDF (or paste the text instead)."
        )
    return text
