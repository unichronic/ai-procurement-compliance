"""Extract text from uploaded tender documents.

Tenders arrive as PDF or DOCX, not as text someone pasted into a box, so
requiring paste is requiring the user to do the awkward part by hand.

PDF tables flatten badly (the same column-flattening seen in scanned BIS
annexes), so DOCX tables are read cell-by-cell to keep line items intact —
a tender's schedule of items is almost always a table, and that's precisely
the content the linter needs to read.
"""
from __future__ import annotations

import io
from typing import Optional

MAX_UPLOAD_BYTES = 10 * 1024 * 1024

SUPPORTED_TYPES = {
    "application/pdf": "pdf",
    "application/vnd.openxmlformats-officedocument.wordprocessingml.document": "docx",
    "text/plain": "txt",
}

SUPPORTED_EXTENSIONS = {".pdf": "pdf", ".docx": "docx", ".txt": "txt", ".md": "txt"}


class UnsupportedDocument(Exception):
    pass


def detect_kind(filename: Optional[str], content_type: Optional[str]) -> str:
    if content_type and content_type in SUPPORTED_TYPES:
        return SUPPORTED_TYPES[content_type]
    if filename:
        lowered = filename.lower()
        for ext, kind in SUPPORTED_EXTENSIONS.items():
            if lowered.endswith(ext):
                return kind
    raise UnsupportedDocument(
        f"Unsupported document type (filename={filename!r}, content_type={content_type!r}). "
        f"Supported: PDF, DOCX, TXT."
    )


def extract_pdf_text(data: bytes) -> str:
    from pypdf import PdfReader

    reader = PdfReader(io.BytesIO(data))
    pages = []
    for page in reader.pages:
        try:
            pages.append(page.extract_text() or "")
        except Exception:
            pages.append("")  # one unreadable page shouldn't lose the document
    return "\n".join(pages)


def extract_docx_text(data: bytes) -> str:
    import docx

    document = docx.Document(io.BytesIO(data))
    lines = [p.text for p in document.paragraphs]

    for table in document.tables:
        for row in table.rows:
            cells = [c.text.strip() for c in row.cells]
            # Word repeats a merged cell's text across the cells it spans.
            deduped = [c for i, c in enumerate(cells) if c and (i == 0 or c != cells[i - 1])]
            if deduped:
                lines.append(" | ".join(deduped))

    return "\n".join(lines)


def extract_text(data: bytes, *, filename: Optional[str] = None,
                 content_type: Optional[str] = None) -> str:
    if len(data) > MAX_UPLOAD_BYTES:
        raise UnsupportedDocument(
            f"Document exceeds {MAX_UPLOAD_BYTES // (1024 * 1024)} MB limit."
        )

    kind = detect_kind(filename, content_type)
    if kind == "pdf":
        text = extract_pdf_text(data)
    elif kind == "docx":
        text = extract_docx_text(data)
    else:
        text = data.decode("utf-8", errors="replace")

    if not text.strip():
        raise UnsupportedDocument(
            "No text could be extracted. If this is a scanned PDF it needs OCR first."
        )
    return text
