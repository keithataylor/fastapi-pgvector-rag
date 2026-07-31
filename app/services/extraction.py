"""Plain-text extraction for supported document formats."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from pypdf import PdfReader
from pypdf.errors import PdfReadError


class DocumentExtractionError(ValueError):
    """Raised when a document cannot yield usable plain text."""


class UnsupportedDocumentTypeError(DocumentExtractionError):
    """Raised when a file suffix is not supported for ingestion."""


class EmptyDocumentError(DocumentExtractionError):
    """Raised when extracted document content contains no text."""


@dataclass(frozen=True)
class ExtractedSection:
    """Text from a document section with optional PDF page provenance."""

    text: str
    page_number: int | None


def extract_document(path: Path) -> list[ExtractedSection]:
    """Extract usable text from a UTF-8 text file or text-bearing PDF."""
    suffix = path.suffix.lower()
    if suffix == ".txt":
        return _extract_text_file(path)
    if suffix == ".pdf":
        return _extract_pdf(path)
    raise UnsupportedDocumentTypeError(f"Unsupported document type: {suffix or 'none'}")


def _extract_text_file(path: Path) -> list[ExtractedSection]:
    try:
        text = path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError) as error:
        raise DocumentExtractionError("Unable to read UTF-8 text document.") from error

    return _nonempty_sections([ExtractedSection(text=text, page_number=None)])


def _extract_pdf(path: Path) -> list[ExtractedSection]:
    try:
        reader = PdfReader(path)
        sections = [
            ExtractedSection(text=page.extract_text() or "", page_number=index)
            for index, page in enumerate(reader.pages, start=1)
        ]
    except (OSError, PdfReadError) as error:
        raise DocumentExtractionError("Unable to read PDF document.") from error

    return _nonempty_sections(sections)


def _nonempty_sections(sections: list[ExtractedSection]) -> list[ExtractedSection]:
    nonempty_sections = [section for section in sections if section.text.strip()]
    if not nonempty_sections:
        raise EmptyDocumentError("Document contains no extractable text.")
    return nonempty_sections
