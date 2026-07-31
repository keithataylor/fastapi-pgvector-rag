from __future__ import annotations

from pathlib import Path

import pytest

from app.services.extraction import (
    DocumentExtractionError,
    EmptyDocumentError,
    ExtractedSection,
    UnsupportedDocumentTypeError,
    extract_document,
)


def test_extracts_utf8_text_with_no_page_number(tmp_path: Path) -> None:
    path = tmp_path / "notes.txt"
    path.write_text("Hello, world.", encoding="utf-8")

    assert extract_document(path) == [ExtractedSection("Hello, world.", None)]


def test_rejects_invalid_utf8_text(tmp_path: Path) -> None:
    path = tmp_path / "invalid.txt"
    path.write_bytes(b"\xff")

    with pytest.raises(DocumentExtractionError, match="Unable to read UTF-8"):
        extract_document(path)


def test_rejects_empty_text_document(tmp_path: Path) -> None:
    path = tmp_path / "empty.txt"
    path.write_text(" \n\t ", encoding="utf-8")

    with pytest.raises(EmptyDocumentError, match="no extractable text"):
        extract_document(path)


def test_rejects_unsupported_document_type(tmp_path: Path) -> None:
    path = tmp_path / "notes.md"
    path.write_text("Hello", encoding="utf-8")

    with pytest.raises(UnsupportedDocumentTypeError, match="Unsupported document type"):
        extract_document(path)


def test_extracts_pdf_pages_with_one_based_page_numbers(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    class FakePage:
        def __init__(self, text: str | None) -> None:
            self.text = text

        def extract_text(self) -> str | None:
            return self.text

    class FakeReader:
        pages = [FakePage("First page"), FakePage("Second page")]

    path = tmp_path / "pages.pdf"
    path.touch()
    monkeypatch.setattr("app.services.extraction.PdfReader", lambda _: FakeReader())

    assert extract_document(path) == [
        ExtractedSection("First page", 1),
        ExtractedSection("Second page", 2),
    ]


def test_rejects_pdf_with_no_extractable_text(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    class FakePage:
        def extract_text(self) -> None:
            return None

    class FakeReader:
        pages = [FakePage()]

    path = tmp_path / "empty.pdf"
    path.touch()
    monkeypatch.setattr("app.services.extraction.PdfReader", lambda _: FakeReader())

    with pytest.raises(EmptyDocumentError, match="no extractable text"):
        extract_document(path)


def test_rejects_malformed_pdf(tmp_path: Path) -> None:
    path = tmp_path / "broken.pdf"
    path.write_bytes(b"not a pdf")
