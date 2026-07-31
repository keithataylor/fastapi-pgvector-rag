from __future__ import annotations

import pytest

from app.services.chunking import DEFAULT_CHUNK_SIZE, ChunkData, chunk_sections
from app.services.extraction import ExtractedSection


def test_default_chunk_size_is_1000() -> None:
    assert DEFAULT_CHUNK_SIZE == 1000


def test_chunks_are_deterministic_and_page_aware() -> None:
    sections = [
        ExtractedSection("alpha beta gamma delta", 1),
        ExtractedSection("epsilon zeta", 2),
    ]

    first = chunk_sections(sections, chunk_size=11)
    second = chunk_sections(sections, chunk_size=11)

    assert first == second
    assert first == [
        ChunkData(0, "alpha beta", 1),
        ChunkData(1, "gamma delta", 1),
        ChunkData(2, "epsilon zet", 2),
        ChunkData(3, "a", 2),
    ]


def test_ignores_an_early_boundary_and_hard_splits_at_chunk_size() -> None:
    text = "word break early-characters"

    chunks = chunk_sections([ExtractedSection(text, None)], chunk_size=20)

    assert chunks == [
        ChunkData(0, "word break early-cha", None),
        ChunkData(1, "racters", None),
    ]


def test_prefers_paragraph_boundary_within_final_twenty_percent() -> None:
    text = "abcdefghijklm\n\nrest"

    chunks = chunk_sections([ExtractedSection(text, 1)], chunk_size=15)

    assert chunks == [
        ChunkData(0, "abcdefghijklm", 1),
        ChunkData(1, "rest", 1),
    ]


def test_normalizes_line_endings_and_does_not_emit_empty_chunks() -> None:
    chunks = chunk_sections([ExtractedSection("\r\n alpha\r\nbeta \r", None)])

    assert chunks == [ChunkData(0, "alpha\nbeta", None)]


@pytest.mark.parametrize("chunk_size", [0, 10_001, True, "1000"])
def test_rejects_invalid_chunk_size(chunk_size: object) -> None:
    with pytest.raises(ValueError, match="chunk_size"):
        chunk_sections([ExtractedSection("text", None)], chunk_size=chunk_size)  # type: ignore[arg-type]


def test_rejects_empty_sections() -> None:
    with pytest.raises(ValueError, match="no text to chunk"):
        chunk_sections([ExtractedSection(" \t", None)])
