"""Deterministic, page-aware plain-text chunking."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from app.services.extraction import ExtractedSection


DEFAULT_CHUNK_SIZE = 1000
MIN_CHUNK_SIZE = 1
MAX_CHUNK_SIZE = 10_000
_BOUNDARY_WINDOW_PERCENT = 20


@dataclass(frozen=True)
class ChunkData:
    """A chunk ready for later embedding and persistence."""

    chunk_index: int
    chunk_text: str
    page_number: int | None


def chunk_sections(
    sections: Sequence[ExtractedSection], chunk_size: int = DEFAULT_CHUNK_SIZE
) -> list[ChunkData]:
    """Split sections deterministically without crossing section boundaries."""
    _validate_chunk_size(chunk_size)
    chunks: list[ChunkData] = []
    for section in sections:
        normalized_text = _normalize_text(section.text)
        if not normalized_text:
            continue
        for chunk_text in _split_section(normalized_text, chunk_size):
            chunks.append(
                ChunkData(
                    chunk_index=len(chunks),
                    chunk_text=chunk_text,
                    page_number=section.page_number,
                )
            )
    if not chunks:
        raise ValueError("Document contains no text to chunk.")
    return chunks


def _validate_chunk_size(chunk_size: int) -> None:
    if isinstance(chunk_size, bool) or not isinstance(chunk_size, int):
        raise ValueError("chunk_size must be an integer.")
    if not MIN_CHUNK_SIZE <= chunk_size <= MAX_CHUNK_SIZE:
        raise ValueError(
            f"chunk_size must be between {MIN_CHUNK_SIZE} and {MAX_CHUNK_SIZE}."
        )


def _normalize_text(text: str) -> str:
    return text.replace("\r\n", "\n").replace("\r", "\n").strip()


def _split_section(text: str, chunk_size: int) -> list[str]:
    chunks: list[str] = []
    remaining = text
    while len(remaining) > chunk_size:
        split_at = _find_split_at(remaining, chunk_size)
        chunk = remaining[:split_at].rstrip()
        if chunk:
            chunks.append(chunk)
        remaining = remaining[split_at:].lstrip()
    if remaining:
        chunks.append(remaining)
    return chunks


def _find_split_at(text: str, chunk_size: int) -> int:
    boundary_start = chunk_size - (chunk_size * _BOUNDARY_WINDOW_PERCENT // 100)
    candidate = text[:chunk_size]
    for boundary in ("\n\n", "\n"):
        position = candidate.rfind(boundary, boundary_start)
        if position != -1:
            return position
    for position in range(chunk_size - 1, boundary_start - 1, -1):
        if candidate[position].isspace():
            return position
    return chunk_size
