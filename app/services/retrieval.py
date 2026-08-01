"""Cosine-similarity chunk retrieval."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Protocol
from uuid import UUID

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.db.models import Chunk, Document
from app.services.embeddings import EmbeddingError, validate_embeddings


class RetrievalError(ValueError):
    """Raised when a retrieval request is invalid."""


class QueryEmbedder(Protocol):
    """Narrow provider contract required by retrieval."""

    def embed_query(self, query: str) -> list[float]: ...


@dataclass(frozen=True)
class RetrievalResult:
    document_id: UUID
    filename: str
    page_number: int | None
    chunk_index: int
    text: str


class RetrievalService:
    """Retrieves source chunks ordered by pgvector cosine distance."""

    def __init__(
        self,
        session_factory: Callable[[], Session],
        embedder: QueryEmbedder,
        default_top_k: int,
    ) -> None:
        self._session_factory = session_factory
        self._embedder = embedder
        self._default_top_k = _validate_top_k(default_top_k)

    def retrieve(self, query: str, top_k: int | None = None) -> list[RetrievalResult]:
        limit = self._default_top_k if top_k is None else _validate_top_k(top_k)
        try:
            embedding = validate_embeddings(
                [query], [self._embedder.embed_query(query)]
            )[0]
        except EmbeddingError:
            raise

        statement = build_retrieval_statement(embedding, limit)
        with self._session_factory() as session:
            rows = session.execute(statement).all()
        return [
            RetrievalResult(
                document_id=chunk.document_id,
                filename=filename,
                page_number=chunk.page_number,
                chunk_index=chunk.chunk_index,
                text=chunk.text,
            )
            for chunk, filename in rows
        ]


def _validate_top_k(top_k: int) -> int:
    if isinstance(top_k, bool) or not isinstance(top_k, int) or top_k <= 0:
        raise RetrievalError("top_k must be an integer greater than zero.")
    return top_k


def build_retrieval_statement(query_embedding: list[float], top_k: int):
    """Build the pgvector-index-eligible retrieval statement."""
    distance = Chunk.embedding.cosine_distance(query_embedding)
    return (
        select(Chunk, Document.filename)
        .join(Document, Chunk.document_id == Document.id)
        .order_by(distance)
        .limit(top_k)
    )
