"""Idempotent document ingestion workflow."""

from __future__ import annotations

import hashlib
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

from sqlalchemy import select
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session

from app.db.models import Chunk, Document
from app.services.chunking import chunk_sections
from app.services.embeddings import Embedder, EmbeddingError, validate_embeddings
from app.services.extraction import DocumentExtractionError, extract_document


_MEDIA_TYPES = {".pdf": "application/pdf", ".txt": "text/plain"}


@dataclass(frozen=True)
class IngestionMessage:
    path: Path
    status: str
    detail: str


@dataclass
class IngestionReport:
    messages: list[IngestionMessage] = field(default_factory=list)


class IngestionService:
    """Processes each supported file in a folder in its own transaction."""

    def __init__(
        self,
        session_factory: Callable[[], Session],
        embedder: Embedder,
        chunk_size: int,
    ) -> None:
        self._session_factory = session_factory
        self._embedder = embedder
        self._chunk_size = chunk_size

    def ingest_folder(self, folder: Path) -> IngestionReport:
        report = IngestionReport()
        paths = sorted(
            (item for item in folder.iterdir() if item.is_file()),
            key=lambda item: item.name,
        )
        for path in paths:
            if path.suffix.lower() not in _MEDIA_TYPES:
                report.messages.append(
                    IngestionMessage(path, "unsupported", "Unsupported file type.")
                )
                continue
            report.messages.append(self._ingest_file(path))
        return report

    def _ingest_file(self, path: Path) -> IngestionMessage:
        try:
            checksum = hashlib.sha256(path.read_bytes()).hexdigest()
        except OSError:
            return IngestionMessage(path, "failed", "Unable to read document.")

        with self._session_factory() as session:
            try:
                existing = session.scalar(
                    select(Document.id).where(Document.content_checksum == checksum)
                )
                if existing is not None:
                    return IngestionMessage(
                        path, "skipped", "Unchanged document already ingested."
                    )

                chunks = chunk_sections(extract_document(path), self._chunk_size)
                chunk_texts = [chunk.chunk_text for chunk in chunks]
                embeddings = validate_embeddings(
                    chunk_texts, self._embedder.embed_documents(chunk_texts)
                )
                document = Document(
                    filename=path.name,
                    content_checksum=checksum,
                    media_type=_MEDIA_TYPES[path.suffix.lower()],
                    chunks=[
                        Chunk(
                            chunk_index=chunk.chunk_index,
                            page_number=chunk.page_number,
                            text=chunk.chunk_text,
                            embedding=embedding,
                        )
                        for chunk, embedding in zip(chunks, embeddings, strict=True)
                    ],
                )
                session.add(document)
                session.commit()
                return IngestionMessage(path, "ingested", "Document ingested.")
            except (DocumentExtractionError, EmbeddingError, ValueError) as error:
                session.rollback()
                return IngestionMessage(path, "failed", str(error))
            except SQLAlchemyError:
                session.rollback()
                return IngestionMessage(path, "failed", "Database operation failed.")
