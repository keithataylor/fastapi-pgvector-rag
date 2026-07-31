"""Command-line entry point for document ingestion."""

from __future__ import annotations

from app.core.config import ConfigurationError, get_ingestion_settings
from app.db.session import SessionLocal
from app.services.embeddings import OpenAIEmbedder
from app.services.ingestion import IngestionService


def main() -> int:
    try:
        settings = get_ingestion_settings()
    except ConfigurationError as error:
        print(f"Configuration error: {error}")
        return 2

    report = IngestionService(
        session_factory=SessionLocal,
        embedder=OpenAIEmbedder(settings.openai_api_key),
        chunk_size=settings.chunk_size,
    ).ingest_folder(settings.document_folder)
    for message in report.messages:
        print(f"{message.status}: {message.path} - {message.detail}")
    return 1 if any(message.status == "failed" for message in report.messages) else 0


if __name__ == "__main__":
    raise SystemExit(main())
