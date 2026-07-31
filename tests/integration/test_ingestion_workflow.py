from __future__ import annotations

import os
from pathlib import Path

import pytest
from sqlalchemy import create_engine, text
from sqlalchemy.orm import Session, sessionmaker

from app.services.embeddings import EMBEDDING_DIMENSIONS
from app.services.ingestion import IngestionService


pytestmark = pytest.mark.migration_integration


def _require_integration() -> str:
    if os.environ.get("RUN_MIGRATION_INTEGRATION") != "1":
        pytest.skip(
            "set RUN_MIGRATION_INTEGRATION=1 to run ingestion integration tests"
        )
    database_url = os.environ.get("MIGRATION_TEST_DATABASE_URL", "").strip()
    if not database_url:
        pytest.fail("MIGRATION_TEST_DATABASE_URL must be set for ingestion tests")
    return database_url


class FakeEmbedder:
    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [[0.0] * EMBEDDING_DIMENSIONS for _ in texts]


def test_ingestion_persists_and_is_idempotent(tmp_path: Path) -> None:
    database_url = _require_integration()
    engine = create_engine(database_url)
    session_factory = sessionmaker[Session](bind=engine, expire_on_commit=False)
    path = tmp_path / "document.txt"
    path.write_text("initial content", encoding="utf-8")

    try:
        with engine.begin() as connection:
            connection.execute(text("DELETE FROM chunks"))
            connection.execute(text("DELETE FROM documents"))

        service = IngestionService(session_factory, FakeEmbedder(), 1000)
        assert service.ingest_folder(tmp_path).messages[0].status == "ingested"
        assert service.ingest_folder(tmp_path).messages[0].status == "skipped"

        path.write_text("changed content", encoding="utf-8")
        assert service.ingest_folder(tmp_path).messages[0].status == "ingested"

        with engine.connect() as connection:
            assert (
                connection.execute(text("SELECT count(*) FROM documents")).scalar_one()
                == 2
            )
            assert (
                connection.execute(text("SELECT count(*) FROM chunks")).scalar_one()
                == 2
            )
    finally:
        engine.dispose()
