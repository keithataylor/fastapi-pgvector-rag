from __future__ import annotations

import hashlib
import os
import uuid

import pytest
from sqlalchemy.exc import IntegrityError
from sqlalchemy import create_engine
from sqlalchemy.engine import URL, make_url
from sqlalchemy.orm import Session, sessionmaker

from app.core.config import get_settings
from app.db.models import Chunk, Document


pytestmark = pytest.mark.migration_integration


def _require_migration_integration() -> None:
    if os.environ.get("RUN_MIGRATION_INTEGRATION") != "1":
        pytest.skip(
            "set RUN_MIGRATION_INTEGRATION=1 to run persistence integration tests"
        )


def _test_checksum() -> str:
    return hashlib.sha256(uuid.uuid4().bytes).hexdigest()


def _migration_test_database_url() -> str:
    raw_url = os.environ.get("MIGRATION_TEST_DATABASE_URL", "").strip()
    if not raw_url:
        pytest.fail("MIGRATION_TEST_DATABASE_URL must be set for persistence tests")

    application_url = make_url(get_settings().database_url)
    migration_url = make_url(raw_url)
    if _database_identity(application_url) == _database_identity(migration_url):
        pytest.fail("MIGRATION_TEST_DATABASE_URL must target a different database")

    return raw_url


def _database_identity(database_url: URL) -> tuple[str | None, int | None, str | None]:
    return (database_url.host, database_url.port, database_url.database)


def test_document_and_chunk_can_be_persisted_and_loaded() -> None:
    _require_migration_integration()
    engine = create_engine(_migration_test_database_url())
    session_factory = sessionmaker[Session](bind=engine, expire_on_commit=False)
    checksum = _test_checksum()
    embedding = [0.0] * 1536

    try:
        with session_factory() as session:
            document = Document(
                filename="example.txt",
                content_checksum=checksum,
                media_type="text/plain",
                chunks=[
                    Chunk(
                        chunk_index=0,
                        page_number=0,
                        text="Example content",
                        embedding=embedding,
                    )
                ],
            )
            session.add(document)
            session.commit()
            document_id = document.id

            session.expunge_all()
            stored_document = session.get(Document, document_id)
            assert stored_document is not None
            assert stored_document.content_checksum == checksum
            assert len(stored_document.chunks) == 1
            assert stored_document.chunks[0].document_id == stored_document.id
            assert stored_document.chunks[0].chunk_index == 0
            assert stored_document.chunks[0].page_number == 0
            assert len(stored_document.chunks[0].embedding) == 1536

            session.add(
                Document(
                    filename="duplicate.txt",
                    content_checksum=checksum,
                    media_type="text/plain",
                )
            )
            with pytest.raises(IntegrityError):
                session.commit()
            session.rollback()

            session.add(
                Chunk(
                    document_id=document_id,
                    chunk_index=0,
                    page_number=0,
                    text="Duplicate chunk",
                    embedding=embedding,
                )
            )
            with pytest.raises(IntegrityError):
                session.commit()
            session.rollback()

            session.add(
                Chunk(
                    document_id=document_id,
                    chunk_index=-1,
                    page_number=0,
                    text="Negative chunk index",
                    embedding=embedding,
                )
            )
            with pytest.raises(IntegrityError):
                session.commit()
            session.rollback()

            session.add(
                Chunk(
                    document_id=document_id,
                    chunk_index=1,
                    page_number=-1,
                    text="Negative page number",
                    embedding=embedding,
                )
            )
            with pytest.raises(IntegrityError):
                session.commit()
            session.rollback()

            session.delete(stored_document)
            session.commit()
    finally:
        engine.dispose()
