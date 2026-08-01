from __future__ import annotations

import hashlib
import os

from collections.abc import Mapping
from typing import TypeGuard
import pytest
from sqlalchemy import create_engine, text
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy.dialects import postgresql

from app.db.models import Chunk, Document
from app.services.embeddings import EMBEDDING_DIMENSIONS
from app.services.retrieval import RetrievalService, build_retrieval_statement


pytestmark = pytest.mark.migration_integration


def _database_url() -> str:
    if os.environ.get("RUN_MIGRATION_INTEGRATION") != "1":
        pytest.skip(
            "set RUN_MIGRATION_INTEGRATION=1 to run retrieval integration tests"
        )
    value = os.environ.get("MIGRATION_TEST_DATABASE_URL", "").strip()
    if not value:
        pytest.fail("MIGRATION_TEST_DATABASE_URL must be set for retrieval tests")
    return value


def _vector(*values: float) -> list[float]:
    return [*values, *([0.0] * (EMBEDDING_DIMENSIONS - len(values)))]


class FakeQueryEmbedder:
    def embed_query(self, query: str) -> list[float]:
        return _vector(1.0)


def test_cosine_ranking_and_hnsw_query_plan() -> None:
    engine = create_engine(_database_url())
    session_factory = sessionmaker[Session](bind=engine, expire_on_commit=False)
    try:
        with session_factory() as session:
            session.execute(text("DELETE FROM chunks"))
            session.execute(text("DELETE FROM documents"))
            session.add_all(
                [
                    Document(
                        filename="first.txt",
                        content_checksum=hashlib.sha256(b"first").hexdigest(),
                        media_type="text/plain",
                        chunks=[
                            Chunk(
                                chunk_index=0,
                                page_number=None,
                                text="first",
                                embedding=_vector(1.0),
                            )
                        ],
                    ),
                    Document(
                        filename="second.txt",
                        content_checksum=hashlib.sha256(b"second").hexdigest(),
                        media_type="text/plain",
                        chunks=[
                            Chunk(
                                chunk_index=0,
                                page_number=None,
                                text="second",
                                embedding=_vector(0.5, 0.5),
                            )
                        ],
                    ),
                    Document(
                        filename="third.txt",
                        content_checksum=hashlib.sha256(b"third").hexdigest(),
                        media_type="text/plain",
                        chunks=[
                            Chunk(
                                chunk_index=0,
                                page_number=None,
                                text="third",
                                embedding=_vector(0.0, 1.0),
                            )
                        ],
                    ),
                ]
            )
            session.commit()

        results = RetrievalService(session_factory, FakeQueryEmbedder(), 2).retrieve(
            "query"
        )
        assert [result.text for result in results] == ["first", "second"]
        assert [result.filename for result in results] == ["first.txt", "second.txt"]

        statement = build_retrieval_statement(_vector(1.0), 2)
        compiled = statement.compile(
            dialect=postgresql.dialect(), compile_kwargs={"literal_binds": True}
        )
        with engine.begin() as connection:
            connection.execute(text("SET LOCAL enable_seqscan = off"))
            plan = connection.execute(
                text(f"EXPLAIN (FORMAT JSON) {compiled}"),
            ).scalar_one()[0]["Plan"]
        assert _is_plan(plan)
        assert _uses_hnsw_index(plan)
    finally:
        engine.dispose()


def _is_plan(value: object) -> TypeGuard[Mapping[str, object]]:
    return isinstance(value, Mapping)


def _uses_hnsw_index(plan: Mapping[str, object]) -> bool:
    if plan.get("Index Name") == "ix_chunks_embedding_hnsw_cosine":
        return True
    children = plan.get("Plans")
    if not isinstance(children, list):
        return False
    for child in children:
        if _is_plan(child) and _uses_hnsw_index(child):
            return True
    return False
