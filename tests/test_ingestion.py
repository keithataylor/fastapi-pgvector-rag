from __future__ import annotations

from pathlib import Path

from app.services.embeddings import EMBEDDING_DIMENSIONS, EmbeddingError
from app.services.ingestion import IngestionService


class FakeSession:
    def __init__(self, persisted: list[object]) -> None:
        self.persisted = persisted
        self.added: list[object] = []
        self.rollback_called = False

    def __enter__(self) -> FakeSession:
        return self

    def __exit__(self, *_: object) -> None:
        return None

    def scalar(self, _: object) -> None:
        return None

    def add(self, document: object) -> None:
        self.added.append(document)

    def commit(self) -> None:
        self.persisted.extend(self.added)

    def rollback(self) -> None:
        self.rollback_called = True


class FailingSecondEmbedder:
    def __init__(self) -> None:
        self.calls = 0

    def embed_documents(self, texts: object) -> list[list[float]]:
        self.calls += 1
        if self.calls == 2:
            raise EmbeddingError("Embedding provider failed.")
        return [[0.0] * EMBEDDING_DIMENSIONS for _ in texts]  # type: ignore[union-attr]


def test_embedding_failure_does_not_attempt_persistence_and_keeps_prior_document(
    tmp_path: Path,
) -> None:
    (tmp_path / "a.txt").write_text("first", encoding="utf-8")
    (tmp_path / "b.txt").write_text("second", encoding="utf-8")
    persisted: list[object] = []
    sessions: list[FakeSession] = []

    def session_factory() -> FakeSession:
        session = FakeSession(persisted)
        sessions.append(session)
        return session

    report = IngestionService(
        session_factory, FailingSecondEmbedder(), 1000
    ).ingest_folder(tmp_path)

    assert [message.status for message in report.messages] == ["ingested", "failed"]
    assert len(persisted) == 1
    assert sessions[1].added == []
    assert sessions[1].rollback_called


class InvalidEmbedder:
    def embed_documents(self, _: object) -> list[list[float]]:
        return [[0.0]]


def test_invalid_embedding_result_is_not_persisted(tmp_path: Path) -> None:
    (tmp_path / "invalid.txt").write_text("content", encoding="utf-8")
    persisted: list[object] = []
    sessions: list[FakeSession] = []

    def session_factory() -> FakeSession:
        session = FakeSession(persisted)
        sessions.append(session)
        return session

    report = IngestionService(session_factory, InvalidEmbedder(), 1000).ingest_folder(
        tmp_path
    )

    assert report.messages[0].status == "failed"
    assert persisted == []
    assert sessions[0].added == []
    assert sessions[0].rollback_called


class ExistingDocumentSession(FakeSession):
    def scalar(self, _: object) -> object:
        return object()


class RecordingEmbedder:
    def __init__(self) -> None:
        self.calls = 0

    def embed_documents(self, _: object) -> list[list[float]]:
        self.calls += 1
        return []


def test_duplicate_checksum_skips_embedding(tmp_path: Path) -> None:
    (tmp_path / "duplicate.txt").write_text("unchanged", encoding="utf-8")
    persisted: list[object] = []
    embedder = RecordingEmbedder()

    def session_factory() -> ExistingDocumentSession:
        return ExistingDocumentSession(persisted)

    report = IngestionService(session_factory, embedder, 1000).ingest_folder(tmp_path)

    assert report.messages[0].status == "skipped"
    assert embedder.calls == 0
