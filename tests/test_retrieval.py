from __future__ import annotations


import pytest

from app.services.embeddings import EMBEDDING_DIMENSIONS
from app.services.retrieval import RetrievalError, RetrievalService


class FakeQueryEmbedder:
    def __init__(self) -> None:
        self.calls = 0

    def embed_query(self, _: str) -> list[float]:
        self.calls += 1
        return [0.0] * EMBEDDING_DIMENSIONS


class FakeResult:
    def all(self) -> list[object]:
        return []


class FakeSession:
    def __enter__(self) -> FakeSession:
        return self

    def __exit__(self, *_: object) -> None:
        return None

    def execute(self, _: object) -> FakeResult:
        return FakeResult()


def test_empty_store_returns_empty_results() -> None:
    embedder = FakeQueryEmbedder()
    service = RetrievalService(FakeSession, embedder, default_top_k=5)

    assert service.retrieve("question") == []
    assert embedder.calls == 1


@pytest.mark.parametrize("top_k", [0, -1, True, "5"])
def test_invalid_top_k_is_rejected_before_embedding(top_k: object) -> None:
    embedder = FakeQueryEmbedder()
    service = RetrievalService(FakeSession, embedder, default_top_k=5)

    with pytest.raises(RetrievalError, match="top_k"):
        service.retrieve("question", top_k=top_k)  # type: ignore[arg-type]
    assert embedder.calls == 0
