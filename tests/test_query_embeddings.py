from __future__ import annotations

import pytest

from app.services.embeddings import EMBEDDING_DIMENSIONS, EmbeddingError, OpenAIEmbedder


class FakeClient:
    def __init__(self, result: list[float] | Exception) -> None:
        self.result = result
        self.queries: list[str] = []

    def embed_query(self, query: str) -> list[float]:
        self.queries.append(query)
        if isinstance(self.result, Exception):
            raise self.result
        return self.result


def _embedder(monkeypatch: pytest.MonkeyPatch, client: FakeClient) -> OpenAIEmbedder:
    monkeypatch.setattr("app.services.embeddings.OpenAIEmbeddings", lambda **_: client)
    return OpenAIEmbedder("test-key")


def test_embed_query_calls_client_and_returns_valid_vector(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client = FakeClient([0.0] * EMBEDDING_DIMENSIONS)

    assert (
        _embedder(monkeypatch, client).embed_query("question")
        == [0.0] * EMBEDDING_DIMENSIONS
    )
    assert client.queries == ["question"]


def test_embed_query_translates_provider_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client = FakeClient(RuntimeError("provider detail"))

    with pytest.raises(EmbeddingError, match="Embedding provider failed"):
        _embedder(monkeypatch, client).embed_query("question")


def test_embed_query_rejects_invalid_dimension(monkeypatch: pytest.MonkeyPatch) -> None:
    client = FakeClient([0.0])

    with pytest.raises(EmbeddingError, match="vector dimension"):
        _embedder(monkeypatch, client).embed_query("question")
