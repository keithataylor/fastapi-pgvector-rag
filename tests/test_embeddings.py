from __future__ import annotations

import pytest

from app.services.embeddings import (
    EMBEDDING_DIMENSIONS,
    EmbeddingError,
    validate_embeddings,
)


def test_validate_embeddings_rejects_wrong_count() -> None:
    with pytest.raises(EmbeddingError, match="vector count"):
        validate_embeddings(["one"], [])


def test_validate_embeddings_rejects_invalid_dimension() -> None:
    with pytest.raises(EmbeddingError, match="vector dimension"):
        validate_embeddings(["one"], [[0.0]])


def test_openai_embedder_translates_provider_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FailingClient:
        def __init__(self, **_: object) -> None:
            pass

        def embed_documents(self, _: list[str]) -> list[list[float]]:
            raise RuntimeError("provider detail")

    monkeypatch.setattr("app.services.embeddings.OpenAIEmbeddings", FailingClient)
    from app.services.embeddings import OpenAIEmbedder

    with pytest.raises(EmbeddingError, match="Embedding provider failed"):
        OpenAIEmbedder("test-key").embed_documents(["text"])


def test_validate_embeddings_accepts_schema_dimension() -> None:
    vector = [0.0] * EMBEDDING_DIMENSIONS
    assert validate_embeddings(["one"], [vector]) == [vector]
