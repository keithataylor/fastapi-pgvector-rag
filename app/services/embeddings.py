"""Fixed-model embedding integration."""

from __future__ import annotations

from pydantic import SecretStr

import math
from collections.abc import Sequence
from typing import Protocol

from langchain_openai import OpenAIEmbeddings


EMBEDDING_MODEL = "text-embedding-3-small"
EMBEDDING_DIMENSIONS = 1536


class EmbeddingError(RuntimeError):
    """Raised when embeddings cannot be produced or validated."""


class Embedder(Protocol):
    """Protocol used by ingestion without coupling it to a provider."""

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]: ...


class OpenAIEmbedder:
    """LangChain adapter for the fixed OpenAI embedding baseline."""

    def __init__(self, api_key: str) -> None:
        self._client = OpenAIEmbeddings(
            model=EMBEDDING_MODEL,
            api_key=SecretStr(api_key),
        )

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        try:
            embeddings = self._client.embed_documents(list(texts))
        except Exception as error:
            raise EmbeddingError("Embedding provider failed.") from error
        return validate_embeddings(texts, embeddings)


def validate_embeddings(
    texts: Sequence[str], embeddings: Sequence[Sequence[float]]
) -> list[list[float]]:
    """Ensure provider output matches the immutable database vector schema."""
    if len(embeddings) != len(texts):
        raise EmbeddingError("Embedding provider returned an unexpected vector count.")
    validated: list[list[float]] = []
    for embedding in embeddings:
        if len(embedding) != EMBEDDING_DIMENSIONS:
            raise EmbeddingError(
                "Embedding provider returned an invalid vector dimension."
            )
        vector: list[float] = []
        for value in embedding:
            if isinstance(value, bool) or not isinstance(value, int | float):
                raise EmbeddingError(
                    "Embedding provider returned a non-numeric vector."
                )
            numeric_value = float(value)
            if not math.isfinite(numeric_value):
                raise EmbeddingError("Embedding provider returned a non-finite vector.")
            vector.append(numeric_value)
        validated.append(vector)
    return validated
