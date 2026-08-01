"""Environment-backed application configuration."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from app.services.chunking import DEFAULT_CHUNK_SIZE, MAX_CHUNK_SIZE, MIN_CHUNK_SIZE


class ConfigurationError(RuntimeError):
    """Raised when required application configuration is unavailable."""


@dataclass(frozen=True)
class IngestionSettings:
    """Settings required only by the document-ingestion workflow."""

    document_folder: Path
    openai_api_key: str
    chunk_size: int


@dataclass(frozen=True)
class Settings:
    database_url: str


def get_settings() -> Settings:
    """Load the database URL required by database and migration commands."""
    database_url = os.environ.get("DATABASE_URL", "").strip()
    if not database_url:
        raise ConfigurationError("DATABASE_URL must be set.")

    return Settings(database_url=database_url)


def get_ingestion_settings() -> IngestionSettings:
    """Load the environment configuration required for document ingestion."""
    openai_api_key = _required_setting("OPENAI_API_KEY")
    document_folder = Path(_required_setting("DOCUMENT_FOLDER"))
    if not document_folder.is_dir():
        raise ConfigurationError("DOCUMENT_FOLDER must be an existing directory.")

    chunk_size = _chunk_size_setting()
    return IngestionSettings(
        document_folder=document_folder,
        openai_api_key=openai_api_key,
        chunk_size=chunk_size,
    )


def _required_setting(name: str) -> str:
    value = os.environ.get(name, "").strip()
    if not value:
        raise ConfigurationError(f"{name} must be set.")
    return value


def _chunk_size_setting() -> int:
    raw_chunk_size = os.environ.get("CHUNK_SIZE", "").strip()
    if not raw_chunk_size:
        return DEFAULT_CHUNK_SIZE
    try:
        chunk_size = int(raw_chunk_size)
    except ValueError as error:
        raise ConfigurationError("CHUNK_SIZE must be an integer.") from error
    if not MIN_CHUNK_SIZE <= chunk_size <= MAX_CHUNK_SIZE:
        raise ConfigurationError(
            f"CHUNK_SIZE must be between {MIN_CHUNK_SIZE} and {MAX_CHUNK_SIZE}."
        )
    return chunk_size


@dataclass(frozen=True)
class RetrievalSettings:
    """Settings required only by cosine retrieval."""

    openai_api_key: str
    retrieval_top_k: int


def get_retrieval_settings() -> RetrievalSettings:
    """Load the environment configuration required for retrieval."""
    return RetrievalSettings(
        openai_api_key=_required_setting("OPENAI_API_KEY"),
        retrieval_top_k=_top_k_setting(),
    )


def _top_k_setting() -> int:
    raw_top_k = os.environ.get("RETRIEVAL_TOP_K", "").strip()
    if not raw_top_k:
        return 5
    try:
        top_k = int(raw_top_k)
    except ValueError as error:
        raise ConfigurationError("RETRIEVAL_TOP_K must be an integer.") from error
    if top_k <= 0:
        raise ConfigurationError("RETRIEVAL_TOP_K must be greater than zero.")
    return top_k
