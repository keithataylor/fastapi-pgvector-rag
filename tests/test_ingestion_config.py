from __future__ import annotations

from pathlib import Path

import pytest

from app.core.config import (
    ConfigurationError,
    get_ingestion_settings,
)
from app.services.chunking import DEFAULT_CHUNK_SIZE


def test_ingestion_settings_require_api_key(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("DOCUMENT_FOLDER", str(tmp_path))
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    with pytest.raises(ConfigurationError, match="OPENAI_API_KEY must be set"):
        get_ingestion_settings()


def test_ingestion_settings_validate_folder_and_chunk_size(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setenv("DOCUMENT_FOLDER", str(tmp_path / "missing"))
    with pytest.raises(ConfigurationError, match="existing directory"):
        get_ingestion_settings()

    monkeypatch.setenv("DOCUMENT_FOLDER", str(tmp_path))
    monkeypatch.delenv("CHUNK_SIZE", raising=False)
    assert get_ingestion_settings().chunk_size == DEFAULT_CHUNK_SIZE

    monkeypatch.setenv("CHUNK_SIZE", "0")
    with pytest.raises(ConfigurationError, match="between"):
        get_ingestion_settings()
