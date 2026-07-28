import pytest

from app.core.config import ConfigurationError, get_settings


def test_get_settings_requires_database_url(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("DATABASE_URL", raising=False)

    with pytest.raises(ConfigurationError, match="DATABASE_URL must be set"):
        get_settings()


def test_get_settings_reads_database_url(monkeypatch: pytest.MonkeyPatch) -> None:
    database_url = "postgresql+psycopg://rag:password@localhost:5432/rag"
    monkeypatch.setenv("DATABASE_URL", database_url)

    assert get_settings().database_url == database_url
