import pytest

from app.core.config import ConfigurationError, get_chat_settings


def test_chat_settings_requires_chat_model(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.delenv("CHAT_MODEL", raising=False)

    with pytest.raises(ConfigurationError, match="CHAT_MODEL must be set"):
        get_chat_settings()


def test_chat_settings_uses_required_model(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setenv("CHAT_MODEL", "gpt-test")
    monkeypatch.setenv("RETRIEVAL_TOP_K", "3")

    settings = get_chat_settings()

    assert settings.openai_api_key == "test-key"
    assert settings.chat_model == "gpt-test"
    assert settings.retrieval_top_k == 3
