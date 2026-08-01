from __future__ import annotations

import pytest

from app.core.config import ConfigurationError, get_retrieval_settings


def test_retrieval_top_k_default_and_valid_value(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.delenv("RETRIEVAL_TOP_K", raising=False)
    assert get_retrieval_settings().retrieval_top_k == 5

    monkeypatch.setenv("RETRIEVAL_TOP_K", "3")
    assert get_retrieval_settings().retrieval_top_k == 3


@pytest.mark.parametrize("value", ["0", "-1", "invalid", "true"])
def test_retrieval_top_k_rejects_invalid_values(
    monkeypatch: pytest.MonkeyPatch, value: str
) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setenv("RETRIEVAL_TOP_K", value)

    with pytest.raises(ConfigurationError, match="RETRIEVAL_TOP_K"):
        get_retrieval_settings()
