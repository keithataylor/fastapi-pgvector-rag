from __future__ import annotations

from uuid import uuid4

import pytest

from app.services.chat import (
    ChatProviderError,
    ChatService,
    EmptyCorpusError,
    OpenAIChatModel,
)
from app.services.retrieval import RetrievalResult


def _source() -> RetrievalResult:
    return RetrievalResult(
        document_id=uuid4(),
        filename="agreement.pdf",
        page_number=7,
        chunk_index=18,
        text="Written notice is required.",
    )


class FakeRetrieval:
    def __init__(self, sources: list[RetrievalResult]) -> None:
        self.sources = sources
        self.calls: list[tuple[str, int | None]] = []

    def retrieve(self, query: str, top_k: int | None = None) -> list[RetrievalResult]:
        self.calls.append((query, top_k))
        return self.sources


class FakeChatModel:
    def __init__(self) -> None:
        self.messages: list[object] | None = None

    def invoke(self, messages: list[object]) -> str:
        self.messages = messages
        return "Grounded answer"


def test_chat_service_uses_ranked_sources_as_grounded_context() -> None:
    source = _source()
    retrieval = FakeRetrieval([source])
    model = FakeChatModel()

    answer, sources = ChatService(retrieval, model).answer("What is required?", 2)

    assert answer == "Grounded answer"
    assert sources == [source]
    assert retrieval.calls == [("What is required?", 2)]
    assert model.messages is not None
    assert "Answer only from the supplied source context" in model.messages[0].content
    assert "What is required?" in model.messages[1].content
    assert source.text in model.messages[1].content


def test_empty_retrieval_does_not_call_chat_model() -> None:
    retrieval = FakeRetrieval([])
    model = FakeChatModel()

    with pytest.raises(EmptyCorpusError):
        ChatService(retrieval, model).answer("question")

    assert model.messages is None


def test_openai_chat_model_rejects_unusable_provider_answer(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class EmptyResponse:
        content = "   "

    class FakeClient:
        def __init__(self, **_: object) -> None:
            pass

        def invoke(self, _: list[object]) -> EmptyResponse:
            return EmptyResponse()

    monkeypatch.setattr("app.services.chat.ChatOpenAI", FakeClient)

    with pytest.raises(ChatProviderError, match="unusable answer"):
        OpenAIChatModel("test-key", "gpt-test").invoke([])


def test_openai_chat_model_translates_provider_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FakeClient:
        def __init__(self, **_: object) -> None:
            pass

        def invoke(self, _: list[object]) -> object:
            raise RuntimeError("provider detail")

    monkeypatch.setattr("app.services.chat.ChatOpenAI", FakeClient)

    with pytest.raises(ChatProviderError, match="Chat provider failed"):
        OpenAIChatModel("test-key", "gpt-test").invoke([])


def test_openai_chat_model_returns_valid_answer_and_uses_configured_model(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: dict[str, object] = {}

    class Response:
        content = "  Grounded answer  "

    class FakeClient:
        def __init__(self, **kwargs: object) -> None:
            calls.update(kwargs)

        def invoke(self, messages: list[object]) -> Response:
            calls["messages"] = messages
            return Response()

    monkeypatch.setattr("app.services.chat.ChatOpenAI", FakeClient)

    answer = OpenAIChatModel("test-key", "gpt-test").invoke([])

    assert answer == "Grounded answer"
    assert calls["model"] == "gpt-test"
    assert calls["messages"] == []
