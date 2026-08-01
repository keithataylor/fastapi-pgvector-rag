from __future__ import annotations

from uuid import UUID

import pytest
from fastapi.testclient import TestClient
from sqlalchemy.exc import SQLAlchemyError

import app.main as main
from app.services.chat import ChatProviderError, ChatService
from app.services.embeddings import EMBEDDING_DIMENSIONS, EmbeddingError
from app.services.retrieval import RetrievalResult, RetrievalService


class FakeRetrieval:
    def __init__(self, sources: list[RetrievalResult]) -> None:
        self.sources = sources
        self.calls: list[tuple[str, int | None]] = []

    def retrieve(self, query: str, top_k: int | None = None) -> list[RetrievalResult]:
        self.calls.append((query, top_k))
        return self.sources


class FakeChatModel:
    def __init__(self, answer: str = "Grounded answer") -> None:
        self.answer = answer
        self.calls = 0

    def invoke(self, _: object) -> str:
        self.calls += 1
        return self.answer


@pytest.fixture
def client() -> TestClient:
    return TestClient(main.app)


@pytest.fixture
def source() -> RetrievalResult:
    return RetrievalResult(
        document_id=UUID("00000000-0000-0000-0000-000000000001"),
        filename="agreement.pdf",
        page_number=7,
        chunk_index=18,
        text="Written notice is required.",
    )


def test_chat_returns_answer_and_unchanged_ranked_sources(
    monkeypatch: pytest.MonkeyPatch, client: TestClient, source: RetrievalResult
) -> None:
    second_source = RetrievalResult(
        document_id=UUID("00000000-0000-0000-0000-000000000002"),
        filename="appendix.txt",
        page_number=None,
        chunk_index=0,
        text="Additional context.",
    )
    retrieval = FakeRetrieval([source, second_source])
    model = FakeChatModel()
    monkeypatch.setattr(main, "get_chat_service", lambda: ChatService(retrieval, model))

    response = client.post(
        "/chat", json={"question": "  What is required?  ", "top_k": 2}
    )

    assert response.status_code == 200
    assert response.json() == {
        "answer": "Grounded answer",
        "sources": [
            {
                "document_id": str(source.document_id),
                "filename": "agreement.pdf",
                "page_number": 7,
                "chunk_index": 18,
                "text": "Written notice is required.",
            },
            {
                "document_id": str(second_source.document_id),
                "filename": "appendix.txt",
                "page_number": None,
                "chunk_index": 0,
                "text": "Additional context.",
            },
        ],
    }
    assert retrieval.calls == [("What is required?", 2)]
    assert model.calls == 1


@pytest.mark.parametrize(
    "payload",
    [
        {"question": "   "},
        {"question": "question", "top_k": True},
        {"question": "question", "top_k": 1.5},
        {"question": "question", "top_k": "5"},
        {"question": "question", "top_k": 0},
        {"question": "question", "top_k": -1},
    ],
)
def test_invalid_chat_request_does_not_invoke_services(
    monkeypatch: pytest.MonkeyPatch, client: TestClient, payload: dict[str, object]
) -> None:
    retrieval = FakeRetrieval([])
    model = FakeChatModel()
    factory_calls = 0

    def get_service() -> ChatService:
        nonlocal factory_calls
        factory_calls += 1
        return ChatService(retrieval, model)

    monkeypatch.setattr(main, "get_chat_service", get_service)

    response = client.post("/chat", json=payload)

    assert response.status_code == 422
    assert factory_calls == 0
    assert retrieval.calls == []
    assert model.calls == 0


def test_empty_corpus_returns_conflict_without_chat_call(
    monkeypatch: pytest.MonkeyPatch, client: TestClient
) -> None:
    retrieval = FakeRetrieval([])
    model = FakeChatModel()
    monkeypatch.setattr(main, "get_chat_service", lambda: ChatService(retrieval, model))

    response = client.post("/chat", json={"question": "question"})

    assert response.status_code == 409
    assert response.json() == {"detail": "No indexed source chunks are available."}
    assert model.calls == 0


def test_chat_provider_failure_returns_safe_service_error(
    monkeypatch: pytest.MonkeyPatch, client: TestClient, source: RetrievalResult
) -> None:
    class FailingModel:
        def invoke(self, _: object) -> str:
            raise ChatProviderError("provider detail")

    monkeypatch.setattr(
        main,
        "get_chat_service",
        lambda: ChatService(FakeRetrieval([source]), FailingModel()),
    )

    response = client.post("/chat", json={"question": "question"})

    assert response.status_code == 503
    assert response.json() == {"detail": "Chat service is unavailable."}


def test_embedding_failure_returns_safe_service_error(
    monkeypatch: pytest.MonkeyPatch, client: TestClient
) -> None:
    class FailingRetrieval:
        def retrieve(self, _: str, __: int | None = None) -> list[RetrievalResult]:
            raise EmbeddingError("provider detail")

    monkeypatch.setattr(
        main,
        "get_chat_service",
        lambda: ChatService(FailingRetrieval(), FakeChatModel()),
    )

    response = client.post("/chat", json={"question": "question"})

    assert response.status_code == 503
    assert response.json() == {"detail": "Chat service is unavailable."}


def test_missing_chat_configuration_returns_safe_service_error(
    monkeypatch: pytest.MonkeyPatch, client: TestClient
) -> None:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("CHAT_MODEL", raising=False)

    response = client.post("/chat", json={"question": "question"})

    assert response.status_code == 503
    assert response.json() == {"detail": "Service configuration is unavailable."}


def test_database_failure_returns_safe_service_error(
    monkeypatch: pytest.MonkeyPatch, client: TestClient
) -> None:
    class FailingRetrieval:
        def retrieve(self, _: str, __: int | None = None) -> list[RetrievalResult]:
            raise SQLAlchemyError("database detail")

    monkeypatch.setattr(
        main,
        "get_chat_service",
        lambda: ChatService(FailingRetrieval(), FakeChatModel()),
    )

    response = client.post("/chat", json={"question": "question"})

    assert response.status_code == 503
    assert response.json() == {"detail": "Chat service is unavailable."}


def test_chat_omitted_top_k_uses_retrieval_service_default(
    monkeypatch: pytest.MonkeyPatch, client: TestClient, source: RetrievalResult
) -> None:
    class FakeQueryEmbedder:
        def __init__(self) -> None:
            self.calls = 0

        def embed_query(self, query: str) -> list[float]:
            assert query == "question"
            self.calls += 1
            return [0.0] * EMBEDDING_DIMENSIONS

    class FakeChunk:
        document_id = source.document_id
        page_number = source.page_number
        chunk_index = source.chunk_index
        text = source.text

    class FakeResult:
        def all(self) -> list[tuple[FakeChunk, str]]:
            return [(FakeChunk(), source.filename)]

    class FakeSession:
        statement: object | None = None

        def __enter__(self) -> FakeSession:
            return self

        def __exit__(self, *_: object) -> None:
            return None

        def execute(self, statement: object) -> FakeResult:
            type(self).statement = statement
            return FakeResult()

    class RecordingRetrieval:
        def __init__(self) -> None:
            self.top_k: int | None = 1
            self.embedder = FakeQueryEmbedder()
            self.service = RetrievalService(FakeSession, self.embedder, default_top_k=3)

        def retrieve(
            self, query: str, top_k: int | None = None
        ) -> list[RetrievalResult]:
            self.top_k = top_k
            return self.service.retrieve(query, top_k)

    retrieval = RecordingRetrieval()
    model = FakeChatModel()
    monkeypatch.setattr(main, "get_chat_service", lambda: ChatService(retrieval, model))

    response = client.post("/chat", json={"question": "question"})

    assert response.status_code == 200
    assert response.json()["answer"] == "Grounded answer"
    assert response.json()["sources"] == [
        {
            "document_id": str(source.document_id),
            "filename": source.filename,
            "page_number": source.page_number,
            "chunk_index": source.chunk_index,
            "text": source.text,
        }
    ]
    assert retrieval.top_k is None
    assert retrieval.embedder.calls == 1
    assert FakeSession.statement is not None
    assert FakeSession.statement._limit_clause.value == 3
    assert model.calls == 1
