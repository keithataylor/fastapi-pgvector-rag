"""Grounded LangChain chat generation."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol

from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from pydantic import SecretStr

from app.services.retrieval import RetrievalResult


class ChatProviderError(RuntimeError):
    """Raised when the chat provider cannot produce a usable answer."""


class ChatModel(Protocol):
    """Narrow model contract used by grounded chat generation."""

    def invoke(self, messages: Sequence[BaseMessage]) -> str: ...


class Retrieval(Protocol):
    """Narrow retrieval contract used by the chat workflow."""

    def retrieve(
        self, query: str, top_k: int | None = None
    ) -> list[RetrievalResult]: ...


class EmptyCorpusError(RuntimeError):
    """Raised when no indexed chunks are available for a chat request."""


class OpenAIChatModel:
    """LangChain adapter for the configured OpenAI-compatible chat model."""

    def __init__(self, api_key: str, model: str) -> None:
        self._client = ChatOpenAI(model=model, api_key=SecretStr(api_key))

    def invoke(self, messages: Sequence[BaseMessage]) -> str:
        try:
            response = self._client.invoke(list(messages))
        except Exception as error:
            raise ChatProviderError("Chat provider failed.") from error

        content = getattr(response, "content", None)
        if not isinstance(content, str) or not content.strip():
            raise ChatProviderError("Chat provider returned an unusable answer.")
        return content.strip()


class ChatService:
    """Retrieves grounded context and generates a chat answer."""

    def __init__(self, retrieval: Retrieval, chat_model: ChatModel) -> None:
        self._retrieval = retrieval
        self._chat_model = chat_model

    def answer(
        self, question: str, top_k: int | None = None
    ) -> tuple[str, list[RetrievalResult]]:
        sources = self._retrieval.retrieve(question, top_k)
        if not sources:
            raise EmptyCorpusError("No indexed source chunks are available.")
        return self._chat_model.invoke(_messages(question, sources)), sources


def _messages(question: str, sources: Sequence[RetrievalResult]) -> list[BaseMessage]:
    context = "\n\n".join(
        (
            f"Source {position}: filename={source.filename}; "
            f"document_id={source.document_id}; page_number={source.page_number}; "
            f"chunk_index={source.chunk_index}\n{source.text}"
        )
        for position, source in enumerate(sources, start=1)
    )
    return [
        SystemMessage(
            content=(
                "Answer only from the supplied source context. If the context is "
                "insufficient to answer the question, say that you cannot determine "
                "the answer from the supplied context."
            )
        ),
        HumanMessage(content=f"Question:\n{question}\n\nSource context:\n{context}"),
    ]
