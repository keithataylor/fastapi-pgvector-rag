"""FastAPI application entry point."""

from __future__ import annotations
from typing import Annotated

from uuid import UUID

from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, StrictInt, field_validator
from sqlalchemy.exc import SQLAlchemyError

from app.core.config import ConfigurationError, get_chat_settings
from app.services.chat import (
    ChatProviderError,
    ChatService,
    EmptyCorpusError,
    OpenAIChatModel,
)
from app.services.embeddings import EmbeddingError, OpenAIEmbedder
from app.services.retrieval import RetrievalService


app = FastAPI()


class ChatRequest(BaseModel):
    question: str
    top_k: Annotated[StrictInt, Field(gt=0)] | None = None

    @field_validator("question")
    @classmethod
    def validate_question(cls, question: str) -> str:
        question = question.strip()
        if not question:
            raise ValueError("question must not be empty.")
        return question


class ChatSource(BaseModel):
    document_id: UUID
    filename: str
    page_number: int | None
    chunk_index: int
    text: str


class ChatResponse(BaseModel):
    answer: str
    sources: list[ChatSource]


@app.exception_handler(ConfigurationError)
async def configuration_error_handler(
    _: object, __: ConfigurationError
) -> JSONResponse:
    return JSONResponse(
        status_code=503,
        content={"detail": "Service configuration is unavailable."},
    )


def get_chat_service() -> ChatService:
    """Construct chat dependencies lazily so non-chat tooling needs no chat config."""
    settings = get_chat_settings()
    from app.db.session import SessionLocal

    retrieval = RetrievalService(
        session_factory=SessionLocal,
        embedder=OpenAIEmbedder(settings.openai_api_key),
        default_top_k=settings.retrieval_top_k,
    )
    return ChatService(
        retrieval=retrieval,
        chat_model=OpenAIChatModel(settings.openai_api_key, settings.chat_model),
    )


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/chat", response_model=ChatResponse)
def chat(
    request: ChatRequest,
) -> ChatResponse:
    service = get_chat_service()
    try:
        answer, sources = service.answer(request.question, request.top_k)
    except EmptyCorpusError as error:
        raise HTTPException(
            status_code=409,
            detail="No indexed source chunks are available.",
        ) from error
    except (EmbeddingError, ChatProviderError, SQLAlchemyError) as error:
        raise HTTPException(
            status_code=503,
            detail="Chat service is unavailable.",
        ) from error

    return ChatResponse(
        answer=answer,
        sources=[
            ChatSource(
                document_id=source.document_id,
                filename=source.filename,
                page_number=source.page_number,
                chunk_index=source.chunk_index,
                text=source.text,
            )
            for source in sources
        ],
    )
