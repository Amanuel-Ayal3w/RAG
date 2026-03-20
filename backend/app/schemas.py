from datetime import datetime
from uuid import UUID

from pydantic import BaseModel, Field


class IngestDocument(BaseModel):
    source_id: str | None = None
    text: str = Field(min_length=1)
    metadata: dict = Field(default_factory=dict)


class IngestRequest(BaseModel):
    documents: list[IngestDocument] = Field(min_length=1)


class IngestResponse(BaseModel):
    inserted: int


class IngestedDocument(BaseModel):
    source_id: str
    chunks: int
    last_ingested_at: datetime


class ChatRequest(BaseModel):
    conversation_id: UUID | None = None
    message: str = Field(min_length=1)


class ChatMessage(BaseModel):
    role: str
    content: str
    created_at: datetime


class RetrievedChunk(BaseModel):
    source_id: str | None
    score: float
    content: str
    metadata: dict


class ChatResponse(BaseModel):
    conversation_id: UUID
    answer: str
    retrieved_chunks: list[RetrievedChunk]
    memory_messages: list[ChatMessage]
