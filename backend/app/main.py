from contextlib import asynccontextmanager

from fastapi import Depends, FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import settings
from app.db import engine, get_session
from app.models import Base
from app.schemas import (
    ChatRequest,
    ChatResponse,
    IngestDocument,
    IngestRequest,
    IngestResponse,
    IngestedDocument,
)
from app.services.rag_service import (
    IMAGE_MEDIA_TYPES,
    ingest_documents,
    list_ingested_documents,
    parse_document_bytes,
    run_chat_turn,
)


@asynccontextmanager
async def lifespan(_: FastAPI):
    async with engine.begin() as conn:
        await conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
        await conn.run_sync(Base.metadata.create_all)
        await conn.execute(
            text(
                "CREATE INDEX IF NOT EXISTS document_chunks_embedding_idx "
                "ON document_chunks USING ivfflat (embedding vector_cosine_ops) WITH (lists = 100)"
            )
        )
    yield


app = FastAPI(title=settings.app_name, lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origin_list,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post(f"{settings.api_prefix}/ingest", response_model=IngestResponse)
async def ingest_endpoint(
    payload: IngestRequest, session: AsyncSession = Depends(get_session)
) -> IngestResponse:
    inserted = await ingest_documents(session, payload.documents)
    return IngestResponse(inserted=inserted)


@app.get(f"{settings.api_prefix}/documents", response_model=list[IngestedDocument])
async def list_documents_endpoint(
    session: AsyncSession = Depends(get_session),
) -> list[IngestedDocument]:
    return await list_ingested_documents(session)


@app.post(f"{settings.api_prefix}/ingest/documents", response_model=IngestResponse)
async def ingest_documents_endpoint(
    files: list[UploadFile] = File(...), session: AsyncSession = Depends(get_session)
) -> IngestResponse:
    docs: list[IngestDocument] = []

    for file in files:
        if not file.filename:
            raise HTTPException(status_code=400, detail="One file is missing a filename")

        raw_bytes = await file.read()
        if not raw_bytes:
            continue

        try:
            # parse_document_bytes is now async (images need a vision API call)
            text_content = await parse_document_bytes(file.filename, raw_bytes)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

        if not text_content:
            continue

        # Determine source type for metadata
        lower_name = file.filename.lower()
        is_image = any(lower_name.endswith(ext) for ext in IMAGE_MEDIA_TYPES)
        source_type = "image_upload" if is_image else "upload"

        docs.append(
            IngestDocument(
                source_id=file.filename,
                text=text_content,
                metadata={"source": source_type, "filename": file.filename},
            )
        )

    if not docs:
        raise HTTPException(status_code=400, detail="No readable text found in uploaded files")

    inserted = await ingest_documents(session, docs)
    return IngestResponse(inserted=inserted)


@app.post(f"{settings.api_prefix}/chat", response_model=ChatResponse)
async def chat_endpoint(
    payload: ChatRequest, session: AsyncSession = Depends(get_session)
) -> ChatResponse:
    conversation_id, answer, retrieved_chunks, memory_messages = await run_chat_turn(
        session=session,
        conversation_id=payload.conversation_id,
        user_message=payload.message,
        image_b64=payload.image_base64,
        image_media_type=payload.image_media_type,
    )

    return ChatResponse(
        conversation_id=conversation_id,
        answer=answer,
        retrieved_chunks=retrieved_chunks,
        memory_messages=memory_messages,
    )
