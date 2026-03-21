import base64
import re
from io import BytesIO

from docx import Document
from pypdf import PdfReader
from sqlalchemy import desc, func, select
from sqlalchemy.ext.asyncio import AsyncSession

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate

from app.config import settings
from app.models import Conversation, DocumentChunk, Message
from app.schemas import ChatMessage, IngestDocument, IngestedDocument, RetrievedChunk
from app.services.openai_service import (
    describe_image_base64,
    chat_llm,
    embeddings,
)

from langchain_text_splitters import RecursiveCharacterTextSplitter

# Supported image media types for ingestion
IMAGE_MEDIA_TYPES: dict[str, str] = {
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".webp": "image/webp",
    ".gif": "image/gif",
}

def chunk_text(text: str) -> list[str]:
    """Split *text* into chunks using LangChain's RecursiveCharacterTextSplitter.

    This attempts to split on paragraphs ('\n\n') first, then on sentences ('. '),
    then on words (' '), guaranteeing chunks under the max size while preserving
    semantic boundaries.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1200,
        chunk_overlap=150,
        length_function=len,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    return splitter.split_text(text)


async def ingest_documents(session: AsyncSession, docs: list[IngestDocument]) -> int:
    total_chunks = 0
    for doc in docs:
        chunks = chunk_text(doc.text)
        if not chunks:
            continue

        # Use LangChain to batch-embed all chunks for this document
        chunk_embeddings = await embeddings.aembed_documents(chunks)

        rows_to_add: list[DocumentChunk] = []
        for chunk_content, chunk_embedding in zip(chunks, chunk_embeddings):
            rows_to_add.append(
                DocumentChunk(
                    source_id=doc.source_id,
                    content=chunk_content,
                    chunk_metadata=doc.metadata,
                    embedding=chunk_embedding,
                )
            )

        session.add_all(rows_to_add)
        total_chunks += len(rows_to_add)

    await session.commit()
    return total_chunks





async def list_ingested_documents(session: AsyncSession) -> list[IngestedDocument]:
    stmt = (
        select(
            DocumentChunk.source_id,
            func.count(DocumentChunk.id).label("chunks"),
            func.max(DocumentChunk.created_at).label("last_ingested_at"),
        )
        .where(DocumentChunk.source_id.is_not(None))
        .group_by(DocumentChunk.source_id)
        .order_by(desc("last_ingested_at"))
        .limit(100)
    )

    result = await session.execute(stmt)
    rows = result.all()

    return [
        IngestedDocument(
            source_id=source_id,
            chunks=chunks,
            last_ingested_at=last_ingested_at,
        )
        for source_id, chunks, last_ingested_at in rows
        if source_id and last_ingested_at
    ]


def _extract_text_from_pdf(file_bytes: bytes) -> str:
    reader = PdfReader(BytesIO(file_bytes))
    return "\n".join((page.extract_text() or "") for page in reader.pages).strip()


def _extract_text_from_docx(file_bytes: bytes) -> str:
    document = Document(BytesIO(file_bytes))
    return "\n".join(p.text for p in document.paragraphs).strip()


def _extract_text_from_plain(file_bytes: bytes) -> str:
    return file_bytes.decode("utf-8", errors="ignore").strip()


async def _extract_text_from_image(file_bytes: bytes, media_type: str) -> str:
    """Base64-encode image bytes and get a rich text description via the vision model."""
    b64 = base64.b64encode(file_bytes).decode("utf-8")
    return await describe_image_base64(b64, media_type)


async def parse_document_bytes(filename: str, file_bytes: bytes) -> str:
    """Parse a document file by extension. Returns extracted text content.

    Supports: .txt, .md, .pdf, .docx, .png, .jpg, .jpeg, .webp, .gif
    Image files are described by the vision model and the description is returned.
    """
    lower_name = filename.lower()
    if lower_name.endswith(".pdf"):
        return _extract_text_from_pdf(file_bytes)
    if lower_name.endswith(".docx"):
        return _extract_text_from_docx(file_bytes)
    if lower_name.endswith((".txt", ".md")):
        return _extract_text_from_plain(file_bytes)

    for ext, media_type in IMAGE_MEDIA_TYPES.items():
        if lower_name.endswith(ext):
            return await _extract_text_from_image(file_bytes, media_type)

    raise ValueError("Unsupported file type. Use .txt, .md, .pdf, .docx, .png, .jpg, .jpeg, .webp, or .gif")


async def get_or_create_conversation(
    session: AsyncSession, conversation_id
) -> Conversation:
    if conversation_id:
        conversation = await session.get(Conversation, conversation_id)
        if conversation:
            return conversation

    conversation = Conversation()
    session.add(conversation)
    await session.flush()
    return conversation


async def fetch_sliding_window_messages(
    session: AsyncSession, conversation_id
) -> list[Message]:
    stmt = (
        select(Message)
        .where(Message.conversation_id == conversation_id)
        .order_by(Message.created_at.desc())
        .limit(settings.memory_window)
    )
    result = await session.execute(stmt)
    rows = result.scalars().all()
    rows.reverse()
    return rows


async def retrieve_relevant_chunks(
    session: AsyncSession, query: str
) -> list[tuple[DocumentChunk, float]]:
    query_embedding = await embeddings.aembed_query(query)

    stmt = (
        select(
            DocumentChunk,
            DocumentChunk.embedding.cosine_distance(query_embedding).label("distance"),
        )
        .order_by("distance")
        .limit(settings.top_k)
    )

    result = await session.execute(stmt)
    return result.all()


def build_context(chunks: list[tuple[DocumentChunk, float]]) -> str:
    parts: list[str] = []
    running = 0
    for idx, (chunk, distance) in enumerate(chunks, start=1):
        entry = (
            f"[Chunk {idx}] source={chunk.source_id or 'unknown'} score={1 - float(distance):.4f}\n"
            f"{chunk.content}\n"
        )
        if running + len(entry) > settings.max_context_chars:
            break
        parts.append(entry)
        running += len(entry)
    return "\n".join(parts)


async def run_chat_turn(
    session: AsyncSession,
    conversation_id,
    user_message: str,
    image_b64: str | None = None,
    image_media_type: str | None = None,
):
    conversation = await get_or_create_conversation(session, conversation_id)

    user_row = Message(conversation_id=conversation.id, role="user", content=user_message)
    session.add(user_row)
    await session.flush()

    memory = await fetch_sliding_window_messages(session, conversation.id)
    retrieved = await retrieve_relevant_chunks(session, user_message)
    context = build_context(retrieved)

    # Construct the base system instructions
    system_prompt = (
        "You are a professional RAG assistant. "
        "Write clear, concise responses with a brief direct answer first, then structured details. "
        "Use short bullet points when listing key ideas. "
        "Do not use markdown markers like **, #, or code fences. "
        "Do not use LaTeX notation such as \\[ ... \\], \\( ... \\), or \\text{...}. "
        "Return plain readable text only. "
        "Use retrieved context when relevant and avoid fabricating facts. "
        "If context is insufficient, state that clearly and provide best-effort guidance."
    )

    # Convert past string-based history into LangChain message objects
    langchain_messages = [SystemMessage(content=system_prompt)]

    if context:
        langchain_messages.append(
            SystemMessage(content=f"Retrieved knowledge:\n{context}")
        )

    for msg in memory:
        if msg.role == "user":
            langchain_messages.append(HumanMessage(content=msg.content))
        elif msg.role == "assistant":
            langchain_messages.append(AIMessage(content=msg.content))

    # Append the current user message, handling potential image attachments
    if image_b64 and image_media_type:
        # LangChain vision format
        vision_content = [
            {"type": "text", "text": user_message},
            {
                "type": "image_url",
                "image_url": {"url": f"data:{image_media_type};base64,{image_b64}"},
            },
        ]
        langchain_messages.append(HumanMessage(content=vision_content))
    else:
        langchain_messages.append(HumanMessage(content=user_message))

    # Invoke the LangChain ChatOpenAI instance
    response = await chat_llm.ainvoke(langchain_messages)
    answer = str(response.content)

    assistant_row = Message(conversation_id=conversation.id, role="assistant", content=answer)
    session.add(assistant_row)
    await session.commit()
    await session.refresh(assistant_row)

    memory_payload = [
        ChatMessage(role=message.role, content=message.content, created_at=message.created_at)
        for message in memory
    ]

    retrieved_payload = [
        RetrievedChunk(
            source_id=chunk.source_id,
            score=max(0.0, min(1.0, 1 - float(distance))),
            content=chunk.content,
            metadata=chunk.chunk_metadata,
        )
        for chunk, distance in retrieved
    ]

    return conversation.id, answer, retrieved_payload, memory_payload
