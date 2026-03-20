from io import BytesIO

from docx import Document
from pypdf import PdfReader
from sqlalchemy import desc, func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import settings
from app.models import Conversation, DocumentChunk, Message
from app.schemas import ChatMessage, IngestDocument, IngestedDocument, RetrievedChunk
from app.services.openai_service import generate_chat_completion, get_embedding


def chunk_text(text: str, chunk_size: int = 900, overlap: int = 150) -> list[str]:
    cleaned = " ".join(text.split())
    if len(cleaned) <= chunk_size:
        return [cleaned]

    chunks: list[str] = []
    start = 0
    while start < len(cleaned):
        end = start + chunk_size
        chunks.append(cleaned[start:end])
        if end >= len(cleaned):
            break
        start = end - overlap
    return chunks


async def ingest_documents(session: AsyncSession, docs: list[IngestDocument]) -> int:
    rows_to_add: list[DocumentChunk] = []
    for doc in docs:
        chunks = chunk_text(doc.text)
        for chunk in chunks:
            embedding = await get_embedding(chunk)
            rows_to_add.append(
                DocumentChunk(
                    source_id=doc.source_id,
                    content=chunk,
                    chunk_metadata=doc.metadata,
                    embedding=embedding,
                )
            )

    session.add_all(rows_to_add)
    await session.commit()
    return len(rows_to_add)


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


def parse_document_bytes(filename: str, file_bytes: bytes) -> str:
    lower_name = filename.lower()
    if lower_name.endswith(".pdf"):
        return _extract_text_from_pdf(file_bytes)
    if lower_name.endswith(".docx"):
        return _extract_text_from_docx(file_bytes)
    if lower_name.endswith(".txt") or lower_name.endswith(".md"):
        return _extract_text_from_plain(file_bytes)
    raise ValueError("Unsupported file type. Use .txt, .md, .pdf, or .docx")


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
    query_embedding = await get_embedding(query)

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


async def run_chat_turn(session: AsyncSession, conversation_id, user_message: str):
    conversation = await get_or_create_conversation(session, conversation_id)

    user_row = Message(conversation_id=conversation.id, role="user", content=user_message)
    session.add(user_row)
    await session.flush()

    memory = await fetch_sliding_window_messages(session, conversation.id)
    retrieved = await retrieve_relevant_chunks(session, user_message)
    context = build_context(retrieved)

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

    llm_messages: list[dict[str, str]] = [{"role": "system", "content": system_prompt}]
    if context:
        llm_messages.append(
            {
                "role": "system",
                "content": f"Retrieved knowledge:\n{context}",
            }
        )

    llm_messages.extend(
        [{"role": message.role, "content": message.content} for message in memory]
    )

    answer = await generate_chat_completion(llm_messages)

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
