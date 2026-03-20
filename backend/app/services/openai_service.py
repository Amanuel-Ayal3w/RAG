from openai import AsyncOpenAI

from app.config import settings


client = AsyncOpenAI(api_key=settings.openai_api_key)


async def get_embedding(text: str) -> list[float]:
    response = await client.embeddings.create(
        model=settings.openai_embedding_model,
        input=text,
        dimensions=settings.embedding_dimensions,
    )
    return response.data[0].embedding


async def generate_chat_completion(messages: list[dict[str, str]]) -> str:
    response = await client.chat.completions.create(
        model=settings.openai_chat_model,
        messages=messages,
        temperature=0.2,
    )
    return response.choices[0].message.content or ""
