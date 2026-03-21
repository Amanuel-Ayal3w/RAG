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


async def generate_chat_completion(messages: list[dict]) -> str:
    response = await client.chat.completions.create(
        model=settings.openai_chat_model,
        messages=messages,
        temperature=0.2,
    )
    return response.choices[0].message.content or ""


async def describe_image_base64(b64: str, media_type: str) -> str:
    """Send an image to the vision model and return a rich text description.

    Used during ingestion — the description is then chunked and embedded
    exactly like any other text document.
    """
    response = await client.chat.completions.create(
        model=settings.openai_vision_model,
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:{media_type};base64,{b64}"},
                    },
                    {
                        "type": "text",
                        "text": (
                            "Describe this image in full detail. "
                            "Include all visible text, data, labels, objects, colors, layout, "
                            "and any information that would help answer questions about it. "
                            "Write in plain prose."
                        ),
                    },
                ],
            }
        ],
        max_tokens=1024,
        temperature=0.2,
    )
    return response.choices[0].message.content or ""


async def generate_chat_completion_vision(
    messages: list[dict],
    image_b64: str,
    image_media_type: str,
) -> str:
    """Like generate_chat_completion but attaches an image to the last user turn.

    The image is injected into the final message so the vision model can see
    both the retrieved context (in system messages) and the user's image.
    """
    # Find the last user message and convert it to a multipart content block
    vision_messages: list[dict] = []
    for msg in messages:
        if msg["role"] == "user" and msg is messages[-1]:
            vision_messages.append(
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:{image_media_type};base64,{image_b64}"},
                        },
                        {"type": "text", "text": msg["content"]},
                    ],
                }
            )
        else:
            vision_messages.append(msg)

    response = await client.chat.completions.create(
        model=settings.openai_vision_model,
        messages=vision_messages,
        temperature=0.2,
    )
    return response.choices[0].message.content or ""
