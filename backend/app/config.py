from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    app_name: str = "RAG API"
    api_prefix: str = "/api/v1"

    database_url: str = Field(
        default="postgresql+asyncpg://rag:rag@localhost:5432/ragdb",
        alias="DATABASE_URL",
    )

    openai_api_key: str = Field(default="", alias="OPENAI_API_KEY")
    openai_chat_model: str = Field(default="gpt-4.1-mini", alias="OPENAI_CHAT_MODEL")
    openai_vision_model: str = Field(default="gpt-4o", alias="OPENAI_VISION_MODEL")
    openai_embedding_model: str = Field(
        default="text-embedding-3-small", alias="OPENAI_EMBEDDING_MODEL"
    )
    embedding_dimensions: int = Field(default=1536, alias="EMBEDDING_DIMENSIONS")

    top_k: int = Field(default=5, alias="RAG_TOP_K")
    memory_window: int = Field(default=8, alias="MEMORY_WINDOW")
    max_context_chars: int = Field(default=7000, alias="MAX_CONTEXT_CHARS")

    cors_origins: str = Field(default="http://localhost:3000", alias="CORS_ORIGINS")

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")

    @property
    def cors_origin_list(self) -> list[str]:
        return [origin.strip() for origin in self.cors_origins.split(",") if origin.strip()]


settings = Settings()
