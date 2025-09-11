#!/usr/bin/env python3
import os
from functools import lru_cache
from typing import Any, Dict, List, Literal
from pydantic import model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

BASE_PATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=f"{BASE_PATH}/.env",
        env_file_encoding="utf-8",
        extra="ignore", 
        case_sensitive=True,
        env_nested_delimiter="__"
    )

    # Environment
    ENVIRONMENT: Literal["dev", "pro"]

    # OpenAI
    OPENAI_API_KEY: str
    # LLM configuration
    LLM_MODEL: str = "gpt-4o-mini"
    LLM_TEMPERATURE_DEFAULT: float = 0.4
    LLM_TEMPERATURE_LEGAL: float = 0.2
    LLM_MAX_TOKENS: int = 2000
    LLM_PRESENCE_PENALTY: float = 0.1
    LLM_FREQUENCY_PENALTY: float = 0.1

    # Qdrant
    QDRANT_HOST: str = "localhost"
    QDRANT_PORT: int = 6333
    QDRANT_COLLECTION: str = "law_documents_enhanced"

    # Embeddings
    EMBEDDING_MODEL: str = "text-embedding-3-small"

    # CORS
    CORS_ALLOWED_ORIGINS: List[str] = [
        "http://127.0.0.1:8000",
        "http://localhost:5173",
    ]
    CORS_EXPOSE_HEADERS: List[str] = ["X-Request-ID"]

    # FastAPI
    FASTAPI_API_V1_PATH: str = "/api/v1"

    @model_validator(mode="before")
    @classmethod
    def check_env(cls, values: Any) -> Dict[str, Any]:
        """Validate and modify settings based on environment."""
        if not isinstance(values, dict):
            return values
            
        if values.get("ENVIRONMENT") == "pro":
            values["FASTAPI_OPENAPI_URL"] = None
            values["FASTAPI_STATIC_FILES"] = False
            values["CELERY_BROKER"] = "rabbitmq"
        return values


@lru_cache
def get_settings() -> Settings:
    return Settings()

# Global config instance
settings = get_settings()
