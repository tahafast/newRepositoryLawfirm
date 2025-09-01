#!/usr/bin/env python3
import os
from functools import lru_cache
from typing import Any, Literal
from pydantic import model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

BASE_PATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=f"{BASE_PATH}/.env",
        env_file_encoding="utf-8",
        extra="ignore", 
        case_sensitive=True,
    )

    # Environment
    ENVIRONMENT: Literal["dev", "pro"]

    # OpenAI
    OPENAI_API_KEY: str

    # CORS
    CORS_ALLOWED_ORIGINS: list[str] = [
        "http://127.0.0.1:8000",
        "http://localhost:5173",
    ]
    CORS_EXPOSE_HEADERS: list[str] = ["X-Request-ID"]

    # FastAPI
    FASTAPI_API_V1_PATH: str = "/api/v1"

    @model_validator(mode="before")
    @classmethod
    def check_env(cls, values: Any) -> Any:
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
