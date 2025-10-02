"""
Core Configuration and Services
Consolidated configuration settings and service wiring for the Law Firm Chatbot
"""

import logging
from typing import Optional
from pydantic_settings import BaseSettings
from fastapi import FastAPI, Request
from openai import AsyncOpenAI
from qdrant_client import QdrantClient

logger = logging.getLogger(__name__)

class Settings(BaseSettings):
    DEBUG: bool = False
    LOG_LEVEL: str = "INFO"
    
    # Qdrant
    QDRANT_MODE: str = "cloud"
    QDRANT_URL: str = "http://localhost:6333"
    QDRANT_API_KEY: str | None = None
    QDRANT_COLLECTION: str = "law_docs_v1"       # physical
    QDRANT_COLLECTION_ALIAS: str = "law_docs_v1" # same as physical to avoid alias ops/warnings
    QDRANT_VECTOR_NAME: str = ""                 # "" or e.g. "text"
    
    # Legacy Qdrant settings for backward compatibility
    QDRANT_HOST: str = "localhost"
    QDRANT_PORT: int = 6333
    LOG_QDRANT_HTTP: str = "0"
    
    # LLM
    LLM_PROVIDER: str = "openai"
    LLM_MODEL: str = "gpt-5-mini"
    LLM_MODEL_FALLBACK: str | None = None
    OPENAI_API_KEY: str | None = None
    AZURE_OPENAI_API_KEY: str | None = None
    AZURE_OPENAI_ENDPOINT: str | None = None
    AZURE_OPENAI_DEPLOYMENT: str | None = None
    ANTHROPIC_API_KEY: str | None = None
    GOOGLE_API_KEY: str | None = None
    
    # LLM Behavior Settings (optimized for speed and quality)
    LLM_TEMPERATURE_DEFAULT: float = 0.3
    LLM_TEMPERATURE_LEGAL: float = 0.3
    LLM_MAX_TOKENS: int = 700
    LLM_MAX_OUTPUT_TOKENS: int = 700
    LLM_TOP_P: float = 1.0
    LLM_PRESENCE_PENALTY: float = 0.1
    LLM_FREQUENCY_PENALTY: float = 0.1
    
    # OpenAI Configuration
    OPENAI_CHAT_MODEL: str = "gpt-4o-mini"  # Fast, high-quality, no reasoning tokens
    OPENAI_MAX_TOKENS: int = 2000  # Increased for detailed, comprehensive responses
    OPENAI_TEMPERATURE: float = 0.3  # Lower for consistency, reduce hallucination
    OPENAI_TOP_P: float = 1.0
    OPENAI_TIMEOUT_SECS: int = 30  # Increased for longer, detailed responses
    
    # Qdrant Configuration (optimized for fast, high-quality retrieval)
    QDRANT_SEARCH_TIMEOUT_SECS: int = 2  # Tight timeout for faster responses
    QDRANT_TOP_K_DEFAULT: int = 6  # Focused retrieval: 6 high-quality chunks
    QDRANT_TOP_K_LONG_QUERY: int = 18  # Fetch more for MMR diversity (3x top_k)
    QDRANT_SCORE_THRESHOLD: float = 0.40  # Lowered to include more relevant chunks (was 0.55, too strict)
    
    # Embeddings Configuration
    EMBEDDING_MODEL: str = "text-embedding-3-small"
    
    # RAG Debug Configuration
    DEBUG_RAG: bool = False  # Enable detailed RAG pipeline logging (set to True for debugging)

    # Web Search
    WEB_SEARCH_ENABLED: bool | None = None  # default: enabled if TAVILY_API_KEY exists
    TAVILY_API_KEY: str | None = None
    
    # CORS Configuration
    CORS_ALLOWED_ORIGINS: list[str] = [
        "http://127.0.0.1:8000",
        "http://localhost:5173",
        "http://localhost:3000"
    ]
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = True
        extra = "ignore"

settings = Settings()


def get_qdrant_client() -> QdrantClient:
    """Create Qdrant client based on settings configuration."""
    if settings.QDRANT_MODE == "embedded":
        return QdrantClient(path="./qdrant_data")
    else:
        return QdrantClient(
            url=settings.QDRANT_URL,
            api_key=settings.QDRANT_API_KEY or None,
            timeout=settings.QDRANT_SEARCH_TIMEOUT_SECS
        )


def get_embedder() -> AsyncOpenAI:
    """Create embeddings client based on LLM configuration."""
    return AsyncOpenAI(
        api_key=settings.OPENAI_API_KEY,
        timeout=120.0,
        max_retries=3
    )


def get_llm_client() -> AsyncOpenAI:
    """Create LLM client based on LLM configuration."""
    return AsyncOpenAI(
        api_key=settings.OPENAI_API_KEY,
        timeout=settings.OPENAI_TIMEOUT_SECS
    )


def wire_services(app: FastAPI) -> None:
    """Wire all singleton services into app.state on startup."""
    logger.info("Wiring global services...")
    
    # Wire settings
    app.state.settings = settings
    
    # Wire Qdrant client
    app.state.qdrant = get_qdrant_client()
    
    # Wire embedding client
    app.state.embedder = get_embedder()
    
    # Wire LLM client
    app.state.llm_client = get_llm_client()
    
    # Ensure collection (fail-soft)
    try:
        from app.modules.lawfirmchatbot.services.vector_store import ensure_collection
        ensure_collection(app.state.qdrant, dim=1536)
        logger.info("Qdrant collection initialized successfully")
            
    except Exception as e:
        logger.warning(f"Qdrant collection initialization failed: {str(e)}")
        logger.info("Application will continue without Qdrant initialization")
    
    logger.info("Service container wiring completed successfully")


async def perform_warmup(app: FastAPI) -> None:
    """Perform async warmup operations (call this from startup event)."""
    try:
        from app.modules.lawfirmchatbot.services.embeddings import embed_text
        from app.modules.lawfirmchatbot.services.vector_store import get_runtime_collection_name
        
        v = await embed_text("warmup")
        app.state.qdrant.search(
            collection_name=get_runtime_collection_name(),
            query_vector=v,
            limit=1
        )
        logger.info("Qdrant warmup completed successfully")
    except Exception as warmup_error:
        logger.info(f"Qdrant warmup failed (non-critical): {warmup_error}")


# Legacy compatibility - maintain the old Services class structure
class Services:
    """Legacy Services container for backward compatibility."""
    
    def __init__(self, qdrant_client: QdrantClient, embed_client: AsyncOpenAI, llm_client: AsyncOpenAI):
        self.qdrant_client = qdrant_client
        self.embed_client = embed_client
        self.llm_client = llm_client


def get_legacy_services(request: Request) -> Services:
    """
    Get services in the legacy format for backward compatibility.
    
    Args:
        request: FastAPI request object containing app.state
        
    Returns:
        Services: Legacy services container
    """
    state = request.app.state
    return Services(
        qdrant_client=state.qdrant,
        embed_client=state.embedder,
        llm_client=state.llm_client
    )
