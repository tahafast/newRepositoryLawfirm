"""
Global LLM Services Module
Provides global access to LLM services and configuration
"""

from .config import (
    llm_service,
    get_llm_service,
    get_llm_client,
    chat_completion,
    generate_embeddings,
    is_llm_available,
    get_llm_config,
    GlobalLLMService
)

# Export the global LLM service instance
__all__ = [
    "llm_service",
    "get_llm_service", 
    "get_llm_client",
    "chat_completion",
    "generate_embeddings",
    "is_llm_available",
    "get_llm_config",
    "GlobalLLMService"
]
