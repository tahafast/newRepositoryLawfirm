"""
Compatibility shim for OpenAI configuration imports
Redirects to the new global LLM configuration
"""

# Thin re-export for backward compatibility
from app.services.LLM.config import get_llm_settings, get_llm_config, LLMConfig  # noqa: F401
