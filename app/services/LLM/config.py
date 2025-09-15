"""
Global LLM Configuration
Centralized LLM provider configuration and key management
"""

from typing import Optional
from core.config import settings


class LLMConfig:
    """Global LLM configuration class for provider and key management."""
    
    def __init__(self):
        self.provider: str = settings.LLM_PROVIDER
        self.model: str = settings.LLM_MODEL
        
        # OpenAI Configuration
        self.openai_api_key: Optional[str] = settings.OPENAI_API_KEY
        
        # Azure OpenAI Configuration
        self.azure_api_key: Optional[str] = settings.AZURE_OPENAI_API_KEY
        self.azure_endpoint: Optional[str] = settings.AZURE_OPENAI_ENDPOINT
        self.azure_deployment: Optional[str] = settings.AZURE_OPENAI_DEPLOYMENT
        
        # Other LLM Providers
        self.anthropic_api_key: Optional[str] = settings.ANTHROPIC_API_KEY
        self.google_api_key: Optional[str] = settings.GOOGLE_API_KEY
        
        # LLM Behavior Settings
        self.temperature_default: float = settings.LLM_TEMPERATURE_DEFAULT
        self.temperature_legal: float = settings.LLM_TEMPERATURE_LEGAL
        self.max_tokens: int = settings.LLM_MAX_TOKENS
        self.presence_penalty: float = settings.LLM_PRESENCE_PENALTY
        self.frequency_penalty: float = settings.LLM_FREQUENCY_PENALTY
        
        # Embeddings
        self.embedding_model: str = settings.EMBEDDING_MODEL
    
    def get_api_key(self) -> Optional[str]:
        """Get the appropriate API key based on the current provider."""
        if self.provider == "openai":
            return self.openai_api_key
        elif self.provider == "azure_openai":
            return self.azure_api_key
        elif self.provider == "anthropic":
            return self.anthropic_api_key
        elif self.provider == "vertex":
            return self.google_api_key
        return None
    
    def is_configured(self) -> bool:
        """Check if the current provider is properly configured."""
        api_key = self.get_api_key()
        if self.provider == "azure_openai":
            return bool(api_key and self.azure_endpoint and self.azure_deployment)
        return bool(api_key)


def get_llm_settings() -> LLMConfig:
    """Get the global LLM configuration instance."""
    return LLMConfig()


# Legacy compatibility - keep existing imports working
def get_llm_config() -> LLMConfig:
    """Legacy alias for get_llm_settings()."""
    return get_llm_settings()
