from typing import List, Dict, Any
from openai import AsyncOpenAI
from core.config import settings


_client: AsyncOpenAI | None = None


def _get_client() -> AsyncOpenAI:
    global _client
    if _client is None:
        _client = AsyncOpenAI(api_key=settings.OPENAI_API_KEY, timeout=settings.OPENAI_TIMEOUT_SECS)
    return _client


async def chat_completion(messages: List[Dict[str, str]], *, is_legal_query: bool = False) -> str:
    client = _get_client()
    
    # Use optimized settings for faster responses
    try:
        params = {
            "model": settings.OPENAI_CHAT_MODEL,
            "messages": messages,
            "temperature": settings.OPENAI_TEMPERATURE,
            "max_tokens": settings.OPENAI_MAX_TOKENS,
            "top_p": settings.OPENAI_TOP_P,
            "timeout": settings.OPENAI_TIMEOUT_SECS
        }
        
        response = await client.chat.completions.create(**params)
        return response.choices[0].message.content
    except Exception as e:
        # Try fallback model if available
        if settings.LLM_MODEL_FALLBACK:
            try:
                fb_params = {
                    "model": settings.LLM_MODEL_FALLBACK,
                    "messages": messages,
                    "temperature": settings.OPENAI_TEMPERATURE,
                    "max_tokens": settings.OPENAI_MAX_TOKENS,
                    "top_p": settings.OPENAI_TOP_P,
                    "timeout": settings.OPENAI_TIMEOUT_SECS
                }
                response = await client.chat.completions.create(**fb_params)
                return response.choices[0].message.content
            except Exception as fallback_error:
                raise e  # Re-raise original error if fallback also fails
        else:
            raise e


