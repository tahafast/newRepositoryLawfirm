from typing import List, Dict, Any
from openai import AsyncOpenAI
from core.config import settings


_client: AsyncOpenAI | None = None


def _get_client() -> AsyncOpenAI:
    global _client
    if _client is None:
        _client = AsyncOpenAI(api_key=settings.OPENAI_API_KEY, timeout=settings.OPENAI_TIMEOUT_SECS)
    return _client


def looks_like_citations_only(text: str) -> bool:
    """Check if response is citations-only without substantive content."""
    t = text.strip().lower()
    if not t:
        return True
    # Heuristics: starts with "citations" or contains no sentence terminators before citations list
    just_citations = t.startswith("citations ") or ("citations [" in t and len(t.splitlines()) < 4)
    low_content = (t.count(".") + t.count("•") + t.count("- ")) < 2
    return just_citations or low_content


async def chat_completion(messages: List[Dict[str, str]], *, is_legal_query: bool = False, max_tokens: int = None, temperature: float = None) -> str:
    client = _get_client()
    
    # Use optimized settings for faster responses
    try:
        # Use max_completion_tokens for gpt-5 models, max_tokens for older models
        # gpt-5-mini only supports temperature=1.0 (default), so we omit it for gpt-5 models
        model = settings.OPENAI_CHAT_MODEL
        effective_max_tokens = max_tokens if max_tokens is not None else min(settings.OPENAI_MAX_TOKENS, 650)
        effective_temperature = temperature if temperature is not None else settings.OPENAI_TEMPERATURE
        
        if model.startswith("gpt-5"):
            # gpt-5-mini has restrictions: temperature must be 1.0, limited parameters
            params = {
                "model": model,
                "messages": messages,
                "max_completion_tokens": effective_max_tokens,
            }
        elif model.startswith("gpt-4o"):
            params = {
                "model": model,
                "messages": messages,
                "temperature": effective_temperature,
                "max_completion_tokens": effective_max_tokens,
                "top_p": settings.OPENAI_TOP_P,
                "presence_penalty": 0.3,
            }
        else:
            params = {
                "model": model,
                "messages": messages,
                "temperature": effective_temperature,
                "max_tokens": effective_max_tokens,
                "top_p": settings.OPENAI_TOP_P,
                "presence_penalty": 0.3,
            }
        
        response = await client.chat.completions.create(**params)
        return response.choices[0].message.content
    except Exception as e:
        # Try fallback model if available
        if settings.LLM_MODEL_FALLBACK:
            try:
                fallback_model = settings.LLM_MODEL_FALLBACK
                if fallback_model.startswith("gpt-5"):
                    fb_params = {
                        "model": fallback_model,
                        "messages": messages,
                        "max_completion_tokens": effective_max_tokens,
                    }
                elif fallback_model.startswith("gpt-4o"):
                    fb_params = {
                        "model": fallback_model,
                        "messages": messages,
                        "temperature": effective_temperature,
                        "max_completion_tokens": effective_max_tokens,
                        "top_p": settings.OPENAI_TOP_P,
                        "presence_penalty": 0.3,
                    }
                else:
                    fb_params = {
                        "model": fallback_model,
                        "messages": messages,
                        "temperature": effective_temperature,
                        "max_tokens": effective_max_tokens,
                        "top_p": settings.OPENAI_TOP_P,
                        "presence_penalty": 0.3,
                    }
                response = await client.chat.completions.create(**fb_params)
                return response.choices[0].message.content
            except Exception as fallback_error:
                raise e  # Re-raise original error if fallback also fails
        else:
            raise e


