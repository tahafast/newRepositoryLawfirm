from typing import List, Dict, Any
from openai import AsyncOpenAI
from core.config import settings


_client: AsyncOpenAI | None = None


def _get_client() -> AsyncOpenAI:
    global _client
    if _client is None:
        _client = AsyncOpenAI(api_key=settings.OPENAI_API_KEY, timeout=60.0)
    return _client


async def chat_completion(messages: List[Dict[str, str]], *, is_legal_query: bool = False) -> str:
    client = _get_client()
    
    # Try primary model first
    try:
        temp = settings.LLM_TEMPERATURE_LEGAL if is_legal_query else settings.LLM_TEMPERATURE_DEFAULT
        # Choose proper token param based on model family
        token_arg = {("max_completion_tokens" if settings.LLM_MODEL.startswith("gpt-5") else "max_tokens"): settings.LLM_MAX_OUTPUT_TOKENS}
        params = {
            "model": settings.LLM_MODEL,
            "messages": messages,
            "temperature": temp,
            **token_arg,
        }
        if not settings.LLM_MODEL.startswith("gpt-5"):
            params.update({
                "top_p": settings.LLM_TOP_P,
                "presence_penalty": settings.LLM_PRESENCE_PENALTY,
                "frequency_penalty": settings.LLM_FREQUENCY_PENALTY,
            })
        response = await client.chat.completions.create(**params)
        return response.choices[0].message.content
    except Exception as e:
        # Try fallback model if available
        if settings.LLM_MODEL_FALLBACK:
            try:
                fallback_temp = settings.LLM_TEMPERATURE_LEGAL if is_legal_query else settings.LLM_TEMPERATURE_DEFAULT
                fb_token_arg = {("max_completion_tokens" if settings.LLM_MODEL_FALLBACK.startswith("gpt-5") else "max_tokens"): settings.LLM_MAX_OUTPUT_TOKENS}
                fb_params = {
                    "model": settings.LLM_MODEL_FALLBACK,
                    "messages": messages,
                    "temperature": fallback_temp,
                    **fb_token_arg,
                }
                if not settings.LLM_MODEL_FALLBACK.startswith("gpt-5"):
                    fb_params.update({
                        "top_p": settings.LLM_TOP_P,
                        "presence_penalty": settings.LLM_PRESENCE_PENALTY,
                        "frequency_penalty": settings.LLM_FREQUENCY_PENALTY,
                    })
                response = await client.chat.completions.create(**fb_params)
                return response.choices[0].message.content
            except Exception as fallback_error:
                raise e  # Re-raise original error if fallback also fails
        else:
            raise e


