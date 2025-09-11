from typing import List, Dict, Any
from openai import AsyncOpenAI
from core.conf import settings


_client: AsyncOpenAI | None = None


def _get_client() -> AsyncOpenAI:
    global _client
    if _client is None:
        _client = AsyncOpenAI(api_key=settings.OPENAI_API_KEY, timeout=60.0)
    return _client


async def chat_completion(messages: List[Dict[str, str]], *, is_legal_query: bool = False) -> str:
    client = _get_client()
    response = await client.chat.completions.create(
        model=settings.LLM_MODEL,
        messages=messages,
        temperature=settings.LLM_TEMPERATURE_LEGAL if is_legal_query else settings.LLM_TEMPERATURE_DEFAULT,
        max_tokens=settings.LLM_MAX_TOKENS,
        presence_penalty=settings.LLM_PRESENCE_PENALTY,
        frequency_penalty=settings.LLM_FREQUENCY_PENALTY,
    )
    return response.choices[0].message.content


