"""Coreference resolver for rewriting user queries into self-contained form."""

import os
import json
from typing import List, Dict
from openai import AsyncOpenAI
from core.config import settings

# Initialize OpenAI client
_client: AsyncOpenAI | None = None

def _get_client() -> AsyncOpenAI:
    global _client
    if _client is None:
        _client = AsyncOpenAI(api_key=settings.OPENAI_API_KEY, timeout=6)
    return _client

SYSTEM = (
    "You rewrite short follow-up chat turns into a self-contained query. "
    "Use the recent dialogue to resolve pronouns/ellipses (it, they, that, this, those, the topic). "
    "Return strict JSON: {\"resolved_query\": \"...\", \"changed\": true|false} . "
    "Do not add explanations."
)

def build_history_preview(history: List[Dict[str, str]], max_chars: int = 1200) -> str:
    """Build a compact history preview for coreference resolution.
    
    Args:
        history: List of message dicts with 'role' and 'content'
        max_chars: Maximum characters to return
        
    Returns:
        Formatted history string
    """
    # Last few pairs; keep small to be fast
    s = []
    for m in history[-8:]:
        role = m.get("role", "user")[:9]
        txt = (m.get("content", "") or "").strip().replace("\n", " ")
        s.append(f"{role.upper()}: {txt}")
    t = "\n".join(s)
    return t[-max_chars:]

async def chat_completion_raw(messages: List[Dict[str, str]], model: str = None, temperature: float = 0.0, 
                              max_tokens: int = 200, timeout: int = 6, 
                              response_format: Dict = None) -> str:
    """Thin wrapper for chat completion that returns raw text.
    
    Args:
        messages: Chat messages
        model: Model to use (defaults to env COREF_MODEL or settings default)
        temperature: Sampling temperature
        max_tokens: Maximum tokens to generate
        timeout: Request timeout in seconds
        response_format: Optional response format (e.g., {"type": "json_object"})
        
    Returns:
        Raw text response
    """
    client = _get_client()
    
    # Use fast model for coreference (gpt-4o-mini or similar)
    if model is None:
        model = os.getenv("COREF_MODEL", "gpt-4o-mini")
    
    params = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    
    if response_format:
        params["response_format"] = response_format
    
    response = await client.chat.completions.create(**params)
    return response.choices[0].message.content or ""

async def resolve_coref(history: List[Dict[str, str]], user_text: str) -> str:
    """Resolve coreferences in user text using recent conversation history.
    
    Returns a best-effort resolved query using recent history.
    If resolver fails, fallback to user_text.
    
    Args:
        history: Recent conversation messages
        user_text: Current user message to resolve
        
    Returns:
        Resolved query string
    """
    history_preview = build_history_preview(history)
    user = (
        "Recent dialogue (most recent last):\n"
        f"{history_preview}\n\n"
        f"User turn to resolve: {user_text}\n"
        "Return JSON ONLY."
    )
    
    try:
        out = await chat_completion_raw(
            messages=[{"role": "system", "content": SYSTEM}, {"role": "user", "content": user}],
            model=os.getenv("COREF_MODEL", "gpt-4o-mini"),
            temperature=0.0,
            max_tokens=200,
            timeout=6,
            response_format={"type": "json_object"}
        )
        data = json.loads(out)
        rq = (data.get("resolved_query") or "").strip()
        return rq if rq else user_text
    except Exception:
        # Fallback to original text on any error
        return user_text

