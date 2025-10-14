from typing import List, Dict, Any, Optional, Tuple
import time
import hashlib
import httpx
import logging
import re
from openai import AsyncOpenAI, BadRequestError
from core.config import settings
from core.utils.perf import profile_stage

logger = logging.getLogger(__name__)

_HTTPX = httpx.AsyncClient(timeout=6, limits=httpx.Limits(max_keepalive_connections=20, max_connections=40))
_EMBED_CACHE: dict[str, tuple[float, list[float]]] = {}
_EMBED_TTL = settings.EMBED_CACHE_TTL_S
_EMBED_MAX = settings.EMBED_CACHE_MAX

_client: AsyncOpenAI | None = None


def _ekey(model: str, text: str) -> str:
    h = hashlib.blake2b(text.encode("utf-8"), digest_size=16).hexdigest()
    return f"{model}:{h}"


def _embed_cache_get(model: str, text: str):
    key = _ekey(model, text)
    item = _EMBED_CACHE.get(key)
    if not item: return None
    ts, vec = item
    if time.time() - ts > _EMBED_TTL:
        _EMBED_CACHE.pop(key, None)
        return None
    return vec


def _embed_cache_put(model: str, text: str, vec: list[float]):
    if len(_EMBED_CACHE) >= _EMBED_MAX:
        _EMBED_CACHE.pop(next(iter(_EMBED_CACHE)))
    _EMBED_CACHE[_ekey(model, text)] = (time.time(), vec)


@profile_stage("embedding")
async def embed_text_async(text: str) -> list[float]:
    """Cached async embedding using persistent HTTP client."""
    model = settings.EMBEDDING_MODEL
    hit = _embed_cache_get(model, text)
    if hit is not None:
        return hit

    # Use shared HTTP client for OpenAI API calls
    import openai
    client = AsyncOpenAI(api_key=settings.OPENAI_API_KEY, http_client=_HTTPX)
    res = await client.embeddings.create(
        model=model, input=text
    )
    vec = res.data[0].embedding
    _embed_cache_put(model, text, vec)
    return vec


def embed_text(text: str) -> list[float]:
    import anyio
    return anyio.run(embed_text_async, text)


def _get_client() -> AsyncOpenAI:
    global _client
    if _client is None:
        _client = AsyncOpenAI(api_key=settings.OPENAI_API_KEY, timeout=settings.OPENAI_TIMEOUT_SECS)
    return _client


# ============================================================
# Helper functions for dual-path LLM calling
# ============================================================

def _normalize_model(name: str) -> str:
    """Normalize model name by removing spaces and lowercasing."""
    return re.sub(r"\s+", "", (name or "")).lower()


def _is_gpt5(model: str) -> bool:
    """Check if model is GPT-5 (requires Responses API)."""
    m = _normalize_model(model)
    return m.startswith("gpt-5")


def _budgets_for(model: str, intent: str) -> Tuple[int, int]:
    """
    Returns (reasoning_budget, output_budget).
    For GPT-5: uses Responses API with reasoning.
    For 4o-mini: uses Chat Completions without reasoning.
    """
    if _is_gpt5(model):
        # Doc-gen needs long output
        if intent == "docgen":
            return (1200, 5500)
        return (600, 2000)
    if intent == "docgen":
        # Allow longer completions for structured documents when using chat-completions models
        return (0, 2400)
    # 4o-mini: no reasoning; we keep outputs modest
    return (0, 850)


def _strip_keys(d: Dict[str, Any], keys: tuple) -> Dict[str, Any]:
    """Strip specified keys from dictionary."""
    return {k: v for k, v in (d or {}).items() if k not in keys}


def looks_like_citations_only(text: str) -> bool:
    """
    Check if response is citations-only without substantive content.
    
    Returns True only if:
    - Response is empty or very short (<50 chars)
    - OR has citations but lacks substantive content
    """
    t = text.strip()
    if not t or len(t) < 50:
        return True
    
    # Count actual content vs citations/references
    lines = [line.strip() for line in t.split('\n') if line.strip()]
    content_lines = 0
    citation_lines = 0
    
    for line in lines:
        line_lower = line.lower()
        # Skip empty lines and pure formatting
        if not line or line in ['---', '***', '']:
            continue
        # Count citation/reference lines
        if (line_lower.startswith(('citations:', 'references:', '**references**', '## references', '[1]', '[2]', '[3]', '[4]', '[5]')) or
            line_lower.startswith('reference pages:') or
            line_lower.startswith('**reference pages**') or
            (line_lower.startswith('[') and ']' in line_lower[:10])):
            citation_lines += 1
        # Count substantive content lines
        elif len(line) > 15 and not line_lower.startswith(('[', 'page ', 'see ', '- [', '* [')):
            content_lines += 1
    
    has_sufficient_content = content_lines >= 2 or (len(t) >= 100 and content_lines >= 1)
    is_citations_only = not has_sufficient_content and (citation_lines > 0 or len(t) < 100)
    
    return is_citations_only


# ============================================================
# Unified chat_completion: Auto-routes GPT-5 → Responses, 4o-mini → Chat
# ============================================================

async def chat_completion(
    client,
    model: str,
    messages: list[dict],
    intent: str = "general",       # "general" | "docgen"
    temperature: Optional[float] = 0.2,
    stream: bool = False,
    extra_params: Optional[Dict[str, Any]] = None,
):
    """
    Unified LLM entry:
      • 4o-mini  → Chat Completions (no 'reasoning')
      • GPT-5    → Responses API (with 'reasoning' + 'max_output_tokens')
    Also handles 'temperature' unsupported errors by retrying without it.
    """
    extra_params = dict(extra_params or {})
    model_norm = _normalize_model(model)
    reasoning_budget, output_budget = _budgets_for(model, intent)

    # --------------------------
    # Path A: GPT-4o/GPT-5 → Use max_completion_tokens
    # --------------------------
    if _is_gpt5(model_norm) or model_norm.startswith("gpt-4o"):
        params = {
            "model": model,
            "messages": messages,
            "max_completion_tokens": output_budget,
            "temperature": 0.2 if intent == "docgen" else 0.3,  # Lower temp for doc-gen consistency
        }
        
        # Strip incompatible params
        params.update(_strip_keys(extra_params, ("max_tokens", "reasoning", "modalities")))

        if settings.DEBUG_RAG:
            logger.info(f"[LLM] Chat Completions ({model}): intent={intent}, max_completion_tokens={output_budget}")

        return await client.chat.completions.create(**params)

    # ----------------------------------------
    # Path B: 4o-mini → Chat Completions (fast)
    # ----------------------------------------
    params = {
        "model": model,
        "messages": messages,
        "max_tokens": output_budget,
        "stream": stream,
    }
    # 4o-mini accepts temperature; keep it if provided
    if temperature is not None:
        params["temperature"] = temperature

    # Never pass Responses-only params to Chat Completions
    params.update(_strip_keys(extra_params, ("reasoning", "max_output_tokens")))

    if settings.DEBUG_RAG:
        logger.info(f"[LLM] Chat Completions: model={model}, intent={intent}, max_tokens={output_budget}")

    try:
        return await client.chat.completions.create(**params)
    except BadRequestError as e:
        # Some endpoints may forbid non-default temperature; retry without it
        msg = str(e)
        if "temperature" in msg and "unsupported" in msg.lower():
            logger.warning(f"[LLM] 4o-mini rejected temperature, retrying without it")
            params.pop("temperature", None)
            return await client.chat.completions.create(**params)
        raise


# ============================================================
# Legacy wrapper for backward compatibility
# ============================================================

@profile_stage("llm_response")
async def run_llm_chat(system_prompt: str, user_message: str, history=None):
    """
    Legacy wrapper for existing code.
    Uses the existing ChatCompletion logic.
    """
    messages = [{"role": "system", "content": system_prompt}]
    if history:
        messages += history
    messages.append({"role": "user", "content": user_message})

    client = _get_client()
    import os
    model_general = os.getenv("GENERAL_MODEL", "gpt-4o-mini")
    
    response = await chat_completion(
        client=client,
        model=model_general,
        messages=messages,
        intent="general",
        temperature=0.6,
        stream=False,
        extra_params={"max_tokens": 900}
    )
    
    # Extract content from response
    if hasattr(response, 'choices'):
        # Chat Completions response
        content = response.choices[0].message.content or ""
    elif hasattr(response, 'output'):
        # Responses API response
        content = response.output[0].content if isinstance(response.output, list) else response.output
    else:
        content = ""
    
    return content.strip()


async def responses_generate(
    model: str,
    system_prompt: str,
    user_prompt: str,
    max_output_tokens: int = 2048,
    extra_messages: Optional[List[Dict[str, str]]] = None,
) -> str:
    """
    Lightweight helper for the Responses API (GPT-5 class models).
    Returns concatenated output text.
    """
    client = _get_client()

    prompt_messages: List[Dict[str, str]] = [
        {"role": "system", "content": (system_prompt or "").strip()}
    ]

    for msg in extra_messages or []:
        role = msg.get("role")
        content = (msg.get("content") or "").strip()
        if not role or not content:
            continue
        prompt_messages.append({"role": role, "content": content})

    prompt_messages.append({"role": "user", "content": (user_prompt or "").strip()})

    response = await client.responses.create(
        model=model,
        input=prompt_messages,
        max_output_tokens=max_output_tokens,
    )

    if hasattr(response, "output_text") and response.output_text:
        return response.output_text.strip()

    output_chunks: List[str] = []
    for item in getattr(response, "output", []) or []:
        if hasattr(item, "content"):
            output_chunks.append(str(item.content))
        elif hasattr(item, "text"):
            output_chunks.append(str(item.text))

    if output_chunks:
        return "\n".join(output_chunks).strip()

    return ""
