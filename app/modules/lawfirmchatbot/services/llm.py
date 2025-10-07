from typing import List, Dict, Any
import time
import hashlib
import httpx
from openai import AsyncOpenAI
from core.config import settings
from core.utils.perf import profile_stage


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


@profile_stage("llm_response")
async def run_llm_chat(system_prompt: str, user_message: str, history=None):
    """
    Uses the existing ChatCompletion logic.
    Ensures answers are always generated, even if retrieval was weak.
    """
    messages = [{"role": "system", "content": system_prompt}]
    if history:
        messages += history
    messages.append({"role": "user", "content": user_message})

    response = await chat_completion(messages, is_legal_query=False, max_tokens=900, temperature=0.6)
    return response.strip()


def _get_client() -> AsyncOpenAI:
    global _client
    if _client is None:
        _client = AsyncOpenAI(api_key=settings.OPENAI_API_KEY, timeout=settings.OPENAI_TIMEOUT_SECS)
    return _client


def looks_like_citations_only(text: str) -> bool:
    """
    RAG-DEBUG: Improved citations-only detection to reduce false positives.
    Check if response is citations-only without substantive content.
    
    Returns True only if:
    - Response is empty or very short (<50 chars)
    - OR has citations but lacks substantive content (needs ≥2 content lines OR ≥100 chars with ≥1 content line)
    """
    from core.config import settings
    
    t = text.strip()
    if not t:
        return True
    
    # Very short responses are likely empty/insufficient
    if len(t) < 50:
        return True
    
    t_lower = t.lower()
    
    # RAG-DEBUG: Log detection attempt
    if settings.DEBUG_RAG:
        from logging import getLogger
        logger = getLogger(__name__)
        logger.info(f"[RAG-DEBUG] Citations detector analyzing: '{t[:150]}...' (length={len(t)})")
    
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
            (line_lower.startswith('[') and ']' in line_lower[:10])):  # [1] Document - Page X pattern
            citation_lines += 1
        # Count substantive content lines (meaningful text, not just list markers)
        elif len(line) > 15 and not line_lower.startswith(('[', 'page ', 'see ', '- [', '* [')):
            content_lines += 1
    
    # RAG-DEBUG: Log analysis results
    if settings.DEBUG_RAG:
        logger.info(f"[RAG-DEBUG] Citations analysis: content_lines={content_lines}, citation_lines={citation_lines}, total_lines={len(lines)}, total_chars={len(t)}")
    
    # Enhanced detection logic:
    # - Need at least 2 content lines OR 100+ chars with at least 1 content line
    # - This allows short but substantive answers while catching citations-only responses
    has_sufficient_content = content_lines >= 2 or (len(t) >= 100 and content_lines >= 1)
    
    # Flag as citations-only if:
    # - No sufficient content AND (has citation markers OR is short)
    is_citations_only = not has_sufficient_content and (citation_lines > 0 or len(t) < 100)
    
    if settings.DEBUG_RAG:
        logger.info(f"[RAG-DEBUG] Citations detector result: is_citations_only={is_citations_only}, has_sufficient_content={has_sufficient_content}")
    
    return is_citations_only


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
            # gpt-5-mini has restrictions: temperature must be 1.0 (default), so we omit it
            # RAG-DEBUG: Log that we're using gpt-5 with restricted params
            if settings.DEBUG_RAG:
                from logging import getLogger
                logger = getLogger(__name__)
                logger.info(f"[RAG-DEBUG] Using {model} with restricted params (no temperature control)")
            
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
        content = response.choices[0].message.content
        
        # RAG-DEBUG: Log token usage including reasoning tokens
        if settings.DEBUG_RAG:
            from logging import getLogger
            logger = getLogger(__name__)
            usage = response.usage
            logger.info(f"[RAG-DEBUG] Token usage: prompt={usage.prompt_tokens}, completion={usage.completion_tokens}, total={usage.total_tokens}")
            if hasattr(usage, 'completion_tokens_details') and usage.completion_tokens_details:
                details = usage.completion_tokens_details
                if hasattr(details, 'reasoning_tokens') and details.reasoning_tokens:
                    logger.info(f"[RAG-DEBUG] Reasoning tokens used: {details.reasoning_tokens} (internal thinking)")
        
        # RAG-DEBUG: Log if we get None or empty content
        if content is None or content == "":
            from logging import getLogger
            logger = getLogger(__name__)
            logger.error(f"[RAG-DEBUG] LLM returned None/empty content!")
            logger.error(f"[RAG-DEBUG] Finish reason: {response.choices[0].finish_reason}")
            logger.error(f"[RAG-DEBUG] Model: {model}, Max tokens requested: {effective_max_tokens}")
            
            # Log token details to diagnose the issue
            usage = response.usage
            logger.error(f"[RAG-DEBUG] Token usage: prompt={usage.prompt_tokens}, completion={usage.completion_tokens}, total={usage.total_tokens}")
            if hasattr(usage, 'completion_tokens_details') and usage.completion_tokens_details:
                details = usage.completion_tokens_details
                if hasattr(details, 'reasoning_tokens'):
                    logger.error(f"[RAG-DEBUG] Reasoning tokens: {details.reasoning_tokens} - THIS IS WHY OUTPUT IS EMPTY!")
                    logger.error(f"[RAG-DEBUG] Fix: Increase max_tokens to allow for reasoning + actual output")
            
            return ""
        
        return content
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
                fb_content = response.choices[0].message.content
                if fb_content is None or fb_content == "":
                    from logging import getLogger
                    logger = getLogger(__name__)
                    logger.error(f"[RAG-DEBUG] Fallback LLM also returned None/empty! Fallback model: {fallback_model}")
                    return ""
                return fb_content
            except Exception as fallback_error:
                raise e  # Re-raise original error if fallback also fails
        else:
            raise e


