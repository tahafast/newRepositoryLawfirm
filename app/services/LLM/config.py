"""
Global LLM Configuration and Prompting Utilities

Centralized provider configuration, plus reusable system prompt and few-shots
for RAG. These helpers are imported by the orchestrator so the prompt stays
consistent across call sites.
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


# ===================== Prompting =====================

SYSTEM_PROMPT = (
    "You are a legal RAG assistant. Use only the supplied KB chunks and/or web snippets.\n"
    "- Never invent citations or facts. If a requested comparison/topic isn’t found in the supplied context, say so and (only if allowed) use web snippets you were given.\n"
    "- Prefer precise, lawyer-friendly wording; keep it concise but substantive.\n"
    "- Output Markdown with bold H3 section headings; tailor section names to the query intent.\n"
    "- Use bracketed numeric citations like [1], [2] inline, and repeat them in a \"Citations\" section.\n"
    "- If the user asks for a summary, include 3–6 bullet key points.\n"
    "- If the user asks to compare, present a short table then bullets.\n"
    "- If context is insufficient and web is disabled/unavailable, ask a clarifying question instead of guessing.\n"
)

FEW_SHOTS = [
    {"role": "assistant", "content": "### **Definition**\n<one-paragraph definition> [1]\n\n### **Key Points**\n- <point> [1]\n\n### **Citations**\n- [[1] Title — source]"},
    {"role": "assistant", "content": "### **Quick Comparison Table**\n| Item | Feature |\n|---|---|\n| A | ... |\n| B | ... |\n\n### **Key Differences**\n- <difference> [1]\n\n### **Citations**\n- [[1] Title — source]"},
    {"role": "assistant", "content": "### **Summary**\n- <key point> [1]\n- <key point> [2]\n\n### **Citations**\n- [[1] Title — source]"},
]


# ===================== Intent Router & Answer Planner =====================

import os
from functools import lru_cache


# ---------- Intent Router (cheap, low temp) ----------
def route_intent(llm_client, user_query: str) -> dict:
    """
    Returns dict: {
      'intent': 'chit_chat'|'domain_qa'|'multi_domain_qa'|'clarify_needed'|'out_of_scope',
      'needs_web': bool,
      'normalized_query': str
    }
    """
    system = (
        "You are a router for a technical/legal RAG. "
        "Return strict JSON with keys: intent, needs_web, normalized_query. "
        "intent ∈ [chit_chat, domain_qa, multi_domain_qa, clarify_needed, out_of_scope]. "
        "normalized_query: rewrite succinctly with synonyms (≤20 words)."
    )
    user = f"Classify and normalize this query:\n{user_query}\nReturn strict JSON."

    resp = llm_client.chat.completions.create(
        model=os.getenv("ROUTER_MODEL", os.getenv("OPENAI_ROUTER_MODEL", "gpt-5-mini")),
        messages=[{"role":"system","content":system},{"role":"user","content":user}],
        temperature=0.1,
        response_format={"type":"json_object"},
        timeout=10,
    )
    try:
        import json
        data = json.loads(resp.choices[0].message.content)
        data.setdefault("intent", "domain_qa")
        data.setdefault("needs_web", False)
        data.setdefault("normalized_query", user_query)
        return data
    except Exception:
        return {"intent":"domain_qa","needs_web":False,"normalized_query":user_query}


# ---------- Small-talk / clarify / safe guard ----------

def reply_small_talk():
    return ("I'm your RAG assistant. Ask about your ingested books "
            "(Operating Systems, Computer Vision, Generative AI, Entrepreneurship, "
            "Prompt Engineering, HRM), or share a topic + the detail you need.")


def reply_clarify():
    return ("Could you share a bit more detail (e.g., book or subtopic and level of detail)? "
            "That helps me pull the exact pages you need.")


def reply_not_found(router_needs_web: bool):
    if router_needs_web:
        return ("I couldn't find enough in the ingested materials. I can use the web "
                "if you enable it, or please narrow the topic/book so I can search precisely.")
    return ("I couldn't find strong matches in the ingested documents. "
            "Please specify the book/topic or add a bit more detail.")


# ---------- Answer Planner (dynamic headings + no citations-only) ----------

def build_context_from_hits(hits, max_chars=12000):
    parts = []
    for h in (hits or []):
        doc = (h.payload or {}).get("document") or (h.payload or {}).get("doc") or ""
        page = (h.payload or {}).get("page") or (h.payload or {}).get("pages")
        txt = (h.payload or {}).get("text") or ""
        head = f"[{doc} p.{page}]" if page else f"[{doc}]"
        parts.append(f"{head}\n{txt}")
    ctx = "\n\n---\n".join(parts)
    return ctx[:max_chars]


def answer_with_context(llm_client, user_query: str, hits: list) -> str:
    system = (
        "You are a faithful RAG answerer. Use ONLY the provided context. "
        "If context is insufficient: say so briefly and suggest 1 clarifying question. "
        "Structure headings dynamically to fit the question (e.g., Definition, Steps, Pros/Cons, Example, Caveats). "
        "Start with a short direct answer, then details. "
        "Use inline page citations like [p.410] when you cite lines from context. "
        "End with a compact 'References' list showing doc names and pages. "
        "Never output only citations; always include an answer. "
        "Output Markdown."
    )
    context = build_context_from_hits(hits)
    prompt = f"User question:\n{user_query}\n\nContext (verbatim):\n{context}\n"

    def _ask(sys_add=""):
        resp = llm_client.chat.completions.create(
            model=os.getenv("ANSWER_MODEL", os.getenv("OPENAI_ANSWER_MODEL","gpt-5-mini")),
            messages=[{"role":"system","content":system+sys_add},
                      {"role":"user","content":prompt}],
            temperature=float(os.getenv("ANSWER_TEMP","0.2")),
            max_tokens=int(os.getenv("ANSWER_MAX_TOKENS","900")),
            timeout=30,
        )
        return resp.choices[0].message.content.strip()

    text = _ask()
    # Guard against "citations-only"
    if len(text) < 120 or text.lower().startswith("citations"):
        text = _ask("\nIMPORTANT: Provide an actual answer section (not only citations).")
    return text


# ---------- Query expansion cache (LRU) ----------
@lru_cache(maxsize=512)
def _cached_normalize(query: str) -> str:
    """Cache normalized queries to avoid re-embedding identical queries."""
    return query.lower().strip()
