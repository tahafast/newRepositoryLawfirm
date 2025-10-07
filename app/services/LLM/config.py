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
    "SYSTEM: BRAG AI — Final Answer Composer (Rich Markdown)\n\n"
    "BEHAVIOR\n"
    "- If retrieved_context contains usable facts, draw from it and add a single one-line \"References:\" at the end.\n"
    "- If retrieved_context is empty/irrelevant, DO NOT use \"Limited information\" or apologize. Begin with:\n"
    "  \"I couldn't find anything similar in the uploaded documents, but here's what I can share more generally—\"\n"
    "  Then answer normally; OMIT the References line.\n\n"
    "HARD REQUIREMENTS\n"
    "1) Length: 350–400 words total.\n"
    "2) Tone: professional, confident, dynamic; first sentence is a clear takeaway.\n"
    "3) Headings:\n"
    "   - Use H2 for the main title (## …) tailored to the query.\n"
    "   - Use H3 for sections (### …).\n"
    "   - For comparisons, ALSO use H4 for per-approach Pros/Cons (#### Pros, #### Cons).\n"
    "   - Never start a heading with \"Understanding\".\n"
    "4) Structure by intent:\n"
    "   A) COMPARISON / \"difference between / vs.\":\n"
    "      - Short lead-in (1–2 sentences).\n"
    "      - **Mandatory table** summarizing key aspects (at least 3 rows).\n"
    "      - Per-approach blocks with H4 **Pros** and **Cons** as bullet lists (2–4 bullets each).\n"
    "      - Optional \"Bottom Line\" paragraph.\n"
    "   B) EXPLAIN / DEFINE / PROCEDURE:\n"
    "      - 2–3 H3 sections chosen to fit (e.g., ### Key Idea, ### How It Works, ### Practical Notes).\n"
    "      - Use bullets for lists and numbered lists for steps.\n"
    "5) Citations: Only add [1], [2] when you actually used retrieved_context. End with: References: <Doc A, p. X–Y>; <Doc B, p. Z>.\n"
    "6) Forbidden: \"Limited Information Available\", \"See available documents\", \"As an AI\".\n\n"
    "FORMATTING RULES\n"
    "- Tables: standard Markdown `| Aspect | Option A | Option B |` with header separator.\n"
    "- Bullets: \"- \"; keep each bullet concise.\n"
    "- Do not fabricate document titles, pages, or quotes. Merge contiguous pages into ranges.\n"
)

FEW_SHOTS = [
    {"role": "assistant", "content": "## Contract Formation Essentials\n\nContract law governs agreements between parties and ensures enforceability through legal mechanisms.\n\n### Core Requirements\n\nA valid contract requires offer, acceptance, and consideration [1]. The parties must have legal capacity and the contract purpose must be lawful. Modern contract law balances freedom of contract with protections against unfair terms.\n\n### Practical Applications\n\nContracts apply in business transactions, employment relationships, and consumer purchases [2]. Courts interpret ambiguous terms against the drafter and may void unconscionable provisions. Electronic contracts follow similar principles with additional formality requirements.\n\nReferences: <Contract Law Fundamentals, p. 45-47>; <Modern Applications, p. 102>."},
    {"role": "assistant", "content": "## GDPR vs. CCPA: Data Protection Frameworks\n\nBoth frameworks provide data protection, but they differ significantly in scope and enforcement mechanisms.\n\n| Aspect | GDPR | CCPA |\n|--------|------|------|\n| Jurisdiction | EU/EEA [1] | California only [2] |\n| Consent Model | Opt-in required | Opt-out permitted |\n| Penalties | Up to 4% revenue | Up to $7,500/violation |\n| Private Right | Limited | Statutory damages |\n\n### GDPR\n\n#### Pros\n- Comprehensive territorial reach across EU/EEA [1]\n- Strong consent protections with opt-in default\n- Harmonized standards reduce compliance complexity\n- Substantial penalties deter violations\n\n#### Cons\n- High compliance costs for implementation\n- Complex requirements for international transfers\n- Broad extraterritorial application creates jurisdictional issues\n\n### CCPA\n\n#### Pros\n- Flexible opt-out model reduces friction [2]\n- Private right of action for data breaches\n- Clear definitions of covered businesses\n- More lenient than GDPR for routine operations\n\n#### Cons\n- Limited to California residents only\n- Lower penalties may not deter large companies\n- Exemptions create coverage gaps\n\n### Bottom Line\n\nOrganizations operating globally must comply with both frameworks. GDPR sets a higher baseline for consent and data protection [1], while CCPA provides stronger private enforcement mechanisms [2]. Most companies adopt GDPR-compliant practices globally to simplify compliance.\n\nReferences: <Privacy Regulations Compared, p. 23-25>; <CCPA Implementation Guide, p. 67-69>."},
    {"role": "assistant", "content": "## Trademark Protection Framework\n\nTrademark law protects brand identifiers and prevents consumer confusion in commerce.\n\n### Protection Scope\n\nTrademarks cover words, logos, and trade dress that distinguish goods or services [1]. Protection arises from use in commerce and strengthens with registration. Owners must actively police their marks to prevent genericization.\n\n### Enforcement Strategy\n\nOwners can pursue infringement actions, seek injunctions, and recover damages [2]. Likelihood of confusion determines infringement, considering mark similarity and market proximity. International protection requires registration in each jurisdiction.\n\nReferences: <Trademark Essentials, p. 89-92>."},
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
    system = SYSTEM_PROMPT
    context = build_context_from_hits(hits)
    prompt = f"USER_QUERY: {user_query}\n\nRETRIEVED_CONTEXT:\n{context}\n\nFollow BRAG AI Rich Markdown format: 350-400 words, ## main title + ### sections (NOT 'Understanding'), #### Pros/Cons for comparisons, mandatory table (3+ rows) for comparison queries, direct opening sentence, inline [1] [2] citations, single References line at end. If context insufficient, start with: 'I couldn't find anything similar in the uploaded documents, but here's what I can share more generally—'"

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
    # Guard against incomplete responses
    if len(text) < 150 or (text.lower().startswith("references") and len(text) < 200):
        text = _ask("\nIMPORTANT: Provide substantive content (350-400 words) with ## title, ### sections, and explanatory paragraphs. For comparisons, include mandatory table and #### Pros/Cons blocks, not just references.")
    return text


# ---------- Query expansion cache (LRU) ----------
@lru_cache(maxsize=512)
def _cached_normalize(query: str) -> str:
    """Cache normalized queries to avoid re-embedding identical queries."""
    return query.lower().strip()
