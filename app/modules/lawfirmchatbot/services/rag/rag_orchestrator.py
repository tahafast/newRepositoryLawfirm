
"""Main RAG orchestration service with modular architecture."""

from typing import Dict, Any, Optional, List, Tuple, Literal, Sequence
from datetime import datetime
import logging
import os
import re

from app.modules.lawfirmchatbot.services.ingestion.document_processor import process_document
from app.modules.lawfirmchatbot.services.retrieval.vector_search import (
    add_documents_to_vector_store,
    search_similar_documents,
    normalize_hits,
    build_user_prompt,
    VectorSearch,
    semantic_search_ephemeral,
    ephemeral_scroll_for_conversation,
)
from app.modules.lawfirmchatbot.services.llm import chat_completion, looks_like_citations_only, run_llm_chat, embed_text_async
from app.modules.lawfirmchatbot.schema.query import QueryResponse, DebugInfo, DebugQueryAnalysis
from app.modules.lawfirmchatbot.services.query_analyzer import QueryComplexityAnalyzer, is_docgen_request
from app.modules.lawfirmchatbot.services.reranker import DocumentReranker
from app.modules.lawfirmchatbot.services.adaptive_retrieval_service import AdaptiveRetrievalService
from app.core.answer_policy import classify_intent, select_strategy, format_markdown, grounding_guardrails
from core.config import settings
from core.utils.perf import profile_stage
from app.services.memory.db import SessionLocal
from app.services.memory.memory_manager import ChatMemoryManager
from app.services.memory.coref_resolver import resolve_coref
from app.modules.lawfirmchatbot.services.prompts.docgen import get_docgen_prompt, build_docgen_prompt
from app.modules.lawfirmchatbot.services.prompts import build_system_prompt
from app.modules.lawfirmchatbot.services.prompts.system_prompt import ATTACHMENT_HINT
from app.modules.lawfirmchatbot.services.ephemeral_priority import fetch_ephemeral_priority_context
from app.modules.lawfirmchatbot.services.intent_router import decide_intent
from qdrant_client import QdrantClient
from app.modules.lawfirmchatbot.services.conversation_state import (
    get_conversation_state,
    update_summary,
    update_last_document,
    record_response,
)
from app.modules.lawfirmchatbot.services.qdrant_collections import get_ephemeral_collection
from app.modules.lawfirmchatbot.services.document_service import render_html_document, extract_case_info
from app.modules.lawfirmchatbot.services.ephemeral_store import (
    collection_exists as ephemeral_collection_exists,
)
from app.modules.lawfirmchatbot.services._lc_compat import ensure_OpenAIEmbeddings, ensure_Document
from app.services.LLM.config import get_llm_settings

logger = logging.getLogger(__name__)

# Initialize LLM config for routing
LLM = get_llm_settings()

OpenAIEmbeddings = ensure_OpenAIEmbeddings()
Document = ensure_Document()
EPHEMERAL_COLLECTION = get_ephemeral_collection()


# === Attachment-Aware Routing ===
from typing import Literal, Sequence

ATTACHMENT_MENTIONS = re.compile(
    r"\b(attached|attachment|this\s+doc(?:ument)?|the\s+file\s+above|provided\s+(?:doc|document|file)|my\s+document|above\s+document|above\s+file|summarize\s+the)\b",
    re.IGNORECASE,
)

Scope = Literal["ephemeral_only", "hybrid"]


def _mentions_attached_text(query: str) -> bool:
    """Check if query mentions attached documents."""
    return bool(ATTACHMENT_MENTIONS.search(query))


def _latest_ephemeral_file_ids(qdrant_client, conversation_id: str, limit: int = 5) -> List[str]:
    """
    Returns the N most recent file_ids for the conversation by scanning ephemerals.
    Safe fallback when the UI did not send file_ids in the query payload.
    
    Args:
        qdrant_client: Qdrant client instance
        conversation_id: Conversation identifier
        limit: Max number of file_ids to return
        
    Returns:
        List of file_ids (most recent first, up to limit)
    """
    from qdrant_client.models import Filter, FieldCondition, MatchValue
    
    try:
        # Scroll ephemeral collection to find file_ids
        # Use larger limit to ensure we catch all docs even if collection is growing
        res = qdrant_client.scroll(
            collection_name=EPHEMERAL_COLLECTION,
            scroll_filter=Filter(
                must=[FieldCondition(key="conversation_id", match=MatchValue(value=conversation_id))]
            ),
            limit=512,  # Larger limit for reliability
            with_payload=True,
            with_vectors=False,
        )
        
        file_ids = []
        seen = set()
        
        # Extract unique file_ids (try both file_id and doc_id for compatibility)
        points = res[0] if isinstance(res, tuple) else res
        for point in points:
            payload = getattr(point, 'payload', {}) or {}
            
            # Try file_id first (new), then doc_id (fallback)
            fid = payload.get("file_id") or payload.get("doc_id")
            
            if fid and fid not in seen:
                seen.add(fid)
                file_ids.append(fid)
                if len(file_ids) >= limit:
                    break
        
        logger.info(f"[ephemeral-autodiscovery] Found {len(file_ids)} file_ids for conversation {conversation_id}: {file_ids[:3]}...")
        return file_ids
    
    except Exception as e:
        logger.error(f"[ephemeral-autodiscovery] Failed to discover file_ids: {e}", exc_info=True)
        return []


def _mentions_attached(query: str) -> bool:
    if not query:
        return False
    lowered = query.lower()
    direct_terms = [
        "attached document",
        "attached file",
        "this document",
        "this file",
        "above document",
        "above file",
    ]
    if any(term in lowered for term in direct_terms):
        return True
    return _mentions_attached_text(query)


def _has_ephemeral_for_conversation(qdrant_client, conversation_id: str) -> bool:
    if not conversation_id:
        return False
    try:
        ids = _latest_ephemeral_file_ids(qdrant_client, conversation_id, limit=1)
        return bool(ids)
    except Exception:
        return False


def _attachment_scope(user_text: str, eph_files: Sequence[dict], conversation_id: str = None, qdrant_client=None) -> Tuple[Scope, List[str]]:
    """
    Decide whether to restrict retrieval to ephemeral docs only.
    If user cites a specific attached filename, return those ids to filter on.
    
    Args:
        user_text: User query text
        eph_files: List of ephemeral file metadata from UI
        conversation_id: Conversation ID for auto-discovery
        qdrant_client: Qdrant client for auto-discovery
    
    Returns:
        Tuple of (scope, file_ids_to_filter)
        - scope: "ephemeral_only" or "hybrid"
        - file_ids_to_filter: list of file_ids if specific file mentioned, else auto-discovered
    """
    text = user_text.lower()
    
    # Check if ephemerals exist (either from UI or need to discover)
    has_ephemeral = bool(eph_files)
    if not has_ephemeral and conversation_id and qdrant_client:
        # Try to discover file_ids from Qdrant
        discovered_ids = _latest_ephemeral_file_ids(qdrant_client, conversation_id, limit=5)
        has_ephemeral = bool(discovered_ids)
        if has_ephemeral:
            logger.info(f"[attachment-scope] Auto-discovered {len(discovered_ids)} ephemeral files")
    
    if not has_ephemeral:
        return "hybrid", []

    # If they mention "attached/this document", FORCE ephemeral-only
    if _mentions_attached_text(text):
        logger.info(f"[attachment-scope] Detected attachment mention, FORCING ephemeral-only")
        
        # Auto-discover file_ids if not provided
        if not eph_files and conversation_id and qdrant_client:
            file_ids = _latest_ephemeral_file_ids(qdrant_client, conversation_id, limit=5)
            return "ephemeral_only", file_ids
        
        # Extract file_ids from eph_files
        file_ids = [f.get("file_id", "") for f in eph_files if f.get("file_id")]
        return "ephemeral_only", file_ids

    # If they mention a specific attached filename, scope to that/those
    named_ids = []
    for f in eph_files:
        name = f.get("file_name", "")
        if name and name.lower() in text:
            named_ids.append(f.get("file_id", ""))
            logger.info(f"[attachment-scope] Detected filename '{name}' in query, filtering to that file")
    
    if named_ids:
        return "ephemeral_only", [fid for fid in named_ids if fid]  # filter out empty ids

    # default: hybrid (ephemeral prioritized, then main)
    logger.info(f"[attachment-scope] No explicit attachment mention, using hybrid scope")
    return "hybrid", []


# === DOCGEN Intent Detection ===
DOCGEN_KEYWORDS = [
    "generate", "draft", "prepare", "write", "compose",
    "affidavit", "synopsis", "petition", "suit", "rejoinder",
    "legal notice", "counter affidavit", "plaint", "reply",
    "document", "bail", "application"
]

TWEAK_KEYWORDS = [
    "tweak", "revise", "modify", "simplify", "edit", "tone",
    "change", "update", "add", "remove", "replace"
]

FOLLOWUP_KEYWORDS = [
    "continue", "carry on", "from above", "use above", "same info", "same details",
    "as before", "as discussed", "earlier", "previous response", "next part",
    "keep going", "follow up", "add to that"
]

# --- Add skeleton/template keywords (used to allow placeholder docgen) ---
SKELETON_KEYWORDS = [
    "skeleton", "format", "template", "placeholders", "placeholder", "sketch", "skeleton of", "format of"
]


def detect_placeholders(user_query: str, state_cache=None) -> bool:
    """
    Detect if user intends a skeleton/template or wants placeholders used.
    Returns True when query explicitly asks for skeleton/template/format/placeholders,
    or when user issues a minimal 'draft ... between X and Y' that should accept placeholders.
    """
    if not user_query:
        return False
    text = (user_query or "").lower()

    # explicit keywords
    for kw in SKELETON_KEYWORDS:
        if kw in text:
            return True

    # "use above" or "use details provided above" -> assume placeholders ok
    if "use above" in text or "use details provided above" in text or "use details provided" in text:
        return True

    # "draft ... between A and B" or "suit between A and B" minimal-names heuristic
    # look for 'between <Name> and <Name>' where Name is capitalized-ish
    m = re.search(r"\bbetween\s+([A-Z][a-zA-Z.\- ]{1,40}?)\s+and\s+([A-Z][a-zA-Z.\- ]{1,40}?)", user_query)
    if m:
        return True

    # short form: "draft a suit between muhammad bilal and taha rahat" (lowercase names) -> also allow
    m2 = re.search(r"\bdraft\b.*\bbetween\b.*\band\b", text)
    if m2:
        return True

    return False

def _looks_like_docgen(q: str) -> bool:
    """Detect if query looks like document generation request."""
    t = (q or "").lower()
    return any(k in t for k in DOCGEN_KEYWORDS)


def detect_doc_type(query: str) -> str:
    """Lightweight intent classifier to determine requested document type."""
    q = (query or "").lower()
    if any(k in q for k in ["affidavit", "counter affidavit", "counter-affidavit"]):
        return "affidavit"
    if any(k in q for k in ["synopsis", "summary", "synopses"]):
        return "synopsis"
    if any(k in q for k in ["rejoinder", "reply", "response"]):
        return "rejoinder"
    if any(k in q for k in ["legal notice", "notice", "serve notice"]):
        return "legal_notice"
    return "general"


def _is_tweak_request(query: str) -> bool:
    text = (query or "").lower()
    return any(kw in text for kw in TWEAK_KEYWORDS)


def _is_followup_request(query: str) -> bool:
    text = (query or "").lower()
    return any(kw in text for kw in FOLLOWUP_KEYWORDS)


def detect_mode(user_query: str, cache=None) -> str:
    """
    Classify query into general QA, fresh document generation, or document tweak.
    """
    text = (user_query or "").lower()
    last_doc = ""
    if cache:
        if isinstance(cache, dict):
            last_doc = cache.get("last_doc") or cache.get("last_document") or ""
        else:
            last_doc = getattr(cache, "last_document", "") or getattr(cache, "last_doc", "")
    has_last_doc = bool(last_doc)

    if has_last_doc and (any(t in text for t in TWEAK_KEYWORDS) or any(f in text for f in FOLLOWUP_KEYWORDS)):
        return "tweak_doc"

    if any(k in text for k in DOCGEN_KEYWORDS):
        return "docgen"

    return "general"


def should_use_retriever(mode: str, user_query: str) -> bool:
    """
    Decide whether to hit Qdrant based on routing mode and trigger keywords.
    """
    text = (user_query or "").lower()

    if mode == "tweak_doc":
        return False

    if mode == "docgen":
        return True

    if mode == "general":
        keywords = ["law", "act", "article", "section", "ordinance", "precedent"]
        return any(k in text for k in keywords)

    return False


async def _collect_ephemeral_fallback_text(
    qdrant_client,
    conversation_id: str,
    file_ids: List[str],
    max_chars: int = 4000,
) -> str:
    """
    Collect raw text from ephemeral collection as fallback when similarity scores are too low.
    This ensures we ALWAYS have context from attached documents when they exist.
    
    Args:
        qdrant_client: Qdrant client instance
        conversation_id: Conversation identifier
        file_ids: List of file_ids to filter on
        max_chars: Maximum characters to collect
        
    Returns:
        Concatenated text from ephemeral chunks
    """
    try:
        points = await ephemeral_scroll_for_conversation(
            qdrant_client,
            conversation_id,
            file_ids=file_ids,
            limit=512,
        )
        
        text_parts = []
        total_len = 0
        
        for point in points:
            payload = getattr(point, 'payload', {}) or {}
            text = payload.get("text") or payload.get("chunk") or payload.get("page_content") or ""
            
            if text and len(text.strip()) > 20:  # Only meaningful chunks
                text_parts.append(text.strip())
                total_len += len(text)
                
                if total_len >= max_chars:
                    break
        
        result = "\n\n".join(text_parts)[:max_chars]
        logger.info(f"[ephemeral-fallback] Collected {len(result)} chars from {len(text_parts)} chunks")
        return result
    
    except Exception as e:
        logger.warning(f"[ephemeral-fallback] Failed to collect text: {e}")
        return ""


async def _update_summary_with_exchange(
    conversation_id: str,
    state,
    user_message: str,
    assistant_reply: str,
) -> None:
    """
    Append the latest exchange to the running summary, summarizing when necessary.
    """
    user_text = (user_message or "").strip()
    assistant_text = (assistant_reply or "").strip()

    truncated_assistant = assistant_text
    if len(truncated_assistant) > 1200:
        truncated_assistant = truncated_assistant[:1200]

    existing = (state.summary or "").strip()
    addition = f"User: {user_text}\nAssistant: {truncated_assistant}"
    combined = f"{existing}\n{addition}".strip() if existing else addition

    # Summarize if the running summary becomes too long.
    if len(combined.split()) > 360:
        summary_prompt = (
            "Summarize the following conversation between a legal assistant and a user. "
            "Capture key facts, commitments, decisions, preferred tone, and outstanding requests "
            "in bullet form (max 6 bullets):\n\n"
            f"{combined}"
        )
        summary = await run_llm_chat(
            system_prompt="You condense legal assistant conversations into concise bullet summaries.",
            user_message=summary_prompt,
        )
        await update_summary(conversation_id, summary.strip())
    else:
        await update_summary(conversation_id, combined)


def ensure_markdown(answer: str, refs: list[str]) -> str:
    """
    Ensure answer has proper markdown formatting.
    Adds heading if missing and normalizes references.
    """
    text = answer.strip()
    
    # Add main heading if missing
    if text and "# " not in text[:80].lower():
        text = "# Answer\n\n" + text
    
    # Add references if provided and not already present
    if refs and "**references**" not in text.lower() and "references:" not in text.lower():
        text += "\n\n**References:** " + "; ".join(refs)
    
    return text


@profile_stage("rag_orchestrator")
async def generate_answer(query: str, history=None):
    """
    Orchestrates retrieval + LLM answer with graceful fallback.
    If context is partial or weak, still generate an informative answer.
    """
    docs, sources, pages = await search_similar_documents(query)
    has_context = bool(docs and any(d.page_content.strip() for d in docs))

    if has_context:
        # Build combined context
        context_text = "\n\n".join([d.page_content for d in docs[:6]])
        system_prompt = (
            "You are BRAG AI, a professional assistant. "
            "Answer clearly using retrieved context below when relevant. "
            "If context seems partial, fill in missing parts from general knowledge "
            "but clearly indicate what is supported vs. general.\n\n"
            f"---\n{context_text}\n---"
        )
    else:
        # True no context fallback
        system_prompt = (
            "You are BRAG AI, a professional assistant. "
            "No relevant context was found in the documents. "
            "Answer the user's question using your general knowledge in a neutral, factual tone. "
            "Avoid speculation and do not mention missing data explicitly."
        )

    response = await run_llm_chat(
        system_prompt=system_prompt,
        user_message=query,
        history=history or [],
    )

    # Prepare structured references (if any)
    references = ""
    if has_context and sources:
        refs = "; ".join([f"<{s}, p. {','.join(map(str, pages))}>" for s in sources])
        references = f"\n\nReferences: {refs}"

    return response + references


# Fast-path detection for greetings and capability questions
FASTPATH_GREET = {"hi", "hello", "hey", "hola", "yo", "hiya"}

def is_smalltalk_or_capability(q: str) -> bool:
    """RAG-DEBUG: Enhanced intent gate for greetings and capability questions."""
    t = (q or "").strip().lower()
    if not t:
        return False
    
    # Fast-path greetings
    if t in FASTPATH_GREET:
        return True
    
    # Enhanced capability detection patterns
    capability_patterns = [
        "what can you do",
        "what do you do", 
        "who are you",
        "what are you",
        "how can you help",
        "what is your purpose",
        "what services do you provide",
        "what functions do you have",
        "tell me about yourself",
        "introduce yourself"
    ]
    
    # Simple greeting patterns (start of query)
    greeting_starts = ["hi ", "hello ", "hey ", "good morning", "good afternoon", "good evening"]
    
    # Check capability patterns
    for pattern in capability_patterns:
        if pattern in t:
            return True
    
    # Check greeting starts
    for greeting in greeting_starts:
        if t.startswith(greeting):
            return True
    
    # Single word help/greeting
    if t in ["help", "hi", "hello", "hey", "hola", "yo", "hiya"]:
        return True
    
    return False


def fit_context(messages: List[Dict[str, str]], max_tokens_for_context: int = 120000) -> List[Dict[str, str]]:
    """
    Fit messages within context limits by truncating if necessary.
    
    Args:
        messages: List of message dictionaries
        max_tokens_for_context: Maximum tokens to reserve for context (default 120k for gpt-5-mini)
        
    Returns:
        Truncated messages that fit within context limits
    """
    # Simple token estimation: ~4 characters per token (conservative estimate)
    def estimate_tokens(text: str) -> int:
        return len(text) // 4
    
    # Calculate current token usage
    total_tokens = sum(estimate_tokens(msg.get('content', '')) for msg in messages)
    
    if total_tokens <= max_tokens_for_context:
        return messages
    
    logger.warning(f"Context too large ({total_tokens} tokens), truncating to fit {max_tokens_for_context} tokens")
    
    # Start with system message (always keep)
    result = [messages[0]] if messages else []
    remaining_tokens = max_tokens_for_context - estimate_tokens(messages[0].get('content', '')) if messages else max_tokens_for_context
    
    # Keep user messages from most recent to oldest, but truncate content if needed
    user_messages = [msg for msg in messages[1:] if msg.get('role') == 'user']
    
    for msg in reversed(user_messages):  # Start with most recent
        msg_tokens = estimate_tokens(msg.get('content', ''))
        
        if msg_tokens <= remaining_tokens:
            result.insert(1, msg)  # Insert after system message
            remaining_tokens -= msg_tokens
        else:
            # Truncate this message
            content = msg.get('content', '')
            max_chars = remaining_tokens * 4  # Convert back to characters
            
            if max_chars > 100:  # Only truncate if we can keep meaningful content
                truncated_content = content[:max_chars] + "... [truncated]"
                truncated_msg = msg.copy()
                truncated_msg['content'] = truncated_content
                result.insert(1, truncated_msg)
            
            break  # Stop adding more messages
    
    return result


def strip_markdown(text: str) -> str:
    """Strip markdown formatting from text for plain text fallback."""
    # Remove headers
    text = re.sub(r'^#+\s*', '', text, flags=re.MULTILINE)
    # Remove bold/italic
    text = re.sub(r'\*\*(.*?)\*\*', r'\1', text)
    text = re.sub(r'\*(.*?)\*', r'\1', text)
    # Remove links but keep text
    text = re.sub(r'\[([^\]]+)\]\([^)]+\)', r'\1', text)
    # Remove code blocks
    text = re.sub(r'```.*?```', '', text, flags=re.DOTALL)
    text = re.sub(r'`([^`]+)`', r'\1', text)
    # Remove bullet points
    text = re.sub(r'^[\s]*[-*+]\s*', '', text, flags=re.MULTILINE)
    # Clean up extra whitespace
    text = re.sub(r'\n\s*\n', '\n\n', text)
    text = text.strip()
    return text


class RAGOrchestrator:
    """Main RAG orchestration service."""
    
    def __init__(self):
        self.query_analyzer = QueryComplexityAnalyzer()
        self.reranker = DocumentReranker()
        self.retrieval_service = AdaptiveRetrievalService()
        self.current_document: Optional[Dict[str, Any]] = None
        # Initialize memory manager with embedding function
        from app.modules.lawfirmchatbot.services.embeddings import embed_text
        self.memory = ChatMemoryManager(embed_text)
        # Initialize docgen state (keyed by conversation_id)

    async def process_document(self, file_path: str, filename: str) -> int:
        """Process and index a document."""
        try:
            logger.info(f"Processing document: {filename}")
            
            # Process document into chunks
            documents = process_document(file_path, filename)
            if not documents:
                raise ValueError("No content extracted from document")
            
            # Store document metadata
            total_pages = max([doc.metadata.get('page', 1) for doc in documents])
            self.current_document = {
                'name': filename,
                'type': 'document',
                'total_pages': total_pages,
                'timestamp': datetime.now().isoformat(),
                'chunks_count': len(documents)
            }
            
            # Add to vector store
            chunks_stored = await add_documents_to_vector_store(documents)
            
            logger.info(f"Successfully processed {filename}: {chunks_stored} chunks indexed")
            return chunks_stored
            
        except Exception as e:
            logger.error(f"Error processing document {filename}: {str(e)}", exc_info=True)
            raise

    async def get_answer(self, query: str) -> QueryResponse:
        """Get answer for a query using optimized RAG with fast intent gate."""
        try:
            # Fast-path for greetings and capability questions (no retrieval, sub-second)
            if is_smalltalk_or_capability(query):
                smalltalk_answer = """## How I can help
- Answer questions using your uploaded documents (Qdrant) with citations.
- Summarize, compare, outline steps, extract key points, draft memos.
- If the context is thin, I give a brief overview and mark limits.

**References**: N/A"""
                return QueryResponse(
                    success=True,
                    answer=smalltalk_answer,
                    answer_markdown=smalltalk_answer,
                    metadata={"mode": "fastpath_greeting", "sources": [], "page_references": "N/A"}
                )
            
            # Check if there are any documents in Qdrant
            from app.modules.lawfirmchatbot.services.vector_store import points_count, get_qdrant_client
            from app.modules.lawfirmchatbot.services.retrieval.vector_search import search_similar_documents
            from app.modules.lawfirmchatbot.services.adaptive_retrieval_service import AdaptiveRetrievalService
            from app.modules.lawfirmchatbot.services.llm import chat_completion
            from core.config import settings
            
            client = get_qdrant_client()
            if points_count(client) <= 0:
                return QueryResponse(
                    success=False,
                    answer="⚠️ No document has been uploaded yet. Please upload a document first.",
                    answer_markdown="⚠️ No document has been uploaded yet. Please upload a document first.",
                    metadata={"error": "no_document_uploaded"},
                    debug_info=None
                )
            
            # Fast intent gate
            retrieval_service = AdaptiveRetrievalService()
            should_skip, reason = retrieval_service.should_skip_retrieval(query)
            
            if should_skip:
                if reason == "small_talk":
                    answer = ("I'm your legal research assistant. I can help you find information from your uploaded documents. "
                             "Try asking about specific legal concepts, definitions, or comparisons. "
                             "Example: 'What is contract liability?' or 'Compare tort vs contract law.'")
                else:
                    answer = "I focus on legal document research. Please ask about concepts from your uploaded materials."
                
                return QueryResponse(
                    success=True,
                    answer=answer,
                    answer_markdown=f"# Answer\n{answer}",
                    metadata={"mode": reason}
                )
            
            # Adaptive top_k based on query characteristics
            top_k = retrieval_service.get_adaptive_top_k(query)
            
            # Search with deduplication and fast retrieval
            documents, unique_sources, unique_pages = await search_similar_documents(
                query, 
                k=top_k, 
                score_threshold=settings.QDRANT_SCORE_THRESHOLD
            )
            
            if not documents:
                # Thin-context fallback: provide a brief helpful overview instead of refusal
                fallback = """## Overview
Based on limited retrieved snippets, here's a concise overview.

- Summary: (context appears limited for a full answer).
- If you can specify the chapter/topic or provide more details, I'll refine it.

**References**: limited matches"""
                return QueryResponse(
                    success=True,
                    answer=fallback,
                    answer_markdown=fallback,
                    metadata={"mode": "thin_context_fallback", "hits": 0, "sources": [], "page_references": ""}
                )
            
            # Convert documents to chunks format for prompt building
            chunks = []
            for doc in documents:
                metadata = doc.metadata
                chunks.append({
                    "text": doc.page_content,
                    "source": metadata.get("source", "Document"),
                    "page": metadata.get("page"),
                    "document": metadata.get("document", metadata.get("source", "Document"))
                })
            
            # Build structured prompt
            user_prompt = self.build_user_prompt(query, chunks, unique_sources, unique_pages)
            
            # Create messages
            messages = [
                {"role": "system", "content": self._get_system_prompt()},
                {"role": "user", "content": user_prompt}
            ]
            
            # Get LLM response with guardrail
            answer = await chat_completion(messages, is_legal_query=True, max_tokens=650, temperature=0.5)
            
            # Guard against citations-only responses using the new helper
            if looks_like_citations_only(answer):
                logger.warning("Detected citations-only response, retrying with guardrail")
                retry_prompt = (
                    "Your previous reply looked like citations only. Provide a concise explanation first (3–6 sentences or bullets) "
                    "and then a small **References** block.\n\n"
                    + user_prompt
                )
                retry_messages = [
                    {"role": "system", "content": self._get_system_prompt()},
                    {"role": "user", "content": retry_prompt}
                ]
                answer = await chat_completion(retry_messages, is_legal_query=True, max_tokens=650, temperature=0.4)
            
            # Ensure Sources section exists if we have sources
            if unique_sources and "## Sources" not in answer:
                sources_section = "\n\n## Sources\n"
                for i, source in enumerate(unique_sources[:5], 1):  # Limit to 5 sources
                    sources_section += f"[{i}] {source}\n"
                answer += sources_section
            
            return QueryResponse(
                success=True,
                answer=answer,
                answer_markdown=answer,
                metadata={
                    "hits": len(documents),
                    "sources": unique_sources,
                    "referenced_pages": unique_pages,
                    "page_references": f"Pages: {', '.join(map(str, unique_pages))}" if unique_pages else "No page references available",
                    "mode": "rag",
                    "top_k": top_k
                }
            )
            
        except Exception as e:
            logger.error(f"Error generating answer: {str(e)}", exc_info=True)
            return QueryResponse(
                success=False,
                answer=f"I encountered an error while processing your query: {str(e)}",
                answer_markdown=f"I encountered an error while processing your query: {str(e)}",
                metadata={"error": str(e)}
            )

    def _looks_like_citations_only(self, txt: str) -> bool:
        """Check if the response is citations-only without real answer."""
        t = (txt or "").strip().lower()
        if not t:
            return True
        # Typical failures: starts with "citations", or contains no letters except "citations"
        only_cites = t.startswith("citations") or (t.replace("citations:", "").strip() == "")
        # also treat very short strings plus "citations" as failure
        return only_cites or (len(t) < 60 and "citation" in t)
    
    def _get_system_prompt(self) -> str:
        """BRAG AI - Final Answer Composer System Prompt (Rich Markdown)."""
        return """SYSTEM: BRAG AI — Final Answer Composer (Rich Markdown)

BEHAVIOR
- If retrieved_context contains usable facts, draw from it and add a single one-line "References:" at the end.
- If retrieved_context is empty/irrelevant, DO NOT use "Limited information" or apologize. Begin with:
  "I couldn't find anything similar in the uploaded documents, but here's what I can share more generally—"
  Then answer normally; OMIT the References line.

HARD REQUIREMENTS
1) Length: 350–400 words total.
2) Tone: professional, confident, dynamic; first sentence is a clear takeaway.
3) Headings:
   - Use H2 for the main title (## …) tailored to the query.
   - Use H3 for sections (### …).
   - For comparisons, ALSO use H4 for per-approach Pros/Cons (#### Pros, #### Cons).
   - Never start a heading with "Understanding".
4) Structure by intent:
   A) COMPARISON / "difference between / vs.":
      - Short lead-in (1–2 sentences).
      - **Mandatory table** summarizing key aspects (at least 3 rows).
      - Per-approach blocks with H4 **Pros** and **Cons** as bullet lists (2–4 bullets each).
      - Optional "Bottom Line" paragraph.
      - If you used context, include inline [n] markers near facts and put specific page(s) for each approach in the final References line.
   B) EXPLAIN / DEFINE / PROCEDURE:
      - 2–3 H3 sections chosen to fit (e.g., ### Key Idea, ### How It Works, ### Practical Notes).
      - Use bullets for lists and numbered lists for steps.
5) Citations:
   - Only add [1], [2] when you actually used retrieved_context.
   - End with a single line: References: <ACTUAL_DOC_NAME.pdf, p. X–Y>; <ANOTHER_DOC.pdf, p. Z>.
   - CRITICAL: Use the EXACT document filenames from the context headers, NOT generic "Document" label.
   - If no context used, no [n] and no References line.
6) Forbidden anywhere: "Limited Information Available", "See available documents", "As an AI".

FORMATTING RULES
- Tables: standard Markdown `| Aspect | Option A | Option B |` with header separator.
- Bullets: "- "; keep each bullet concise.
- Do not fabricate document titles, pages, or quotes. Merge contiguous pages into ranges.

QUALITY CHECK BEFORE RETURN
- Word count is 350–400.
- Contains H2 title and at least two H3 sections; for comparisons also includes H4 Pros/Cons per approach and a comparison table.
- If any [n] appears, References line exists and is correctly formatted; otherwise it is omitted.
- Headings are dynamic and do not begin with "Understanding"."""

    def build_user_prompt(self, query: str, chunks: list[dict], sources: list[str], pages: list[int], require_explanation: bool = True, extra_instruction: str = "") -> str:
        """Build user prompt with structured context - BRAG AI format."""
        ctx_lines = []
        for i, c in enumerate(chunks, 1):
            title = (c.get("document") or c.get("source") or "Document").strip()
            page = c.get("page") or c.get("page_number") or "N/A"
            text = (c.get("text") or "").strip()
            if not text:
                continue
            ctx_lines.append(f"[{i}] {title} (page {page})\n{text}")

        context = "\n\n".join(ctx_lines) if ctx_lines else "NO_RELEVANT_CONTEXT"
        
        user = f"""USER_QUERY: {query}

RETRIEVED_CONTEXT:
{context}

INSTRUCTIONS - Follow BRAG AI Rich Markdown format:
1. Length: 350–400 words
2. Opening: one direct sentence setting the takeaway
3. Headings:
   - ## Main title tailored to query
   - ### for sections (NOT starting with "Understanding")
   - #### Pros and #### Cons for comparisons
4. Structure:
   - COMPARISON queries: lead-in → mandatory table (3+ rows) → per-approach blocks with #### Pros/#### Cons (2-4 bullets each)
   - EXPLAIN queries: 2-3 ### sections (Key Idea, How It Works, Practical Notes)
5. Citations: [1], [2] inline when using context; single References line at end
6. If context is weak/partial: Use what's available and supplement naturally with general knowledge without explicitly mentioning limited data
7. If truly no context: Answer naturally using general knowledge without mentioning missing documents
8. FORBIDDEN: "Limited Information Available", "I couldn't find anything", "See available documents", "As an AI"
{extra_instruction}"""
        return user

    def _create_user_prompt(self, query: str, chunks) -> str:
        """Create user prompt with chunks - BRAG AI format."""
        chunk_text = "\n\n".join([
            f"[Page {chunk.metadata.get('page', '?')}]: {chunk.page_content}"
            for chunk in chunks
        ])
        
        # Extract unique page numbers for reference
        page_numbers = []
        for chunk in chunks:
            page_num = chunk.metadata.get('page')
            if page_num is not None and page_num not in page_numbers:
                page_numbers.append(page_num)
        page_numbers.sort()
        page_ref_text = ', '.join(map(str, page_numbers)) if page_numbers else 'Unknown'
        
        return f"""Document: {self.current_document['name']}
Total Pages: {self.current_document['total_pages']}

RETRIEVED_CONTEXT:
{chunk_text}

USER_QUERY: {query}

INSTRUCTIONS - Follow BRAG AI Rich Markdown format:
1. Length: 350–400 words
2. Opening: direct sentence with takeaway
3. Headings: ## main title, ### sections, #### Pros/Cons for comparisons
4. Structure:
   - COMPARISON: table (3+ rows) + per-approach #### Pros/#### Cons blocks
   - EXPLAIN: 2-3 ### sections (Key Idea, How It Works, Practical Notes, Examples/Applications)
5. Citations: [p.X] inline; References: <{self.current_document['name']}, p. {page_ref_text}>
6. If context is weak/partial: Use what's available and supplement naturally with general knowledge
7. If truly no context: Answer naturally using general knowledge without mentioning missing documents
8. FORBIDDEN: "Limited Information Available", "I couldn't find anything", "See available documents", "As an AI" """

    async def answer_query(self, query: str, conversation_id: Optional[str] = None, user_id: str = "anon") -> Dict[str, Any]:
        """High-level entry: route between KB-only, KB+Web, or Clarify.

        Does not change router signatures; can be called by existing handler.
        """
        import time
        from core.config import settings  # RAG-DEBUG: Import settings for debug logging
        t_start = time.time()
        t_embed = 0.0
        t_qdrant_search = 0.0
        t_llm_first = 0.0
        t_guardrail_retry = 0.0
        
        # Initialize conversation and memory context
        async with SessionLocal() as db:
            conversation_id = await self.memory.start_or_get_conversation(db, user_id, conversation_id)
            
            # Append the raw user message first (so resolver sees it too)
            await self.memory.append(db, user_id, conversation_id, "user", query)
            
            # Pull recent turns for resolver + prompt
            recent_msgs = await self.memory.get_prompt_messages(db, conversation_id, recent_pairs=5)
            
            # NEW: Resolve pronouns / ellipses using recent history
            resolved_query = await resolve_coref(recent_msgs, query)
            
            # Use resolved query for retrieval
            if resolved_query != query:
                logger.info(f"[coref] Resolved '{query[:50]}...' -> '{resolved_query[:50]}...'")
            
            await db.commit()
        
        # RAG-DEBUG: Comprehensive query logging
        if settings.DEBUG_RAG:
            logger.info(f"[RAG-DEBUG] Starting answer_query: query='{query[:150]}...' (length={len(query)}), conversation_id={conversation_id}")
        state_cache = await get_conversation_state(conversation_id)
        detected_mode_label = detect_mode(resolved_query, state_cache)

        # Detect if user explicitly asked for skeleton/template/placeholders mode
        placeholders = detect_placeholders(resolved_query, state_cache)

        if settings.DEBUG_RAG:
            logger.info(f"[RAG-DEBUG] Detected mode={detected_mode_label}, placeholders={placeholders}")

        # === Intent Resolution Layer ===
        # Apply intent routing BEFORE docgen intake to prevent missing fields prompt
        qdrant = QdrantClient(url=os.getenv("QDRANT_URL"), api_key=os.getenv("QDRANT_API_KEY"))
        priority_context = fetch_ephemeral_priority_context(qdrant, conversation_id)

        if not priority_context:
            priority_context = "Always consider law_docs_v1 (main db)"

        has_ephemeral = priority_context.startswith("Attached file(s) context")
        mentions_attached = _mentions_attached(resolved_query)
        discovered_file_ids: List[str] = _latest_ephemeral_file_ids(qdrant, conversation_id, limit=5) if conversation_id else []
        has_ephemeral_docs = bool(discovered_file_ids)
        if has_ephemeral_docs:
            has_ephemeral = True
        if mentions_attached and has_ephemeral_docs and ATTACHMENT_HINT not in priority_context:
            priority_context = f"{ATTACHMENT_HINT}\n\n{priority_context}"

        intent = decide_intent(resolved_query, has_ephemeral)

        # === Patch: Override docgen when user just wants explanation ===
        if intent == "docgen" and (
            "describe" in resolved_query.lower()
            or "explain" in resolved_query.lower()
            or "summarize" in resolved_query.lower()
            or "contents" in resolved_query.lower()
        ):
            intent = "qa"
            logger.info(f"[intent-routing] Overriding docgen->qa for descriptive query with attachments")

        # --- ROUTING FREEZE: Lock the initial intent (authoritative decision) ---
        # The first classified intent is authoritative and must never be downgraded.
        # Never downgrade docgen → general/qa later in the pipeline because of 
        # placeholders, missing fields, or retrieval results.
        original_intent = intent
        logger.info(f"[routing-freeze] Locked original_intent={original_intent}")
        
        def _prevent_downgrade(curr, target):
            """Prevent downgrading docgen to qa/general after initial classification."""
            if curr == "docgen" and target != "docgen":
                logger.warning(f"[routing-freeze] Attempted downgrade docgen -> {target}; BLOCKED")
                return curr
            return target
        # -----------------------------------------------------------------------------

        # Override detected_mode_label based on intent routing
        if intent == "qa" and detected_mode_label == "docgen":
            detected_mode_label = "general"  # Force QA path
            logger.info(f"[intent-routing] Mode override: docgen->general due to intent routing")
        elif intent == "docgen" and detected_mode_label != "docgen":
            detected_mode_label = "docgen"  # Force docgen path
            logger.info(f"[intent-routing] Mode override: {detected_mode_label}->docgen due to intent routing")

        # Store for debugging
        logger.info(f"[intent-routing] intent={intent} has_ephemeral={has_ephemeral} final_mode={detected_mode_label}")

        # === Fast-path for filename questions ===
        # If user is asking about the attached file name, answer immediately without LLM
        import re
        FILENAME_Q = re.compile(
            r"(?:what\s+is|tell\s+me|give\s+me|name|called)\s+(?:the\s+)?(?:file|doc(?:ument)?)\b",
            re.IGNORECASE,
        )
        
        if has_ephemeral and (FILENAME_Q.search(resolved_query) or "file name" in resolved_query.lower() or "document name" in resolved_query.lower()):
            # Try to get filename from ephemeral context
            try:
                # The filename might be in the priority_context or we need to fetch it from conversation state
                filename_answer = "You have attached a document to this conversation."
                
                # Log for debugging
                logger.info(f"[fast-path] Filename question detected, returning quick answer")
                
                return {
                    "success": True,
                    "answer": filename_answer,
                    "answer_markdown": filename_answer,
                    "metadata": {
                        "strategy": "fast_path_filename",
                        "kb_hits": 0,
                        "web_used": False,
                        "latency_ms": int((time.time() - t_start) * 1000),
                        "conversation_id": conversation_id,
                        "response_type": "text",
                        "model": "fast_path",
                        "has_doc": False
                    }
                }
            except Exception as e:
                logger.warning(f"[fast-path] Filename detection failed: {e}, proceeding with normal flow")

        if detected_mode_label == "tweak_doc" and state_cache.last_document:
            logger.info(f"[DocGen] Tweak request detected; reusing last document (doc_type={state_cache.last_doc_type})")
            prior_doc = state_cache.last_document.strip()
            if len(prior_doc) > 16000:
                prior_doc = prior_doc[:16000]

            conversation_summary = state_cache.summary.strip() if state_cache.summary else ""
            doc_type_label = state_cache.last_doc_type or detect_doc_type(resolved_query) or "legal_document"

            system_prompt = (
                "You are updating an existing Pakistani legal filing. "
                "Preserve the provided HTML structure, headings, numbering, references, and signature blocks. "
                "Only change text necessary to satisfy the user request."
            )

            instruction_block = (
                f"User instruction: {resolved_query}\n\n"
                "Implement the instruction precisely while leaving unrelated portions untouched."
            )

            user_prompt_parts = []
            if conversation_summary:
                user_prompt_parts.append(f"Conversation summary so far:\n{conversation_summary}")
            user_prompt_parts.append(f"Existing document HTML:\n{prior_doc}")
            user_prompt_parts.append(instruction_block)
            user_prompt = "\n\n".join(user_prompt_parts)

            from app.modules.lawfirmchatbot.services.llm import _get_client
            openai_client = _get_client()
            model_doc = os.getenv("DOCGEN_MODEL", "gpt-4o")

            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ]

            response = await chat_completion(
                client=openai_client,
                model=model_doc,
                messages=messages,
                intent="docgen",
                temperature=0.2,
                stream=False,
                extra_params={"max_tokens": 2000},
            )

            if hasattr(response, "choices"):
                revised_html = response.choices[0].message.content or ""
            elif hasattr(response, "output_text"):
                revised_html = response.output_text or ""
            elif hasattr(response, "output"):
                if isinstance(response.output, list):
                    chunks = []
                    for item in response.output:
                        if hasattr(item, "type") and item.type == "output_text":
                            chunks.append(getattr(item, "text", ""))
                        elif hasattr(item, "content"):
                            chunks.append(item.content)
                    revised_html = "\n".join(chunks).strip()
                else:
                    revised_html = str(response.output) if response.output else ""
            else:
                revised_html = ""

            if revised_html.startswith("```"):
                first_newline = revised_html.find("\n")
                if first_newline > 0:
                    revised_html = revised_html[first_newline + 1:]
                if revised_html.endswith("```"):
                    revised_html = revised_html[:-3].rstrip()

            if not revised_html:
                raise RuntimeError("Tweak request failed to generate output.")

            normalized_html = render_html_document(revised_html)
            references_list = list(state_cache.last_references)

            await update_last_document(conversation_id, normalized_html, doc_type_label, references_list)
            await record_response(conversation_id, normalized_html, mode="tweak")
            await _update_summary_with_exchange(conversation_id, state_cache, resolved_query, normalized_html)

            async with SessionLocal() as db:
                await self.memory.append(db, user_id, conversation_id, "assistant", normalized_html)
                await db.commit()

            latency = int((time.time() - t_start) * 1000)
            return {
                "success": True,
                "answer": normalized_html,
                "answer_markdown": normalized_html,
                "metadata": {
                    "mode": "docgen_tweak",
                    "doc_type": doc_type_label,
                    "strategy": "docgen",
                    "kb_hits": 0,
                    "web_used": False,
                    "references": references_list,
                    "latency_ms": latency,
                    "conversation_id": conversation_id,
                    "response_type": "html",
                    "model": model_doc,
                    "has_doc": True
                }
            }

        # --- DOCGEN intake + missing-fields logic (respects placeholders flag) ---
        docgen_context_fields: Dict[str, str] = {}
        if detected_mode_label == "docgen":
            extracted_info = extract_case_info(resolved_query)
            extracted_data = extracted_info.get("data", {}) or {}
            docgen_context_fields = dict(getattr(state_cache, "doc_fields", {}) or {})

            # merge any newly extracted data
            for key, value in extracted_data.items():
                if key not in docgen_context_fields:
                    docgen_context_fields[key] = value
                elif value:
                    docgen_context_fields[key] = value

            state_cache.doc_fields = docgen_context_fields
            filled_count = sum(1 for v in docgen_context_fields.values() if v)

            # If user asked for placeholders/skeleton/template => proceed even if fields < threshold
            if filled_count < 3 and not placeholders:
                missing_fields = [k for k, v in docgen_context_fields.items() if not v]
                missing_list = ", ".join(field.replace("_", " ") for field in missing_fields) or "additional case details"
                ask_text = f"To draft your document, please provide the following missing details: {missing_list}."

                await record_response(conversation_id, ask_text, mode="docgen_missing")
                await _update_summary_with_exchange(conversation_id, state_cache, resolved_query, ask_text)

                async with SessionLocal() as db:
                    await self.memory.append(db, user_id, conversation_id, "assistant", ask_text)
                    await db.commit()

                model_general = os.getenv("GENERAL_MODEL", "gpt-4o-mini")
                latency = int((time.time() - t_start) * 1000)
                return {
                    "success": True,
                    "answer": ask_text,
                    "answer_markdown": ask_text,
                    "metadata": {
                        "mode": "docgen_missing",
                        "strategy": "docgen",
                        "kb_hits": 0,
                        "web_used": False,
                        "latency_ms": latency,
                        "conversation_id": conversation_id,
                        "response_type": "text",
                        "model": model_general,
                        "has_doc": False,
                        "missing_fields": [field.replace("_", " ") for field in missing_fields],
                    }
                }

        # Chit-chat short-circuit: handle greetings/meta queries instantly (use resolved query)
        if is_smalltalk_or_capability(resolved_query):
            if settings.DEBUG_RAG:
                logger.info(f"[RAG-DEBUG] Fast-path: chit-chat detected, skipping retrieval")
            else:
                logger.info(f"[answer_query] Fast-path: chit-chat detected, skipping retrieval")
            smalltalk_answer = """I'm here to help you explore your legal documents and answer questions.

### What I Do

I analyze uploaded documents to answer your questions with precise citations. I can summarize content, compare concepts, explain legal principles, and extract key information from your document library.

### How to Use Me

Ask specific questions about your documents, request comparisons between concepts, or seek summaries of particular topics. Each answer includes references to the source pages where information was found."""
            return {
                "success": True,
                "answer": smalltalk_answer,
                "answer_markdown": smalltalk_answer,
                "metadata": {
                    "strategy": "chit_chat",
                    "kb_hits": 0,
                    "web_used": False,
                    "latency_ms": int((time.time() - t_start) * 1000),
                    "conversation_id": conversation_id,
                    "response_type": "text",
                    "model": os.getenv("GENERAL_MODEL", "gpt-4o-mini"),
                    "has_doc": False
                }
            }
        
        # === INTELLIGENT ROUTING: Detect mode (QA vs DocGen) ===
        # Mode and intent are already set by the early intent routing logic above
        # This section is kept for backward compatibility with variables used downstream
        is_docgen = detected_mode_label == "docgen"
        mode = "docgen" if is_docgen else "qa"
        
        # Retrieval sizing based on mode
        if is_docgen:
            k = 6  # Reduced from 10 - enough for template context without timeout
            score_thresh = 0.20
        else:
            k = 4  # fast QA
            score_thresh = 0.22
        
        if settings.DEBUG_RAG:
            logger.info(f"[RAG-DEBUG] Mode detected: {mode}, top_k={k}, score_threshold={score_thresh}")
        
        # === Attachment-Aware Scope Detection with Auto-Discovery ===
        # Determine if user is asking about attached files specifically
        # Pass Qdrant client and conversation_id for auto-discovery of file_ids
        eph_files = []  # Will be populated from UI when available
        attachment_scope, filter_file_ids = _attachment_scope(
            resolved_query,
            eph_files,
            conversation_id=conversation_id,
            qdrant_client=qdrant,
        )
        attachment_reason = f"router:{attachment_scope}"
        if mentions_attached and has_ephemeral_docs:
            attachment_scope = "ephemeral_only"
            filter_file_ids = discovered_file_ids or filter_file_ids
            attachment_reason = "attachment mention"
            logger.info("[attachment-scope] Attachment mention detected; forcing ephemeral-only scope")
        elif not filter_file_ids and has_ephemeral_docs:
            filter_file_ids = discovered_file_ids
            attachment_reason = "auto-discovered ephemerals"

        logger.info(
            "[attachment-scope] Query scope: %s (reason=%s), filter_file_ids: %s",
            attachment_scope,
            attachment_reason,
            filter_file_ids,
        )
        
        # Retrieve more candidates for strategy decision (use resolved query)
        # === Ephemeral-First Retrieval Policy ===
        
        use_retriever = should_use_retriever(detected_mode_label, resolved_query)
        
        # CRITICAL: Never skip retrieval when ephemerals exist or attachment keywords detected
        has_ephemeral = (
            has_ephemeral
            or has_ephemeral_docs
            or attachment_scope == "ephemeral_only"
            or bool(filter_file_ids)
        )
        if has_ephemeral:
            use_retriever = True  # FORCE retrieval when attachments exist
            logger.info(f"[retrieval-gate] FORCING retrieval due to ephemeral presence")
        
        initial_chunks = []
        ephemeral_chunks = []
        total_hits = 0
        strong_hits = 0
        max_score = 0.0
        t_qdrant_search = 0.0
        use_only_ephemeral = False

        if use_retriever:
            t_embed_start = time.time()
            query_embedding = await embed_text_async(resolved_query)
            t_qdrant_search += time.time() - t_embed_start

            # Step 1: Always check ephemeral first if conversation exists
            if conversation_id:
                try:
                    if ephemeral_collection_exists(conversation_id):
                        has_ephemeral = True
                        eph_start = time.time()
                        eph_hits = await semantic_search_ephemeral(
                            qdrant,
                            query_embedding,
                            max(8, k),  # Get more ephemeral candidates
                            conversation_id=conversation_id,
                        )
                        t_qdrant_search += time.time() - eph_start

                        for hit in eph_hits:
                            text = (hit.get("text") or "").strip()
                            if not text:
                                continue
                            payload = hit.get("metadata") or hit.get("payload") or {}
                            payload_file_id = payload.get("file_id") or payload.get("doc_id")
                            if filter_file_ids and payload_file_id not in filter_file_ids:
                                continue
                            metadata = {
                                "source": payload.get("document") or payload.get("source") or "Ephemeral Attachment",
                                "document": payload.get("document") or payload.get("source") or "Ephemeral Attachment",
                                "page": payload.get("page")
                                or payload.get("page_number")
                                or payload.get("idx"),
                                "idx": payload.get("idx"),
                                "conversation_id": payload.get("conversation_id"),
                                "similarity_score": float(hit.get("score") or 0.0),
                                "ephemeral": True,
                            }
                            ephemeral_chunks.append(Document(page_content=text, metadata=metadata))
                except Exception as exc:
                    logger.warning(
                        "[answer_query] Ephemeral retrieval failed for conversation %s: %s",
                        conversation_id,
                        exc,
                    )
            
            # Step 2: Blend ephemeral context with KB while respecting intent
            ephemeral_hits = len(ephemeral_chunks)
            if ephemeral_hits:
                has_ephemeral = True

            kb_chunks = []
            should_fetch_kb = True
            if attachment_scope == "ephemeral_only" and intent != "docgen" and ephemeral_hits > 0:
                should_fetch_kb = False

            if should_fetch_kb:
                kb_k = k
                if k and ephemeral_hits:
                    kb_k = max(1, k - min(ephemeral_hits, max(0, k - 1)))
                kb_start = time.time()
                search_result = await search_similar_documents(
                    resolved_query,
                    k=kb_k,
                    score_threshold=score_thresh,
                )
                t_qdrant_search += time.time() - kb_start

                if isinstance(search_result, tuple):
                    kb_chunks, _, _ = search_result
                else:
                    kb_chunks = search_result
                kb_chunks = list(kb_chunks or [])

                if intent == "docgen":
                    logger.info(
                        "[retrieval-policy] DocGen enforced KB backfill (ephemeral_hits=%d, kb_chunks=%d)",
                        ephemeral_hits,
                        len(kb_chunks),
                    )
                elif not ephemeral_hits:
                    logger.info(
                        "[retrieval-policy] No ephemeral hits - using KB only (%d chunks)",
                        len(kb_chunks),
                    )
                else:
                    logger.info(
                        "[retrieval-policy] Ephemeral hits with KB backfill (kb_chunks=%d)",
                        len(kb_chunks),
                    )
            else:
                logger.info(
                    "[retrieval-policy] Attachment scope staying ephemeral-only (hits=%d)",
                    ephemeral_hits,
                )

            initial_chunks = []
            if ephemeral_hits:
                initial_chunks.extend(ephemeral_chunks)
            if should_fetch_kb and kb_chunks:
                initial_chunks.extend(kb_chunks)

            use_only_ephemeral = bool(ephemeral_hits) and (not should_fetch_kb or not kb_chunks)

            if attachment_scope == "ephemeral_only" and not use_only_ephemeral and intent != "docgen":
                use_only_ephemeral = True
                initial_chunks = list(ephemeral_chunks)
                logger.info(
                    "[retrieval-policy] Attachment scope forced ephemeral-only mode even without hits"
                )

            if not initial_chunks and intent == "docgen" and should_fetch_kb:
                logger.warning("[retrieval-policy] DocGen retrieval empty after KB backfill attempt")

            t_embed = t_qdrant_search  # Includes embedding + search time

            total_hits = len(initial_chunks)
            scores = [float(ch.metadata.get("similarity_score", 0.0)) for ch in initial_chunks]
            strong_hits = sum(1 for s in scores if s >= 0.25)
            max_score = max(scores) if scores else 0.0

            logger.info(
                f"[answer_query] Retrieval complete: "
                f"total_hits={total_hits}, strong_hits={strong_hits}, "
                f"max_score={max_score:.3f}, ephemeral_only={use_only_ephemeral}, "
                f"t_qdrant={t_qdrant_search*1000:.0f}ms"
            )
            
            # CRITICAL FALLBACK: If no chunks found but ephemerals exist, collect raw text
            if not initial_chunks and has_ephemeral and conversation_id:
                logger.warning(f"[ephemeral-fallback] No chunks retrieved but ephemerals exist - collecting raw text")
                fallback_text = await _collect_ephemeral_fallback_text(
                    qdrant, conversation_id, filter_file_ids, max_chars=4000
                )
                
                if fallback_text:
                    # Create synthetic chunk from fallback text
                    metadata = {
                        "source": "Attached Document (fallback)",
                        "document": "Attached Document",
                        "similarity_score": 1.0,  # Perfect score since this is direct retrieval
                        "ephemeral": True,
                        "fallback": True,
                    }
                    initial_chunks = [Document(page_content=fallback_text, metadata=metadata)]
                    total_hits = 1
                    strong_hits = 1
                    max_score = 1.0
                    logger.info(f"[ephemeral-fallback] Synthesized {len(fallback_text)} chars as context")
        else:
            logger.info("[answer_query] Retrieval skipped; using conversation context only")
            t_embed = 0.0
        retrieval_stats = {"total_hits": total_hits, "strong_hits": strong_hits, "max_score": max_score}
        
        # ROUTING FREEZE: Prevent re-classification from overriding the initial intent
        # Use original_intent if already locked, otherwise classify now (fallback for edge cases)
        reclassified_intent = classify_intent(resolved_query)
        intent = _prevent_downgrade(original_intent, reclassified_intent)
        if reclassified_intent != intent:
            logger.info(f"[routing-freeze] Re-classification attempted {original_intent} -> {reclassified_intent}, kept {intent}")
        
        strategy = select_strategy(resolved_query, retrieval_stats)

        use_web = False
        web_results: List[Dict[str, Any]] = []
        web_summary = ""

        # Optionally call web search (use resolved query)
        if strategy == "kb_plus_web" and (settings.WEB_SEARCH_ENABLED is True or (settings.WEB_SEARCH_ENABLED is None and bool(settings.TAVILY_API_KEY))):
            use_web = True
            try:
                from app.services.web_search.tavily_client import search_and_summarize as tavily_search
                web_sources, web_summary = await tavily_search(resolved_query, top_k=3)
                web_results = [
                    {"url": s.url, "title": s.title, "snippet": s.snippet, "score": s.score}
                    for s in web_sources
                ]
            except Exception:
                # suppress to user
                use_web = False

        # Build CONTEXT block from KB + optional web
        numbered_context: List[str] = []
        source_list: List[Tuple[str, str]] = []  # (key, title)

        # === MODE-AWARE CONTEXT BUILDING ===
        # Adjust context limits based on mode
        if mode == "docgen":
            MAX_TOTAL_CONTEXT_CHARS = 4000  # Reduced to prevent timeouts (was 8000)
            MAX_CHUNK_CHARS = 700  # Shorter chunks for faster processing
            # Stitch contiguous pages from same document for better template context
            stitched = []
            by_key = {}
            for ch in initial_chunks[:k * 2]:  # Allow more candidates for stitching
                src = ch.metadata.get("source") or ch.metadata.get("document") or "Source"
                page = ch.metadata.get("page") or ch.metadata.get("page_number")
                key = f"{src}"
                page_num = int(page) if isinstance(page, (int, float)) else -1
                by_key.setdefault(key, []).append((page_num, ch))
            
            # Sort and stitch contiguous pages
            for key, arr in by_key.items():
                arr.sort(key=lambda x: x[0])
                block = []
                prev = None
                for p, ch in arr:
                    if prev is None or (p != -1 and p == prev + 1):
                        block.append(ch)
                    else:
                        if block:
                            stitched.append(block)
                        block = [ch]
                    prev = p
                if block:
                    stitched.append(block)
            
            # Flatten: prefer biggest stitched blocks first for docgen
            stitched.sort(key=lambda blk: sum(len((c.page_content or "")) for c in blk), reverse=True)
            final_chunks = []
            for blk in stitched:
                final_chunks.extend(blk)
        else:
            MAX_TOTAL_CONTEXT_CHARS = 2800  # Keep context short for QA
            MAX_CHUNK_CHARS = 500  # Trim chunks for speed
            final_chunks = initial_chunks[:6]
        
        total_context_chars = 0
        
        for i, ch in enumerate(final_chunks, start=1):
            src = ch.metadata.get("source") or ch.metadata.get("document") or "Document"
            _page = ch.metadata.get("page") or ch.metadata.get("page_number") or ch.metadata.get("page_index")
            # Format header with clear document name
            if isinstance(_page, (int, float)):
                header = f"[{i}] Document: {src}, Page: {int(_page)}"
            else:
                header = f"[{i}] Document: {src}"
            
            # Trim chunk with smart truncation (preserve section headers where possible)
            content = (ch.page_content or "").strip()
            if len(content) > MAX_CHUNK_CHARS:
                # Try to find a good break point (sentence end, paragraph break)
                truncate_at = MAX_CHUNK_CHARS
                for break_char in ['. ', '\n\n', '\n']:
                    break_pos = content.rfind(break_char, 0, MAX_CHUNK_CHARS)
                    if break_pos > MAX_CHUNK_CHARS * 0.7:  # At least 70% of desired length
                        truncate_at = break_pos + len(break_char)
                        break
                content = content[:truncate_at].rstrip() + "..."
            
            chunk_text = f"{header}\n{content}"
            
            # Check if adding this chunk would exceed total limit
            if total_context_chars + len(content) > MAX_TOTAL_CONTEXT_CHARS and numbered_context:
                if settings.DEBUG_RAG:
                    logger.info(f"[RAG-DEBUG] Context limit reached, stopping at chunk {i-1} (total_chars={total_context_chars})")
                break
                
            numbered_context.append(chunk_text)
            total_context_chars += len(content)
            key = f"{src}|{_page}" if _page is not None else src
            source_list.append((key, str(src)))
        
        if state_cache.pinned_facts:
            fact_lines = "\n".join([f"- {k}: {v}" for k, v in state_cache.pinned_facts.items() if v])
            if fact_lines:
                numbered_context.append(f"[F] Conversation Facts\n{fact_lines}")


        # RAG-DEBUG: Context validation before LLM call
        if settings.DEBUG_RAG:
            logger.info(f"[RAG-DEBUG] Context validation: num_chunks={len(numbered_context)}, total_chars={total_context_chars}")
            for i, ctx in enumerate(numbered_context[:3], 1):  # Log first 3 contexts
                snippet = ctx[:200] + "..." if len(ctx) > 200 else ctx
                logger.info(f"[RAG-DEBUG] Context {i}: {snippet}")
        
        # Note: Removed hard minimum context threshold - let LLM work with partial context
        # The system will now use whatever context is available and supplement with general knowledge
        if total_context_chars < 600:
            logger.info(f"[answer_query] Context is small ({total_context_chars} chars) but proceeding with LLM - will use partial context + general knowledge")

        if use_web:
            base = len(numbered_context)
            for j, w in enumerate(web_results, start=1):
                header = f"[{base + j}] {w['title']} — {w['url']}"
                numbered_context.append(f"{header}\n{w['snippet']}")
                source_list.append((w["url"], w["title"]))

        if not numbered_context and (state_cache.summary or state_cache.last_document):
            if state_cache.summary:
                numbered_context.append(f"[S] Conversation Summary\n{state_cache.summary}")
            if state_cache.last_document:
                doc_snippet = state_cache.last_document[:2000]
                numbered_context.append(f"[D] Last Document Snapshot\n{doc_snippet}")

        # Only return "no info" if truly zero snippets
        if not numbered_context:
            logger.warning(f"[answer_query] Zero snippets available, using smart fallback")
            # Use the new generate_answer function for smarter fallback
            fallback_response = await generate_answer(resolved_query, recent_msgs)
            await record_response(conversation_id, fallback_response, mode="qa")
            await _update_summary_with_exchange(conversation_id, state_cache, resolved_query, fallback_response)
            return {
                "success": True,
                "answer": fallback_response,
                "answer_markdown": fallback_response,
                "metadata": {
                    "strategy": "smart_fallback",
                    "kb_hits": total_hits,
                    "web_used": use_web,
                    "latency_ms": int((time.time() - t_start) * 1000),
                    "conversation_id": conversation_id,
                    "response_type": "text",
                    "model": os.getenv("GENERAL_MODEL", "gpt-4o-mini"),
                    "has_doc": False,
                }
            }
        num_chunks_sent = len(numbered_context)
        estimated_prompt_tokens = total_context_chars // 4  # Rough estimate
        logger.info(f"[answer_query] Context: num_chunks={num_chunks_sent}, total_chars={total_context_chars}, est_tokens={estimated_prompt_tokens}")

        # RAG-DEBUG: Compose messages with enhanced prompt to prevent citations-only responses
        # Log first 300 chars of each context chunk for debugging
        if settings.DEBUG_RAG:
            logger.info(f"[RAG-DEBUG] About to send {len(numbered_context)} context chunks to LLM:")
            for idx, ctx in enumerate(numbered_context[:5], 1):  # Log first 5
                preview = ctx[:300].replace('\n', ' ')
                logger.info(f"[RAG-DEBUG]   Chunk {idx}: {preview}...")
        
        # Optionally add semantic recall snippets (use resolved query)
        recall_snippets = await self.memory.semantic_recall(resolved_query, user_id, k=3)
        if recall_snippets:
            base = len(numbered_context)
            for j, r in enumerate(recall_snippets, start=1):
                header = f"[MR{j}] Prior Chat"
                numbered_context.append(f"{header}\n{r['text']}")
        
        # === MODE-AWARE PROMPTING AND MODEL SELECTION ===
        model_general = os.getenv("GENERAL_MODEL", "gpt-4o-mini")
        model_doc = os.getenv("DOCGEN_MODEL", "gpt-4o")
        detected_doc_type: Optional[str] = None
        references_list: List[str] = []
        raw_markdown = ""
        final_md = ""
        t_llm_first = 0.0
        extra_llm_params: Dict[str, Any] = {}
        intent_label = "general"
        temperature = 0.2
        use_stream = True

        from app.modules.lawfirmchatbot.services.llm import _get_client
        openai_client = _get_client()

        # === System Prompt Construction ===
        # Use the priority context we already fetched above
        system_prompt = build_system_prompt(priority_context)

        # ROUTING FREEZE ENFORCEMENT: The intent variable is now frozen and authoritative.
        # NO CODE BELOW THIS POINT should modify the 'intent' variable.
        # Any logic that would change intent must use _prevent_downgrade() instead.
        mode = intent  # Update mode variable to match frozen intent
        
        # Safety check: verify intent hasn't been corrupted
        if intent != original_intent:
            logger.error(f"[routing-freeze] CRITICAL: intent={intent} != original_intent={original_intent}! This should never happen.")
            intent = original_intent  # Force back to original
            mode = intent

        # Initialize model variables safely (NEVER leave unset)
        model_general = os.getenv("GENERAL_MODEL", "gpt-4o-mini")
        model_doc = os.getenv("DOCGEN_MODEL", "gpt-4o")
        
        # Set target_model based on frozen intent (default to QA model for safety)
        if intent == "docgen":
            target_model = model_doc
        else:
            target_model = model_general
        
        # Initialize other variables that might be used in logging
        intent_label = intent
        use_stream = True
        extra_llm_params = {}
        messages = []  # CRITICAL: Initialize to prevent UnboundLocalError

        # === DOCGEN PATH: Execute docgen handler and proceed to LLM call ===
        # If intent is docgen, we build docgen-specific prompts below
        if intent == "docgen":
            detected_doc_type = detect_doc_type(resolved_query)
            # For docgen, we can still use the priority context but combine with docgen-specific prompts
            docgen_system_prompt = get_docgen_prompt(detected_doc_type)
            # Inject priority context into docgen prompt as well
            system_prompt = f"{system_prompt}\n\n{docgen_system_prompt}"
            docgen_answers = {"user_request": resolved_query}
            if docgen_context_fields:
                docgen_answers.update({k: v for k, v in docgen_context_fields.items() if v})
            if recall_snippets:
                docgen_answers["prior_chat"] = " | ".join(r["text"] for r in recall_snippets[:2])
            if source_list:
                ordered_titles = list(dict.fromkeys(title for _, title in source_list))
                if ordered_titles:
                    docgen_answers["retrieved_sources"] = "; ".join(ordered_titles[:4])
            references_list = []
            seen_refs = set()
            for ch in final_chunks[:3]:
                src = ch.metadata.get("source") or ch.metadata.get("document") or ch.metadata.get("title")
                page = ch.metadata.get("page") or ch.metadata.get("page_number")
                if not src:
                    continue
                label = f"{src}, p. {int(page)}" if isinstance(page, (int, float)) else str(src)
                if label in seen_refs:
                    continue
                seen_refs.add(label)
                references_list.append(label)
            user_prompt = build_docgen_prompt(resolved_query, docgen_answers, final_chunks)
            history_messages = [
                msg for msg in (recent_msgs or [])
                if msg.get("role") in ("user", "assistant")
            ][-4:]
            messages = [
                {"role": "system", "content": system_prompt}
            ] + history_messages + [
                {"role": "user", "content": user_prompt},
            ]
            messages = fit_context(messages, max_tokens_for_context=120000)
            intent_label = "docgen"
            use_stream = False
            target_model = model_doc
            if settings.DEBUG_RAG:
                total_prompt_chars = sum(len(m.get("content", "")) for m in messages)
                logger.info(f"[RAG-DEBUG] DocGen type={detected_doc_type}, model={target_model}")
                logger.info(f"[RAG-DEBUG] DocGen prompt messages={len(messages)}, total_chars={total_prompt_chars}, est_tokens={total_prompt_chars//4}")
                logger.info(f"[RAG-DEBUG] DocGen system preview: {messages[0]['content'][:200]}...")
                logger.info(f"[RAG-DEBUG] DocGen user preview: {messages[-1]['content'][:500]}...")
        
        elif intent == "qa":
            # QA mode - system_prompt already set above with priority context
            target_model = model_general
            user_prompt = f"""Using the CONTEXT, answer the QUESTION concisely but with structure.

CONTEXT:
{chr(10).join(numbered_context)}

QUESTION: {query}

OUTPUT RULES:
- Start with a direct one-sentence answer.
- Then use dynamic H2/H3 headings that fit the topic (e.g., Key Points, How It Works, Pros/Cons, Risks, Examples).
- For comparisons/differences: include a **markdown table** first, then bullets.
- Keep 8-14 sentences total for complex topics.
- Use inline bracketed refs [1], [2] and end with **References:** doc/page list.
- No boilerplate like "Limited information available."
- Do NOT format the answer as a legal document, petition, or notice.
"""
            extra_llm_params = {"max_tokens": 850}
            async with SessionLocal() as db:
                qa_recent_msgs = await self.memory.get_prompt_messages(db, conversation_id, recent_pairs=5)
            messages = [
                {"role": "system", "content": system_prompt}
            ] + qa_recent_msgs + [
                {"role": "user", "content": user_prompt},
            ]
            messages = fit_context(messages, max_tokens_for_context=120000)
            if settings.DEBUG_RAG:
                total_prompt_chars = sum(len(m.get("content", "")) for m in messages)
                logger.info(f"[RAG-DEBUG] Mode={mode}, Model={target_model}, extra_params={extra_llm_params}")
                logger.info(f"[RAG-DEBUG] Sending to LLM: {len(messages)} messages, total_chars={total_prompt_chars}, est_tokens={total_prompt_chars//4}")
                logger.info(f"[RAG-DEBUG] System prompt preview: {messages[0]['content'][:200]}...")
                logger.info(f"[RAG-DEBUG] User prompt preview: {messages[-1]['content'][:500]}...")
        
        else:
            # Other intents (summarize, etc.) - treat as QA
            # ROUTING FREEZE: This branch should NEVER execute for docgen intent
            if intent == "docgen":
                logger.error(f"[routing-freeze] CRITICAL ERROR: docgen intent reached 'else' branch! This is a bug.")
                raise RuntimeError(f"Routing freeze violation: docgen intent reached general/QA branch")
            
            target_model = model_general
            user_prompt = f"""Using the CONTEXT, answer the QUESTION concisely but with structure.

CONTEXT:
{chr(10).join(numbered_context)}

QUESTION: {query}

OUTPUT RULES:
- Start with a direct one-sentence answer.
- Then use dynamic H2/H3 headings that fit the topic (e.g., Key Points, How It Works, Pros/Cons, Risks, Examples).
- For comparisons/differences: include a **markdown table** first, then bullets.
- Keep 8-14 sentences total for complex topics.
- Use inline bracketed refs [1], [2] and end with **References:** doc/page list.
- No boilerplate like "Limited information available."
- Do NOT format the answer as a legal document, petition, or notice.
"""
            extra_llm_params = {"max_tokens": 850}
            async with SessionLocal() as db:
                qa_recent_msgs = await self.memory.get_prompt_messages(db, conversation_id, recent_pairs=5)
            messages = [
                {"role": "system", "content": system_prompt}
            ] + qa_recent_msgs + [
                {"role": "user", "content": user_prompt},
            ]
            messages = fit_context(messages, max_tokens_for_context=120000)
            logger.info(f"[RAG-DEBUG] Other intent={intent}, treating as QA, model={target_model}")

        # Log final routing decision (helpful to debug which model & mode are actually invoked)
        try:
            logger.info(f"[routing] mode={mode} placeholders={placeholders} intent={intent_label} target_model={target_model} use_stream={use_stream} extra_params_keys={list(extra_llm_params.keys())}")
        except Exception:
            # safe-guard in logs
            logger.info(f"[routing] mode={mode} target_model={target_model}")

        t_llm_start = time.time()
        raw_response = await chat_completion(
            client=openai_client,
            model=target_model,
            messages=messages,
            intent=intent_label,
            temperature=temperature,
            stream=use_stream,
            extra_params=extra_llm_params,
        )
        t_llm_first = time.time() - t_llm_start

        if hasattr(raw_response, "choices"):
            raw_markdown = raw_response.choices[0].message.content or ""
        elif hasattr(raw_response, "output_text"):
            raw_markdown = raw_response.output_text or ""
        elif hasattr(raw_response, "output"):
            if isinstance(raw_response.output, list):
                chunks = []
                for item in raw_response.output:
                    if hasattr(item, "type") and item.type == "output_text":
                        chunks.append(getattr(item, "text", ""))
                    elif hasattr(item, "content"):
                        chunks.append(item.content)
                raw_markdown = "\n".join(chunks).strip()
            else:
                raw_markdown = str(raw_response.output) if raw_response.output else ""
        else:
            raw_markdown = ""

        if settings.DEBUG_RAG:
            estimated_tokens = len(raw_markdown) // 4
            logger.info(f"[RAG-DEBUG] LLM call complete: mode={mode}, t_llm={t_llm_first*1000:.0f}ms, response_len={len(raw_markdown)} chars, est_tokens={estimated_tokens}")
            if raw_markdown:
                logger.info(f"[RAG-DEBUG] LLM response preview: {raw_markdown[:300]}...")
            else:
                logger.error("[RAG-DEBUG] LLM returned EMPTY response! This is the core issue.")
        else:
            logger.info(f"[answer_query] LLM first call: t_llm={t_llm_first*1000:.0f}ms, response_len={len(raw_markdown)} chars")

        final_md = raw_markdown.strip()

        if mode == "docgen" and final_md:
            if references_list:
                references_text = ", ".join(references_list)
                if final_md.lstrip().startswith("<"):
                    final_md = final_md.rstrip() + f'<div class="doc-references">Referenced from: {references_text}</div>'
                else:
                    final_md += f"\n\nReferenced from: {references_text}"
            final_md = render_html_document(final_md)

        # For QA mode: ensure markdown formatting and add references
        if mode == "qa" and final_md:
            # Build references list from sources
            refs = []
            seen_sources = set()
            for ch in final_chunks[:6]:
                src = ch.metadata.get("source") or ch.metadata.get("document") or "Document"
                p = ch.metadata.get("page") or ch.metadata.get("page_number")
                if src not in seen_sources:
                    if p:
                        refs.append(f"{src}, p. {p}")
                    else:
                        refs.append(src)
                    seen_sources.add(src)
            references_list = refs
            final_md = ensure_markdown(final_md, refs)
        
        final_text = final_md
        
        # Ensure final answer is not empty - if still empty after retry, use smarter fallback
        if not final_text or len(final_text) < 50:
            # Use the new generate_answer function for smarter fallback
            fallback_response = await generate_answer(resolved_query, recent_msgs)
            final_text = fallback_response
            if mode == "docgen":
                final_text = render_html_document(final_text)
            final_md = final_text
            references_list = []
            if mode == "docgen":
                detected_doc_type = None

        if mode == "docgen" and final_text:
            await update_last_document(conversation_id, final_text, detected_doc_type, references_list)
            state_cache.doc_fields = {}

        pages = []
        for ch in final_chunks:
            p = ch.metadata.get("page") or ch.metadata.get("page_number") or ch.metadata.get("page_index")
            if isinstance(p, (int, float)) and int(p) not in pages:
                pages.append(int(p))
        pages = sorted(pages)

        await record_response(conversation_id, final_text, mode=mode)
        await _update_summary_with_exchange(conversation_id, state_cache, resolved_query, final_text)

        # Persist assistant response to memory
        async with SessionLocal() as db:
            await self.memory.append(db, user_id, conversation_id, "assistant", final_text)
            await db.commit()
        
        # RAG-DEBUG: Enhanced final timing and stats summary
        t_total = time.time() - t_start
        if settings.DEBUG_RAG:
            logger.info(f"[RAG-DEBUG] COMPLETE: query='{query[:50]}...', t_total={t_total*1000:.0f}ms (embed={t_embed*1000:.0f}ms, llm_first={t_llm_first*1000:.0f}ms, retry={t_guardrail_retry*1000:.0f}ms), retrieval_stats=(hits={total_hits}, strong={strong_hits}), context_chars={total_context_chars}, num_chunks_sent={num_chunks_sent}, final_answer_len={len(final_text)} chars, conversation_id={conversation_id}")
        else:
            logger.info(f"[answer_query] COMPLETE: t_total={t_total*1000:.0f}ms (embed={t_embed*1000:.0f}ms, llm_first={t_llm_first*1000:.0f}ms, retry={t_guardrail_retry*1000:.0f}ms), num_chunks_sent={num_chunks_sent}, final_answer_len={len(final_text)} chars")

        return {
            "success": True,
            "answer": final_text,
            "answer_markdown": final_md,
            "metadata": {
                "strategy": strategy,
                "mode": mode,
                "model": target_model,
                "kb_hits": total_hits,
                "web_used": use_web,
                "referenced_pages": pages if pages else None,
                "references": references_list,
                "doc_type": detected_doc_type,
                "latency_ms": int(t_total * 1000),
                "num_chunks_sent": num_chunks_sent,
                "conversation_id": conversation_id,
                "response_type": "html" if mode == "docgen" else "text",
                "has_doc": mode == "docgen",
            }
        }


# Global orchestrator instance
_orchestrator: Optional[RAGOrchestrator] = None

def get_rag_orchestrator() -> RAGOrchestrator:
    """Get singleton RAG orchestrator."""
    global _orchestrator
    if _orchestrator is None:
        _orchestrator = RAGOrchestrator()
    return _orchestrator
