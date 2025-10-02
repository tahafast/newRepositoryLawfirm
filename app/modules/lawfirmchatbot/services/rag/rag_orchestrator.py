"""Main RAG orchestration service with modular architecture."""

from typing import Dict, Any, Optional, List, Tuple
from datetime import datetime
import logging
import re

from app.modules.lawfirmchatbot.services.ingestion.document_processor import process_document
from app.modules.lawfirmchatbot.services.retrieval.vector_search import add_documents_to_vector_store, search_similar_documents, normalize_hits, build_user_prompt
from app.modules.lawfirmchatbot.services.llm import chat_completion
from app.modules.lawfirmchatbot.schema.query import QueryResponse, DebugInfo, DebugQueryAnalysis
from app.modules.lawfirmchatbot.services.query_analyzer import QueryComplexityAnalyzer
from app.modules.lawfirmchatbot.services.reranker import DocumentReranker
from app.modules.lawfirmchatbot.services.adaptive_retrieval_service import AdaptiveRetrievalService
from app.core.answer_policy import classify_intent, select_strategy, format_markdown, grounding_guardrails
from core.config import settings

logger = logging.getLogger(__name__)


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
        """Get answer for a query using enhanced RAG with assess→retrieve flow."""
        try:
            # Check if there are any documents in Qdrant (use Qdrant as source of truth)
            from app.modules.lawfirmchatbot.services.vector_store import points_count, get_qdrant_client, search_similar
            from app.services.LLM.config import route_intent, reply_small_talk, reply_clarify, reply_not_found, answer_with_context
            from app.modules.lawfirmchatbot.services.llm import _get_client
            import os
            
            client = get_qdrant_client()
            if points_count(client) <= 0:
                return QueryResponse(
                    success=False,
                    answer="⚠️ No document has been uploaded yet. Please upload a document first.",
                    answer_markdown="⚠️ No document has been uploaded yet. Please upload a document first.",
                    metadata={"error": "no_document_uploaded"},
                    debug_info=None
                )
            
            # 1) Intent routing (no retrieval yet)
            openai_client = _get_client()
            router = route_intent(openai_client, query)
            
            if router["intent"] == "chit_chat":
                return QueryResponse(
                    success=True,
                    answer=reply_small_talk(),
                    answer_markdown=reply_small_talk(),
                    metadata={"mode":"chit_chat"}
                )
            
            if router["intent"] == "clarify_needed":
                return QueryResponse(
                    success=True,
                    answer=reply_clarify(),
                    answer_markdown=reply_clarify(),
                    metadata={"mode":"clarify"}
                )
            
            # 2) Build retrieval query
            expanded = router.get("normalized_query") or query
            
            # 3) Vector search (MMR, small top-k, low timeout)
            from app.modules.lawfirmchatbot.services.embeddings import embed_text
            query_embedding = await embed_text(expanded)
            
            top_k = int(os.getenv("RAG_TOP_K", "6"))
            hits = search_similar(
                client=client,
                query_vector=query_embedding,
                top_k=top_k,
                filter_=None,
                mmr=True  # Enable MMR for diversity
            )
            
            # 4) Confidence gate (avoid hallucination)
            min_score = float(os.getenv("RAG_MIN_SCORE", "0.18"))
            if not hits or (getattr(hits[0], "score", 0.0) < min_score):
                return QueryResponse(
                    success=True,
                    answer=reply_not_found(router.get("needs_web", False)),
                    answer_markdown=reply_not_found(router.get("needs_web", False)),
                    metadata={"mode":"no_context","hits":len(hits) if hits else 0}
                )
            
            # 5) Plan & answer with dynamic headings
            final_text = answer_with_context(openai_client, query, hits)
            
            # Extract metadata from hits
            pages = []
            sources = []
            for h in hits:
                payload = getattr(h, "payload", None) or {}
                metadata = payload.get("metadata", {})
                
                # Extract page number
                for key in ("page", "page_number", "pageIndex", "page_index"):
                    p = metadata.get(key) or payload.get(key)
                    if isinstance(p, (int, float)) and int(p) not in pages:
                        pages.append(int(p))
                        break
                
                # Extract source
                src = metadata.get("source") or payload.get("source") or payload.get("document") or "Source"
                if src not in sources:
                    sources.append(src)
            
            pages = sorted(pages)[:12]
            
            return QueryResponse(
                success=True,
                answer=final_text,
                answer_markdown=final_text,
                metadata={
                    "hits": len(hits),
                    "query": expanded,
                    "mode": "rag",
                    "top_score": getattr(hits[0], "score", None),
                    "sources": sources,
                    "referenced_pages": pages,
                    "page_references": f"Pages: {', '.join(map(str, pages))}" if pages else "No page references available",
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

    def _get_system_prompt(self) -> str:
        """Get system prompt for LLM."""
        return (
            "You are a legal RAG assistant. Use only the supplied KB chunks and/or web snippets.\n"
            "- Never invent citations or facts. If a requested comparison/topic isn’t found in the supplied context, say so and (only if allowed) use web snippets you were given.\n"
            "- Prefer precise, lawyer-friendly wording; keep it concise but substantive.\n"
            "- Output Markdown with bold H3 section headings; tailor section names to the query intent.\n"
            "- Use bracketed numeric citations like [1], [2] inline, and repeat them in a \"Citations\" section.\n"
            "- If the user asks for a summary, include 3–6 bullet key points.\n"
            "- If the user asks to compare, present a short table then bullets.\n"
            "- If context is insufficient and web is disabled/unavailable, ask a clarifying question instead of guessing.\n"
        )

    def _get_few_shots(self) -> List[Dict[str, str]]:
        """Get few-shot examples for the LLM."""
        return [
            {
                "role": "user",
                "content": "key man summary"
            },
            {
                "role": "assistant", 
                "content": """## Summary
Key Man insurance provides financial protection for businesses when a critical employee becomes unavailable due to death or disability [1].

## Key Points
- Protects against loss of key personnel [1]
- Covers death and disability scenarios [1]
- Business pays premiums and receives benefits [1]

## Supporting Evidence
- "Key Man insurance protects businesses from financial loss when critical employees become unavailable" [1]"""
            },
            {
                "role": "user",
                "content": "define chain of title in one paragraph"
            },
            {
                "role": "assistant",
                "content": """## Definition
Chain of title refers to the complete sequence of ownership transfers for a property, tracing from the original owner to the current holder, establishing legal ownership history [1].

## Notes
- Essential for property transactions [1]
- Must be unbroken to ensure clear title [1]"""
            },
            {
                "role": "user", 
                "content": "compare tort vs contract liability"
            },
            {
                "role": "assistant",
                "content": """## Comparison
- Tort liability arises from wrongful acts causing harm, while contract liability stems from breach of agreement [1]
- Tort damages are compensatory, contract damages are expectation-based [2]

## Evidence
- "Tort liability focuses on harm caused by wrongful conduct" [1]
- "Contract liability enforces promises and expectations" [2]"""
            }
        ]

    def _create_user_prompt(self, query: str, chunks) -> str:
        """Create user prompt with chunks."""
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

RELEVANT CHUNKS:
{chunk_text}

USER QUERY: {query}

INSTRUCTIONS:
- Answer using ONLY the information from the chunks above
- Use the exact response format from the system prompt
- Replace {{document_name}} with: {self.current_document['name']}
- Replace {{query_type}} with the type of query this is
- Replace {{page_numbers}} with: {page_ref_text}
- Include specific page references in your supporting evidence
- Be precise and cite the exact pages where you found information"""

    async def answer_query(self, query: str) -> Dict[str, Any]:
        """High-level entry: route between KB-only, KB+Web, or Clarify.

        Does not change router signatures; can be called by existing handler.
        """
        # Retrieve more candidates for strategy decision
        k = 8
        initial_chunks = await search_similar_documents(query, k=k)
        total_hits = len(initial_chunks)
        scores = [float(getattr(ch, "score", 0.0) or ch.metadata.get("score", 0.0)) for ch in initial_chunks]
        strong_hits = sum(1 for s in scores if s >= 0.25)
        max_score = max(scores) if scores else 0.0

        retrieval_stats = {"total_hits": total_hits, "strong_hits": strong_hits, "max_score": max_score}
        intent = classify_intent(query)
        strategy = select_strategy(query, retrieval_stats)

        use_web = False
        web_results: List[Dict[str, Any]] = []
        web_summary = ""

        # Optionally call web search
        if strategy == "kb_plus_web" and (settings.WEB_SEARCH_ENABLED is True or (settings.WEB_SEARCH_ENABLED is None and bool(settings.TAVILY_API_KEY))):
            use_web = True
            try:
                from app.services.web_search.tavily_client import search_and_summarize as tavily_search
                web_sources, web_summary = await tavily_search(query, top_k=3)
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

        final_chunks = initial_chunks[:6]
        for i, ch in enumerate(final_chunks, start=1):
            src = ch.metadata.get("source") or ch.metadata.get("document") or "Source"
            _page = ch.metadata.get("page") or ch.metadata.get("page_number") or ch.metadata.get("page_index")
            header = f"[{i}] {src}" + (f", page {int(_page)}" if isinstance(_page, (int, float)) else "")
            numbered_context.append(f"{header}\n{ch.page_content}")
            key = f"{src}|{_page}" if _page is not None else src
            source_list.append((key, str(src)))

        if use_web:
            base = len(numbered_context)
            for j, w in enumerate(web_results, start=1):
                header = f"[{base + j}] {w['title']} — {w['url']}"
                numbered_context.append(f"{header}\n{w['snippet']}")
                source_list.append((w["url"], w["title"]))

        if not numbered_context:
            # No KB and web disabled/unavailable → clarify
            return {
                "success": True,
                "answer": "Could you clarify what specifically you want to know? For example: define, compare, or summarize a particular item.",
                "answer_markdown": "### **Clarify**\nCould you clarify what specifically you want to know? For example: define, compare, or summarize a particular item.",
                "metadata": {"strategy": "clarify", "kb_hits": total_hits, "web_used": use_web}
            }

        # Compose messages
        user_prompt = (
            f"Question: {query}\n\nCONTEXT:\n" + "\n".join(numbered_context) +
            "\n\nFormat: Markdown only. Use bold H3 section headings suitable to the question. If a claim cannot be tied to a snippet, omit it."
        )
        messages = [
            {"role": "system", "content": self._get_system_prompt()},
            {"role": "user", "content": user_prompt},
        ]

        messages = fit_context(messages)
        raw_markdown = await chat_completion(messages, is_legal_query=True)

        # Build sections for final formatting shell (light-touch)
        sections = [
            {"title": "Answer", "paragraphs": [raw_markdown], "sources": [{"key": k, "title": t} for k, t in source_list]},
        ]
        if use_web:
            sections.append({"title": "Notes", "bullets": [grounding_guardrails(True)]})

        final_md = format_markdown(sections)
        final_text = strip_markdown(final_md)

        pages = []
        for ch in final_chunks:
            p = ch.metadata.get("page") or ch.metadata.get("page_number") or ch.metadata.get("page_index")
            if isinstance(p, (int, float)) and int(p) not in pages:
                pages.append(int(p))
        pages = sorted(pages)

        return {
            "success": True,
            "answer": final_text,
            "answer_markdown": final_md,
            "metadata": {
                "strategy": strategy,
                "kb_hits": total_hits,
                "web_used": use_web,
                "referenced_pages": pages if pages else None,
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
