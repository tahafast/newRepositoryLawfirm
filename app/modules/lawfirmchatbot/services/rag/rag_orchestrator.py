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
        """Get answer for a query using enhanced RAG."""
        try:
            # Check if there are any documents in Qdrant (use Qdrant as source of truth)
            from app.modules.lawfirmchatbot.services.vector_store import points_count, get_qdrant_client
            
            client = get_qdrant_client()
            if points_count(client) <= 0:
                return QueryResponse(
                    success=False,
                    answer="⚠️ No document has been uploaded yet. Please upload a document first.",
                    answer_markdown="⚠️ No document has been uploaded yet. Please upload a document first.",
                    metadata={"error": "no_document_uploaded"},
                    debug_info=None
                )
            
            # Analyze query
            query_analysis = self.query_analyzer.analyze(query)
            logger.info(f"Query analysis - Type: {query_analysis['query_type']}, "
                       f"Complexity: {query_analysis['complexity_score']}")
            
            # Retrieve relevant chunks
            k = query_analysis['suggested_k']
            initial_chunks = await search_similar_documents(query, k=k)
            
            if not initial_chunks:
                return QueryResponse(
                    success=False,
                    answer="I searched your knowledge base but couldn't find a relevant passage for that query.",
                    answer_markdown="I searched your knowledge base but couldn't find a relevant passage for that query.",
                    metadata={"error": "no_relevant_context", "hits": 0},
                    debug_info=None
                )
            
            # Re-rank chunks
            reranked_chunks = await self.reranker.rerank_chunks(
                query, initial_chunks, query_analysis
            )
            
            # Limit final chunks to keep cost down
            top_k = min(4, len(reranked_chunks))  # Cap at 4 chunks to control cost
            final_chunks = reranked_chunks[:top_k]
            
            # Calculate confidence
            confidence_analysis = self.retrieval_service.calculate_advanced_confidence(
                final_chunks, query_analysis
            )
            
            # Collect page numbers from retrieved chunks
            pages = []
            for ch in final_chunks:
                p = None
                # Check common metadata keys for page numbers
                for key in ("page", "page_number", "pageIndex", "page_index"):
                    if key in ch.metadata:
                        p = ch.metadata[key]
                        break
                if isinstance(p, (int, float)) and p not in pages:
                    pages.append(int(p))
            # Keep a short, sorted page list
            pages = sorted(pages)[:12]
            
            # Build numbered context for citations
            numbered_context = []
            for i, ch in enumerate(final_chunks, start=1):
                src = ch.metadata.get("source") or ch.metadata.get("document") or "Source"
                _page = ch.metadata.get("page") or ch.metadata.get("page_number") or ch.metadata.get("page_index")
                header = f"[{i}] {src}" + (f", page {int(_page)}" if isinstance(_page, (int, float)) else "")
                numbered_context.append(f"{header}\n{ch.page_content}")
            
            if not numbered_context:
                logger.warning("no usable contexts from chunks (chunks=%d)", len(final_chunks))
                return QueryResponse(
                    success=False,
                    answer="I searched your knowledge base but couldn't find a relevant passage for that query.",
                    answer_markdown="I searched your knowledge base but couldn't find a relevant passage for that query.",
                    metadata={"error": "no_relevant_context", "hits": len(final_chunks)},
                    debug_info=None
                )
            
            # Build user prompt with numbered context
            user_prompt = f"""Question: {query}

CONTEXT:
{chr(10).join(numbered_context)}

Format: Markdown only. Use dynamic section headings suitable to the question."""
            
            # Create messages for LLM with system prompt, few-shots, and user query
            messages = [
                {"role": "system", "content": self._get_system_prompt()}
            ]
            
            # Add few-shot examples
            messages.extend(self._get_few_shots())
            
            # Add the actual user query with context
            messages.append({"role": "user", "content": user_prompt})
            
            # Apply context safety
            messages = fit_context(messages)
            
            # Get LLM response with simple timing and error guard
            start_ts = datetime.now()
            logger.info("LLM: sending %d messages (k=%d)", len(messages), len(final_chunks))
            answer = await chat_completion(
                messages,
                is_legal_query=query_analysis['is_legal_query']
            )
            elapsed_ms = int((datetime.now() - start_ts).total_seconds() * 1000)
            logger.info("LLM: received response in %d ms (%d chars)", elapsed_ms, len(answer or ""))
            
            # Extract sources for backward compatibility
            sources = list({
                (ch.metadata.get("source") or ch.metadata.get("document") or "Source")
                for ch in final_chunks
            })
            
            # Compute document name from sources (for backward compatibility)
            document_name = sources[0] if sources else "multiple documents"
            
            # Create both markdown and plain text versions
            answer_markdown = answer
            answer_plain = strip_markdown(answer)
            
            return QueryResponse(
                success=True,
                answer=answer_plain,
                answer_markdown=answer_markdown,
                metadata={
                    "document": document_name,
                    "sources": sources,
                    "chunks_retrieved": k,
                    "chunks_used": len(final_chunks),
                    "referenced_pages": pages,  # Use the collected pages
                    "page_references": f"Pages: {', '.join(map(str, pages))}" if pages else "No page references available",
                    "query_complexity": query_analysis['complexity_score'],
                    "confidence": confidence_analysis['confidence'],
                    "confidence_score": confidence_analysis['score'],
                    "confidence_factors": confidence_analysis['factors'],
                    "confidence_rationale": confidence_analysis['rationale'],
                    "processing_strategy": "enhanced_rag"
                },
                debug_info=DebugInfo(
                    query_analysis=DebugQueryAnalysis(**query_analysis),
                    retrieval_k=k,
                    reranking_applied=True,
                    total_pages=len(pages) or 1  # fallback for compatibility
                )
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
