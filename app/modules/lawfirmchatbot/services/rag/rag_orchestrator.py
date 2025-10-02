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
        """Get answer for a query using optimized RAG with fast intent gate."""
        try:
            # Meta-query short-circuit for greetings and help
            META_QS = {"hi", "hello", "hey", "what can you do", "who are you", "help"}
            q_norm = " ".join(query.lower().split())
            if any(kw in q_norm for kw in META_QS):
                meta_answer = (
                    "I can answer questions about your ingested documents (law, HR, entrepreneurship, CV, OS, etc.). "
                    "Ask in your own words; I'll find relevant passages, synthesize a clear answer, and provide citations."
                )
                return QueryResponse(
                    success=True,
                    answer=meta_answer,
                    answer_markdown=meta_answer,
                    metadata={"mode": "meta_query"}
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
                answer = ("I couldn't find relevant information in your documents for this query. "
                         "Could you clarify what specific legal concept or document section you're looking for?")
                return QueryResponse(
                    success=True,
                    answer=answer,
                    answer_markdown=f"# Answer\n{answer}",
                    metadata={"mode": "no_context", "hits": 0}
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
            answer = await chat_completion(messages, is_legal_query=True)
            
            # Guard against citations-only responses
            if self._looks_like_citations_only(answer):
                logger.warning("Detected citations-only response, retrying with guardrail")
                retry_prompt = (
                    "Your previous output contained only citations.\n"
                    "Now produce a **concise answer (3–6 sentences)** first, then a single 'Citations: [..]' line.\n"
                    "Do not apologize. Do not repeat the question. Do not return only citations.\n\n"
                    + user_prompt
                )
                retry_messages = [
                    {"role": "system", "content": self._get_system_prompt()},
                    {"role": "user", "content": retry_prompt}
                ]
                answer = await chat_completion(retry_messages, is_legal_query=True)
            
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
        """Get system prompt for LLM."""
        return """You are a careful RAG assistant for a law-firm knowledge base.
- Answer the user's question **directly first** in clear prose (2–6 short paragraphs or a tight list).
- Use only the provided CONTEXT unless `allow_web=True` is explicitly set (we currently do not do web).
- If the context is insufficient, say what is missing and ask a concise follow-up question.
- After the answer, add a short **Citations** line in the format: [1], [2] …
- Do NOT output only citations. Always include an answer section.
- Keep headings short and descriptive. Avoid boilerplate like "Answer:" unless answer would otherwise be ambiguous.
- No hallucinations: if something isn't in the context, say so briefly."""

    def build_user_prompt(self, query: str, chunks: list[dict], sources: list[str], pages: list[int]) -> str:
        """Build user prompt with structured context."""
        ctx_lines = []
        for i, c in enumerate(chunks, 1):
            title = (c.get("document") or c.get("source") or "Document").strip()
            page = c.get("page") or c.get("page_number") or "N/A"
            text = (c.get("text") or "").strip()
            if not text:
                continue
            ctx_lines.append(f"[{i}] {title} (page {page})\n{text}")

        context = "\n\n".join(ctx_lines) if ctx_lines else "NO_RELEVANT_CONTEXT"
        
        user = f"""QUESTION:
{query}

CONTEXT (top {len(chunks)} chunks):
{context}

Write the answer **first**, then one of these optional sections when appropriate:
- Key Points (bulleted)
- Comparison (if the user asks to compare)
- Steps / Procedure (if procedural)
- Risks / Caveats (legal cautions etc.)

Finally add a single line:
Citations: [{{comma-separated citation indices you actually used}}]

Rules:
- Never output only the "Citations" line.
- Cite only chunks you actually used.
- If context is weak, say so and ask a follow-up."""
        return user

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
        # search_similar_documents now returns a tuple: (documents, unique_sources, unique_pages)
        search_result = await search_similar_documents(query, k=k)
        
        # Unpack the tuple
        if isinstance(search_result, tuple):
            initial_chunks, _, _ = search_result
        else:
            # Fallback for backwards compatibility
            initial_chunks = search_result
        
        total_hits = len(initial_chunks)
        scores = [float(ch.metadata.get("similarity_score", 0.0)) for ch in initial_chunks]
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

        # Guard against citations-only responses
        if self._looks_like_citations_only(raw_markdown):
            logger.warning("Detected citations-only response in answer_query, retrying with guardrail")
            retry_prompt = (
                "Your previous output contained only citations.\n"
                "Now produce a **concise answer (3–6 sentences)** first, then a single 'Citations: [..]' line.\n"
                "Do not apologize. Do not repeat the question. Do not return only citations.\n\n"
                + user_prompt
            )
            retry_messages = [
                {"role": "system", "content": self._get_system_prompt()},
                {"role": "user", "content": retry_prompt},
            ]
            retry_messages = fit_context(retry_messages)
            raw_markdown = await chat_completion(retry_messages, is_legal_query=True)

        # Use raw_markdown directly instead of wrapping with format_markdown
        final_md = raw_markdown.strip()
        final_text = strip_markdown(final_md)
        
        # Ensure final answer is not empty
        if not final_text:
            final_text = "I couldn't find enough information in the provided documents to answer confidently."
            final_md = final_text

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
