"""Main RAG orchestration service with modular architecture."""

from typing import Dict, Any, Optional, List, Tuple
from datetime import datetime
import logging
import re

from app.modules.lawfirmchatbot.services.ingestion.document_processor import process_document
from app.modules.lawfirmchatbot.services.retrieval.vector_search import add_documents_to_vector_store, search_similar_documents, normalize_hits, build_user_prompt
from app.modules.lawfirmchatbot.services.llm import chat_completion, looks_like_citations_only
from app.modules.lawfirmchatbot.schema.query import QueryResponse, DebugInfo, DebugQueryAnalysis
from app.modules.lawfirmchatbot.services.query_analyzer import QueryComplexityAnalyzer
from app.modules.lawfirmchatbot.services.reranker import DocumentReranker
from app.modules.lawfirmchatbot.services.adaptive_retrieval_service import AdaptiveRetrievalService
from app.core.answer_policy import classify_intent, select_strategy, format_markdown, grounding_guardrails
from core.config import settings

logger = logging.getLogger(__name__)

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
        "what functions do you have"
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
        """Enhanced system prompt - completely dynamic structure."""
        return """You are an expert research assistant providing comprehensive, detailed answers using provided context.

**YOUR MISSION: Provide DETAILED, USEFUL answers that thoroughly address the user's question.**

**CRITICAL RULES:**
1. When context is available, write a COMPREHENSIVE answer (8-15+ sentences for detailed topics)
2. Extract and explain ALL relevant information from the context - be thorough, not brief
3. Structure your response dynamically based on what the question asks and what the context contains
4. Use inline citations [1], [2] throughout your answer
5. End with **References:** listing sources

**RESPONSE STRUCTURE - COMPLETELY DYNAMIC:**
- Create headings using ## that FIT THE CONTENT (NOT fixed templates!)
- If discussing a concept: start with what it is, then elaborate on key aspects
- If comparing: create natural comparison structure
- If explaining steps/process: use numbered or logical flow
- Use ## for main headings and ### for subheadings based on actual content
- Choose heading text dynamically based on what you're explaining

**EXAMPLES OF DYNAMIC HEADINGS (use these patterns, not exact text):**
- "## Understanding [Concept]" or "## What is [Topic]"
- "## Key Characteristics" or "## Core Principles"  
- "## Types of [Thing]" or "## Main Categories"
- "## How [Process] Works" or "## Implementation Steps"
- "## Benefits and Advantages" or "## Challenges and Considerations"
- "## Practical Applications" or "## Real-World Examples"
- Use ### for subheadings when breaking down complex sections

**FOR COMPARISONS/DIFFERENCES:**
- When asked to compare or explain differences, create a comparison table:
- Use markdown table format: | Column 1 | Column 2 |
- Include header row with |---------|---------|
- Compare features, characteristics, or aspects side-by-side

**BE COMPREHENSIVE:**
- Write detailed paragraphs (4-6 sentences each) when explaining concepts
- Use bullet points (with -) to organize multiple items/characteristics
- Include examples, applications, implications from the context
- Explain WHY things matter, not just WHAT they are
- Connect ideas and show relationships between concepts
- Format with proper markdown: ## for headings, **bold** for emphasis, - for bullets

**FOR PARTIAL COVERAGE:**
- Thoroughly explain what IS in the context
- Note specific gaps: "The documents don't cover X, but provide extensive detail on Y"
- Never refuse to answer if any relevant information exists

**ANTI-HALLUCINATION:**
- Use ONLY information from context - but use ALL of it
- Don't invent facts, but DO synthesize and explain what's provided
- If context has 5 key points, discuss all 5 in detail

**MANDATORY:** Write detailed, comprehensive answers. If context provides 500 words of information, your answer should reflect that depth. Be thorough!

**FORMATTING RULES:**
- Main headings: ## Heading Text
- Subheadings: ### Subheading Text
- Emphasis: **bold text**
- Lists: use - for bullet points
- References: **References:** at the end

Format: Clean markdown with ## headings that adapt to content."""

    def build_user_prompt(self, query: str, chunks: list[dict], sources: list[str], pages: list[int], require_explanation: bool = True, extra_instruction: str = "") -> str:
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
        
        user = f"""Answer this question using ONLY the context below. Extract and explain ANY relevant information you find.

CONTEXT:
{context}

QUESTION: {query}

INSTRUCTIONS:
- Start with a ## heading that fits the question type (Definition/Overview/Key Points/etc.)
- Write 3-6 sentences explaining what you found in the context - this is MANDATORY
- If the context mentions the topic at all, explain what it says
- Add bullet points for key details if applicable
- Include inline citations [1], [2] next to claims
- End with **References**: [1], [2], etc. listing the sources you used
{f"- Reference pages: {', '.join(map(str, pages))}" if pages else ""}
{extra_instruction}

CRITICAL: You MUST provide a substantive answer using the context above. Extract and explain ANY relevant information - don't refuse if information exists."""
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
        import time
        from core.config import settings  # RAG-DEBUG: Import settings for debug logging
        t_start = time.time()
        t_embed = 0.0
        t_qdrant_search = 0.0
        t_llm_first = 0.0
        t_guardrail_retry = 0.0
        
        # RAG-DEBUG: Comprehensive query logging
        if settings.DEBUG_RAG:
            logger.info(f"[RAG-DEBUG] Starting answer_query: query='{query[:150]}...' (length={len(query)})")
        
        # Chit-chat short-circuit: handle greetings/meta queries instantly
        if is_smalltalk_or_capability(query):
            if settings.DEBUG_RAG:
                logger.info(f"[RAG-DEBUG] Fast-path: chit-chat detected, skipping retrieval")
            else:
                logger.info(f"[answer_query] Fast-path: chit-chat detected, skipping retrieval")
            smalltalk_answer = """## How I Can Help
- Answer questions using your uploaded documents with precise citations
- Summarize, compare, extract key points, explain concepts, draft analysis
- Provide structured responses with relevant page references

**References**: N/A"""
            return {
                "success": True,
                "answer": smalltalk_answer,
                "answer_markdown": smalltalk_answer,
                "metadata": {"strategy": "chit_chat", "kb_hits": 0, "web_used": False, "latency_ms": int((time.time() - t_start) * 1000)}
            }
        
        # Retrieve more candidates for strategy decision
        k = 6
        t_embed_start = time.time()
        # search_similar_documents now returns a tuple: (documents, unique_sources, unique_pages)
        search_result = await search_similar_documents(query, k=k)
        t_qdrant_search = time.time() - t_embed_start
        t_embed = t_qdrant_search  # Includes embedding time
        
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

        logger.info(f"[answer_query] Retrieval: total_hits={total_hits}, strong_hits={strong_hits}, max_score={max_score:.3f}, t_qdrant={t_qdrant_search*1000:.0f}ms")

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

        # RAG-DEBUG: Enhanced context trimming for optimal token management
        final_chunks = initial_chunks[:6]
        total_context_chars = 0
        MAX_TOTAL_CONTEXT_CHARS = 4000  # RAG-DEBUG: Keep total context under ~1000 tokens
        MAX_CHUNK_CHARS = 600  # RAG-DEBUG: Optimized for speed and focus
        
        for i, ch in enumerate(final_chunks, start=1):
            src = ch.metadata.get("source") or ch.metadata.get("document") or "Source"
            _page = ch.metadata.get("page") or ch.metadata.get("page_number") or ch.metadata.get("page_index")
            header = f"[{i}] {src}" + (f", page {int(_page)}" if isinstance(_page, (int, float)) else "")
            
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
        
        # RAG-DEBUG: Context validation before LLM call
        if settings.DEBUG_RAG:
            logger.info(f"[RAG-DEBUG] Context validation: num_chunks={len(numbered_context)}, total_chars={total_context_chars}")
            for i, ctx in enumerate(numbered_context[:3], 1):  # Log first 3 contexts
                snippet = ctx[:200] + "..." if len(ctx) > 200 else ctx
                logger.info(f"[RAG-DEBUG] Context {i}: {snippet}")
        
        # Validate minimum context threshold (600 chars minimum for meaningful answers)
        MIN_CONTEXT_CHARS = 600
        if total_context_chars < MIN_CONTEXT_CHARS:
            logger.warning(f"[RAG-DEBUG] Context too small: {total_context_chars} chars < {MIN_CONTEXT_CHARS} threshold")
            if settings.DEBUG_RAG:
                logger.info(f"[RAG-DEBUG] Available context snippets: {[ctx[:100] + '...' for ctx in numbered_context]}")
            
            # Return thin-context fallback instead of triggering citations-only retry
            fallback_md = """## Limited Information Available

Based on the available context, I can provide a brief overview, though the information is limited.

- The retrieved documents contain some relevant information but may not cover all aspects of your question.
- For a more detailed answer, please try refining your query or asking about specific aspects you're interested in.

**References**: See available documents"""
            return {
                "success": True,
                "answer": fallback_md,
                "answer_markdown": fallback_md,
                "metadata": {"strategy": "thin_context", "kb_hits": total_hits, "web_used": use_web, "context_chars": total_context_chars, "latency_ms": int((time.time() - t_start) * 1000)}
            }

        if use_web:
            base = len(numbered_context)
            for j, w in enumerate(web_results, start=1):
                header = f"[{base + j}] {w['title']} — {w['url']}"
                numbered_context.append(f"{header}\n{w['snippet']}")
                source_list.append((w["url"], w["title"]))

        # Only return "no info" if truly zero snippets
        if not numbered_context:
            logger.warning(f"[answer_query] Zero snippets available, returning fallback")
            fallback_md = """## Limited Information

I couldn't find specific information about this topic in the uploaded documents.

- Try rephrasing your question or asking about a specific aspect
- Check if relevant documents have been uploaded

**References**: None found"""
            return {
                "success": True,
                "answer": fallback_md,
                "answer_markdown": fallback_md,
                "metadata": {"strategy": "no_snippets", "kb_hits": total_hits, "web_used": use_web, "latency_ms": int((time.time() - t_start) * 1000)}
            }
        
        # Log context stats before LLM call
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
        
        user_prompt = f"""Using the context below, provide a DETAILED, COMPREHENSIVE answer to the question.

CONTEXT:
{chr(10).join(numbered_context)}

QUESTION: {query}

INSTRUCTIONS:
- Write a THOROUGH answer using ALL relevant information from the context (aim for 8-15+ sentences for complex topics)
- Create dynamic ## headings that match the actual content (NOT fixed templates like "Definition" or "Overview")
- Structure your response naturally based on what the question asks
- Use multiple paragraphs with 4-6 sentences each to explain concepts thoroughly
- Add bullet points (with -) to organize lists of characteristics, types, or examples
- Include inline citations [1], [2] throughout
- Explain WHY things matter, include applications, implications, and examples from context
- End with **References:** [1], [2], etc.

FORMATTING REQUIREMENTS:
- Main headings: ## Dynamic Heading Based on Content
- Subheadings if needed: ### Subheading Text
- Emphasis: **bold text** for important terms
- Lists: - bullet point format
- Tables: | Column | Column | format for comparisons
- Citations: [1], [2] inline

EXAMPLES OF NATURAL STRUCTURE:
For "What is X?": ## Understanding [X] → explain thoroughly → ## Key Characteristics → ## Applications

For "Difference between X and Y?": 
## Comparing [X] and [Y]
[Brief intro paragraph]
| Aspect | [X] | [Y] |
|--------|-----|-----|
| Feature 1 | Details | Details |
| Feature 2 | Details | Details |
[Additional explanation paragraphs]

For "How does X work?": ## How [X] Works → explain process with steps/stages

CRITICAL: 
- Use ## markdown headings (not bold text)
- For comparisons/differences, ALWAYS include a comparison table
- Use the context exhaustively - be comprehensive!"""

        messages = [
            {"role": "system", "content": self._get_system_prompt()},
            {"role": "user", "content": user_prompt},
        ]

        messages = fit_context(messages)
        
        # RAG-DEBUG: Log full prompt being sent to LLM
        if settings.DEBUG_RAG:
            total_prompt_chars = sum(len(m.get('content', '')) for m in messages)
            logger.info(f"[RAG-DEBUG] Sending to LLM: {len(messages)} messages, total_chars={total_prompt_chars}, est_tokens={total_prompt_chars//4}")
            logger.info(f"[RAG-DEBUG] System prompt preview: {messages[0]['content'][:200]}...")
            logger.info(f"[RAG-DEBUG] User prompt preview: {messages[1]['content'][:500]}...")
        
        t_llm_start = time.time()
        # Increased to 2000 tokens for detailed, comprehensive responses
        raw_markdown = await chat_completion(messages, is_legal_query=True, max_tokens=2000, temperature=0.3)
        t_llm_first = time.time() - t_llm_start
        
        if settings.DEBUG_RAG:
            estimated_tokens = len(raw_markdown) // 4
            logger.info(f"[RAG-DEBUG] LLM first call: t_llm={t_llm_first*1000:.0f}ms, response_len={len(raw_markdown)} chars, est_tokens={estimated_tokens}")
            if len(raw_markdown) > 0:
                logger.info(f"[RAG-DEBUG] LLM response preview: {raw_markdown[:300]}...")
            else:
                logger.error(f"[RAG-DEBUG] LLM returned EMPTY response! This is the core issue.")
        else:
            logger.info(f"[answer_query] LLM first call: t_llm={t_llm_first*1000:.0f}ms, response_len={len(raw_markdown)} chars")

        # RAG-DEBUG: Improved guardrail logic to reduce unnecessary retries
        if looks_like_citations_only(raw_markdown):
            # Only retry if we have sufficient context (≥600 chars) AND response is truly empty
            should_retry = total_context_chars >= MIN_CONTEXT_CHARS and len(raw_markdown.strip()) < 50
            
            if not should_retry:
                if settings.DEBUG_RAG:
                    logger.info(f"[RAG-DEBUG] Skipping retry: insufficient context ({total_context_chars} chars) or response not empty")
                # Use the thin-context fallback instead of retrying
                if total_context_chars < MIN_CONTEXT_CHARS:
                    raw_markdown = """## Limited Information Available

Based on the available context, I can provide a brief overview, though the information is limited.

- The retrieved documents contain some relevant information but may not cover all aspects of your question.
- For a more detailed answer, please try refining your query or asking about specific aspects you're interested in.

**References**: See available documents"""
            else:
                logger.warning("[answer_query] Guardrail triggered: citations-only with sufficient context, retrying once")
                t_retry_start = time.time()
                
                # More direct and stronger retry instruction
                retry_prompt = """Your previous response was flagged as citations-only. Please provide a complete answer:

REQUIRED FORMAT:
1. Opening paragraph: 2-3 sentences explaining the answer in plain language
2. Key points: Bullet list of important details from the context
3. Inline citations: Add [1], [2] markers next to claims
4. References line: **References**: [1] Source - Page X

DO NOT return only references. You MUST include explanatory text based on the context provided above."""
                
                retry_messages = [
                    {"role": "system", "content": self._get_system_prompt()},
                    {"role": "user", "content": user_prompt},
                    {"role": "assistant", "content": raw_markdown},
                    {"role": "user", "content": retry_prompt},
                ]
                retry_messages = fit_context(retry_messages)
                raw_markdown = await chat_completion(retry_messages, is_legal_query=True, max_tokens=2000, temperature=0.2)
                t_guardrail_retry = time.time() - t_retry_start
                
                if settings.DEBUG_RAG:
                    estimated_retry_tokens = len(raw_markdown) // 4
                    logger.info(f"[RAG-DEBUG] Guardrail retry: t_retry={t_guardrail_retry*1000:.0f}ms, new_response_len={len(raw_markdown)} chars, est_tokens={estimated_retry_tokens}")
                else:
                    logger.info(f"[answer_query] Guardrail retry: t_retry={t_guardrail_retry*1000:.0f}ms, new_response_len={len(raw_markdown)} chars")

        # Use raw_markdown directly - keep markdown formatting for frontend
        final_md = raw_markdown.strip()
        # Don't strip markdown - frontend needs it for proper rendering
        final_text = final_md  # Keep markdown intact
        
        # Ensure final answer is not empty - if still empty after retry, use thin-context fallback
        if not final_text or len(final_text) < 50:
            fallback_md = """## Overview
Based on the available context, I can provide a brief overview, though the information is limited.

- The retrieved documents contain some relevant information but may not cover all aspects of your question.
- For a more detailed answer, please try refining your query or specifying particular aspects you're interested in.

**References**: See reference pages above"""
            final_text = fallback_md
            final_md = fallback_md

        pages = []
        for ch in final_chunks:
            p = ch.metadata.get("page") or ch.metadata.get("page_number") or ch.metadata.get("page_index")
            if isinstance(p, (int, float)) and int(p) not in pages:
                pages.append(int(p))
        pages = sorted(pages)

        # RAG-DEBUG: Enhanced final timing and stats summary
        t_total = time.time() - t_start
        if settings.DEBUG_RAG:
            logger.info(f"[RAG-DEBUG] COMPLETE: query='{query[:50]}...', t_total={t_total*1000:.0f}ms (embed={t_embed*1000:.0f}ms, llm_first={t_llm_first*1000:.0f}ms, retry={t_guardrail_retry*1000:.0f}ms), retrieval_stats=(hits={total_hits}, strong={strong_hits}), context_chars={total_context_chars}, num_chunks_sent={num_chunks_sent}, final_answer_len={len(final_text)} chars")
        else:
            logger.info(f"[answer_query] COMPLETE: t_total={t_total*1000:.0f}ms (embed={t_embed*1000:.0f}ms, llm_first={t_llm_first*1000:.0f}ms, retry={t_guardrail_retry*1000:.0f}ms), num_chunks_sent={num_chunks_sent}, final_answer_len={len(final_text)} chars")

        return {
            "success": True,
            "answer": final_text,
            "answer_markdown": final_md,
            "metadata": {
                "strategy": strategy,
                "kb_hits": total_hits,
                "web_used": use_web,
                "referenced_pages": pages if pages else None,
                "latency_ms": int(t_total * 1000),
                "num_chunks_sent": num_chunks_sent,
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
