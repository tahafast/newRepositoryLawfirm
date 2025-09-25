"""Main RAG orchestration service with modular architecture."""

from typing import Dict, Any, Optional
from datetime import datetime
import logging

from app.modules.lawfirmchatbot.services.ingestion.document_processor import process_document
from app.modules.lawfirmchatbot.services.retrieval.vector_search import add_documents_to_vector_store, search_similar_documents, normalize_hits, build_user_prompt
from app.modules.lawfirmchatbot.services.llm import chat_completion
from app.modules.lawfirmchatbot.schema.query import QueryResponse, DebugInfo, DebugQueryAnalysis
from app.modules.lawfirmchatbot.services.query_analyzer import QueryComplexityAnalyzer
from app.modules.lawfirmchatbot.services.reranker import DocumentReranker
from app.modules.lawfirmchatbot.services.adaptive_retrieval_service import AdaptiveRetrievalService

logger = logging.getLogger(__name__)


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
                    metadata={"error": "no_relevant_context", "hits": len(final_chunks)},
                    debug_info=None
                )
            
            # Build user prompt with numbered snippets for citations
            user_prompt = f"""Question: {query}

Context (numbered):
{chr(10).join(numbered_context)}

Instructions:
- Use only the numbered context.
- Use Markdown and include '## Answer', '## Key Points', and '## Supporting Evidence'.
- Keep evidence concise (2–4 bullets) with inline citations [n] matching the numbered context."""
            
            # Create messages for LLM
            messages = [
                {"role": "system", "content": self._get_system_prompt()},
                {"role": "user", "content": user_prompt}
            ]
            
            # Get LLM response
            answer = await chat_completion(
                messages, 
                is_legal_query=query_analysis['is_legal_query']
            )
            
            # Extract sources for backward compatibility
            sources = list({
                (ch.metadata.get("source") or ch.metadata.get("document") or "Source")
                for ch in final_chunks
            })
            
            # Compute document name from sources (for backward compatibility)
            document_name = sources[0] if sources else "multiple documents"
            
            return QueryResponse(
                success=True,
                answer=answer,
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
                metadata={"error": str(e)}
            )

    def _get_system_prompt(self) -> str:
        """Get system prompt for LLM."""
        return """You are a legal RAG assistant. Answer **only** from the supplied context.
- If the context is insufficient, say you don't have enough information.
- Write clean Markdown with headings and spacing.
- Keep "Supporting Evidence" to 2–4 concise bullets with inline citations like [1], [2].
- Do not invent facts or pages not present in the context.

Format:
## Answer
<2–4 short paragraphs (150–250 words total), with subheadings when helpful>

## Key Points
- <3–5 bullets summarizing the answer>

## Supporting Evidence
- <2–4 bullets, each maps to provided context and cites [n]>"""

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


# Global orchestrator instance
_orchestrator: Optional[RAGOrchestrator] = None

def get_rag_orchestrator() -> RAGOrchestrator:
    """Get singleton RAG orchestrator."""
    global _orchestrator
    if _orchestrator is None:
        _orchestrator = RAGOrchestrator()
    return _orchestrator
