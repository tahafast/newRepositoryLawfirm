"""Main RAG orchestration service with modular architecture."""

from typing import Dict, Any, Optional
from datetime import datetime
import logging

from app.services.ingestion.document_processor import process_document
from app.services.retrieval.vector_search import add_documents_to_vector_store, search_similar_documents
from app.services.llm import chat_completion
from app.schemas.query import QueryResponse, DebugInfo, DebugQueryAnalysis
from app.modules.services.reranker import QueryComplexityAnalyzer, DocumentReranker
from app.modules.services.adaptive_retrieval_service import AdaptiveRetrievalService

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
            if not self.current_document:
                return QueryResponse(
                    success=False,
                    answer="⚠️ No document has been uploaded yet. Please upload a document first.",
                    metadata={"error": "no_document_uploaded"}
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
                    answer="⚠️ INFORMATION GAP: I couldn't find relevant information in the document for your query.",
                    metadata={
                        "document": self.current_document['name'],
                        "chunks_retrieved": 0,
                        "chunks_used": 0,
                        "query_complexity": query_analysis['complexity_score'],
                        "confidence": "LOW"
                    },
                    debug_info=DebugInfo(
                        query_analysis=DebugQueryAnalysis(**query_analysis),
                        retrieval_k=0,
                        reranking_applied=False,
                        total_pages=self.current_document['total_pages']
                    )
                )
            
            # Re-rank chunks
            reranked_chunks = await self.reranker.rerank_chunks(
                query, initial_chunks, query_analysis
            )
            
            # Limit final chunks
            final_chunks = reranked_chunks[:min(len(reranked_chunks), 8)]
            
            # Calculate confidence
            confidence_analysis = self.retrieval_service.calculate_advanced_confidence(
                final_chunks, query_analysis
            )
            
            # Create messages for LLM
            messages = [
                {"role": "system", "content": self._get_system_prompt()},
                {"role": "user", "content": self._create_user_prompt(query, final_chunks)}
            ]
            
            # Get LLM response
            answer = await chat_completion(
                messages, 
                is_legal_query=query_analysis['is_legal_query']
            )
            
            return QueryResponse(
                success=True,
                answer=answer,
                metadata={
                    "document": self.current_document['name'],
                    "chunks_retrieved": k,
                    "chunks_used": len(final_chunks),
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
                    total_pages=self.current_document['total_pages']
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
        return """You are an advanced document analysis AI. Provide accurate, well-cited responses based solely on the provided document chunks.

RESPONSE FORMAT:
═══════════════════════════════════════
📄 DOCUMENT: {document_name}
🎯 QUERY TYPE: {query_type}
═══════════════════════════════════════

ANSWER:
{your_detailed_answer}

SUPPORTING EVIDENCE:
• [Page X]: "{relevant_quote}"

CONFIDENCE ASSESSMENT:
• Overall: {HIGH/MEDIUM/LOW}
• Rationale: {explanation}
═══════════════════════════════════════

CRITICAL: Only use information from the provided chunks. If information is not available, clearly state this."""

    def _create_user_prompt(self, query: str, chunks) -> str:
        """Create user prompt with chunks."""
        chunk_text = "\n\n".join([
            f"[Page {chunk.metadata.get('page', '?')}]: {chunk.content}"
            for chunk in chunks
        ])
        
        return f"""Document: {self.current_document['name']}
Total Pages: {self.current_document['total_pages']}

RELEVANT CHUNKS:
{chunk_text}

USER QUERY: {query}

Please answer the query using only the information from the chunks above."""


# Global orchestrator instance
_orchestrator: Optional[RAGOrchestrator] = None

def get_rag_orchestrator() -> RAGOrchestrator:
    """Get singleton RAG orchestrator."""
    global _orchestrator
    if _orchestrator is None:
        _orchestrator = RAGOrchestrator()
    return _orchestrator
