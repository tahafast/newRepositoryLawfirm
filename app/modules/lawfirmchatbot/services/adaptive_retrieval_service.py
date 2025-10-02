"""
Simple Adaptive Retrieval Service with Fast Intent Gate
Provides basic adaptive retrieval functionality with cheap intent detection
"""

from typing import Dict, Any, List
import logging
from core.config import settings

logger = logging.getLogger(__name__)

# Fast intent detection (no model calls)
SMALL_TALK = {"hi", "hello", "hey", "good morning", "good evening", "what can you do", "who are you"}
DOC_FREE_INTENTS = {"joke", "weather", "time"}

def _is_smalltalk(q: str) -> bool:
    """Detect small talk queries without expensive model calls."""
    s = q.strip().lower()
    return any(s.startswith(x) or s == x for x in SMALL_TALK)

def _looks_like_definition(q: str) -> bool:
    """Detect definition queries."""
    s = q.lower()
    return any(k in s for k in ["what is", "define", "explain", "meaning of", "overview of"])

def _is_comparison(q: str) -> bool:
    """Detect comparison queries."""
    s = q.lower()
    return " vs " in s or "compare" in s or "difference between" in s

def _get_adaptive_top_k(query: str) -> int:
    """Determine optimal top_k based on query characteristics."""
    if len(query) > 140 or _is_comparison(query):
        return settings.QDRANT_TOP_K_LONG_QUERY
    return settings.QDRANT_TOP_K_DEFAULT


class AdaptiveRetrievalService:
    """Simple adaptive retrieval service with fast intent gate."""
    
    def __init__(self):
        """Initialize the adaptive retrieval service."""
        logger.info("Initialized AdaptiveRetrievalService")
    
    def should_skip_retrieval(self, query: str) -> tuple[bool, str]:
        """
        Fast intent gate to determine if retrieval should be skipped.
        
        Returns:
            tuple: (should_skip, reason)
        """
        if _is_smalltalk(query):
            return True, "small_talk"
        
        if any(intent in query.lower() for intent in DOC_FREE_INTENTS):
            return True, "doc_free_intent"
        
        return False, "needs_retrieval"
    
    def get_adaptive_top_k(self, query: str) -> int:
        """Get adaptive top_k based on query characteristics."""
        return _get_adaptive_top_k(query)
    
    def adapt_retrieval_strategy(self, query_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Adapt retrieval strategy based on query analysis."""
        # Simple strategy adaptation
        strategy = {
            "k": query_analysis.get("suggested_k", 5),
            "score_threshold": 0.7 if query_analysis.get("is_legal_query", False) else 0.5,
            "rerank": query_analysis.get("complexity_score", 1) > 3
        }
        
        logger.debug(f"Adapted retrieval strategy: {strategy}")
        return strategy
    
    def calculate_advanced_confidence(self, chunks: List[Any], query_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate advanced confidence metrics for the retrieved chunks."""
        try:
            if not chunks:
                return {
                    "confidence": "low",
                    "score": 0.1,
                    "factors": ["no_chunks_retrieved"],
                    "rationale": "No relevant chunks were retrieved for this query."
                }
            
            # Extract similarity scores from chunks
            similarity_scores = []
            for chunk in chunks:
                if hasattr(chunk, 'metadata') and 'similarity_score' in chunk.metadata:
                    similarity_scores.append(chunk.metadata['similarity_score'])
                else:
                    # Default similarity score if not available
                    similarity_scores.append(0.5)
            
            # Calculate confidence metrics
            avg_similarity = sum(similarity_scores) / len(similarity_scores) if similarity_scores else 0.0
            max_similarity = max(similarity_scores) if similarity_scores else 0.0
            min_similarity = min(similarity_scores) if similarity_scores else 0.0
            
            # Determine confidence factors
            factors = []
            if len(chunks) >= 3:
                factors.append("sufficient_chunks")
            else:
                factors.append("limited_chunks")
            
            if avg_similarity >= 0.8:
                factors.append("high_similarity")
            elif avg_similarity >= 0.6:
                factors.append("moderate_similarity")
            else:
                factors.append("low_similarity")
            
            if query_analysis.get("is_legal_query", False):
                factors.append("legal_domain")
            
            if query_analysis.get("complexity_score", 1) > 3:
                factors.append("complex_query")
            
            # Calculate overall confidence score (0.0 to 1.0)
            confidence_score = min(1.0, (avg_similarity + (len(chunks) / 10)) / 2)
            
            # Determine confidence level
            if confidence_score >= 0.8:
                confidence_level = "high"
                rationale = f"High confidence based on {len(chunks)} relevant chunks with average similarity of {avg_similarity:.2f}"
            elif confidence_score >= 0.6:
                confidence_level = "medium"
                rationale = f"Medium confidence based on {len(chunks)} chunks with average similarity of {avg_similarity:.2f}"
            elif confidence_score >= 0.4:
                confidence_level = "low"
                rationale = f"Low confidence based on {len(chunks)} chunks with average similarity of {avg_similarity:.2f}"
            else:
                confidence_level = "very_low"
                rationale = f"Very low confidence - retrieved chunks may not be highly relevant to the query"
            
            result = {
                "confidence": confidence_level,
                "score": round(confidence_score, 3),
                "factors": factors,
                "rationale": rationale,
                "metrics": {
                    "chunks_count": len(chunks),
                    "avg_similarity": round(avg_similarity, 3),
                    "max_similarity": round(max_similarity, 3),
                    "min_similarity": round(min_similarity, 3)
                }
            }
            
            logger.debug(f"Calculated confidence: {result}")
            return result
            
        except Exception as e:
            logger.error(f"Error calculating confidence: {e}")
            return {
                "confidence": "unknown",
                "score": 0.5,
                "factors": ["calculation_error"],
                "rationale": "Could not calculate confidence due to an error."
            }
