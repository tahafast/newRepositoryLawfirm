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

# === PATCH 3: Correct Qdrant Filter Syntax Helper ===
def build_conversation_filter(conversation_id: str):
    """
    Build conversation-scoped filter for Qdrant retrieval.
    Uses correct key-match syntax: key="conversation_id", match=MatchValue(value=...)
    
    Args:
        conversation_id: The conversation identifier to filter by
        
    Returns:
        Filter object configured for conversation-scoped queries
    """
    from qdrant_client.models import Filter, FieldCondition, MatchValue
    
    return Filter(
        must=[
            FieldCondition(key="conversation_id", match=MatchValue(value=conversation_id))
        ]
    )


# === PATCH 2 (NEW): Unified Filter Builder for Multi-File Aggregation ===
def build_ephemeral_filter(conversation_id: str, file_ids: list = None):
    """
    Build merged ephemeral filter combining all file_ids under the same conversation.
    
    When multiple documents are attached, this filter retrieves hits from ANY of them,
    ensuring comprehensive ephemeral coverage without KB fallback.
    
    Args:
        conversation_id: The conversation identifier
        file_ids: Optional list of file_ids to include (None = all for this conversation)
        
    Returns:
        Filter object with merged file_id conditions
    """
    from qdrant_client.models import Filter, FieldCondition, MatchValue, HasIdCondition
    
    if not file_ids:
        # No specific files: just filter by conversation
        return Filter(
            must=[FieldCondition(key="conversation_id", match=MatchValue(value=conversation_id))]
        )
    
    # Build merged filter: conversation_id AND (file_id=f1 OR file_id=f2 OR ...)
    should_conditions = [
        FieldCondition(key="file_id", match=MatchValue(value=fid)) 
        for fid in file_ids if fid
    ]
    
    if not should_conditions:
        # Fallback if all file_ids filtered out
        return Filter(
            must=[FieldCondition(key="conversation_id", match=MatchValue(value=conversation_id))]
        )
    
    # Return filter: must have conversation_id, should match any file_id
    return Filter(
        must=[FieldCondition(key="conversation_id", match=MatchValue(value=conversation_id))],
        should=should_conditions,
        min_should_match=1 if len(should_conditions) > 0 else 0
    )

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
        self.qdrant_client = None
    
    async def _get_qdrant_client(self):
        """Lazy initialize Qdrant client for conversation context."""
        if self.qdrant_client is None:
            try:
                import os
                from qdrant_client import QdrantClient
                self.qdrant_client = QdrantClient(
                    url=os.getenv("QDRANT_URL", "http://localhost:6333"),
                    api_key=os.getenv("QDRANT_API_KEY")
                )
            except Exception as e:
                logger.warning(f"[adaptive-retrieval] Failed to initialize Qdrant client: {e}")
                return None
        return self.qdrant_client
    
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
    
    async def get_conversation_context(self, conversation_id: str, limit: int = 5) -> List[Dict[str, Any]]:
        """
        PATCH_4: Retrieve conversation context filtered by conversation_id and sorted by timestamp.
        This ensures we only pull messages from the current conversation.
        
        Args:
            conversation_id: The conversation to filter by
            limit: Maximum number of messages to retrieve
            
        Returns:
            List of messages sorted by timestamp (oldest first)
        """
        try:
            qdrant = await self._get_qdrant_client()
            if not qdrant:
                logger.warning("[conversation-context] Qdrant client unavailable")
                return []
            
            from qdrant_client.models import Filter, FieldCondition, MatchValue
            from datetime import datetime as dt
            
            # Scroll with conversation_id filter
            results = qdrant.scroll(
                collection_name="conversation_memory",  # Adjust collection name as needed
                scroll_filter=Filter(
                    must=[FieldCondition(key="conversation_id", match=MatchValue(value=conversation_id))]
                ),
                limit=limit,
                with_payload=True,
            )
            
            messages = []
            points = results[0] if isinstance(results, tuple) else results
            
            for point in points:
                payload = getattr(point, 'payload', {}) or {}
                msg = {
                    "role": payload.get("role", "user"),
                    "text": payload.get("text") or payload.get("content") or "",
                    "ts": payload.get("ts") or payload.get("timestamp") or dt.utcnow().isoformat(),
                    "conversation_id": payload.get("conversation_id", conversation_id),
                }
                if msg["text"].strip():
                    messages.append(msg)
            
            # Sort by timestamp to preserve proper sequence (oldest first)
            messages.sort(key=lambda m: m.get("ts", ""))
            
            logger.info(f"[conversation-context] Retrieved {len(messages)} messages for conversation {conversation_id}")
            return messages
        
        except Exception as e:
            logger.warning(f"[conversation-context] Failed to retrieve context: {e}")
            return []
    
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
