"""
Simple Query Analyzer Stub
Provides basic query analysis functionality
"""

from typing import Dict, Any, List
import logging

logger = logging.getLogger(__name__)


# Document generation intent detection
DOCGEN_TRIGGERS = [
    "generate", "draft", "make", "prepare", "compose", "create", "build",
    "format", "convert to", "produce", "write"
]

DOC_TYPES_PK = [
    "affidavit", "counter affidavit", "legal notice", "synopsis", "rejoinder",
    "written statement", "reply", "para-wise reply", "writ petition",
    "application", "stay application", "undertaking", "power of attorney",
    "petition", "plaint", "suit", "complaint"
]


def is_docgen_request(query: str) -> bool:
    """
    Detect if the query is requesting document generation.
    
    Args:
        query: User query string
        
    Returns:
        True if the query appears to be requesting document generation
    """
    q_lower = (query or "").lower()
    has_trigger = any(trigger in q_lower for trigger in DOCGEN_TRIGGERS)
    has_doc_type = any(doc_type in q_lower for doc_type in DOC_TYPES_PK)
    return has_trigger and has_doc_type


class QueryComplexityAnalyzer:
    """Simple query complexity analyzer."""
    
    def analyze(self, query: str) -> Dict[str, Any]:
        """Analyze query complexity and return analysis."""
        # Simple analysis based on query length and keywords
        legal_keywords = ['contract', 'legal', 'law', 'clause', 'agreement', 'liability', 'terms']
        
        words = query.lower().split()
        complexity_score = min(len(words) // 5 + 1, 5)  # 1-5 scale
        
        is_legal_query = any(keyword in query.lower() for keyword in legal_keywords)
        legal_relevance = sum(1 for keyword in legal_keywords if keyword in query.lower()) / len(legal_keywords)
        
        # Check for docgen intent
        is_docgen = is_docgen_request(query)
        query_type = "docgen" if is_docgen else ("legal" if is_legal_query else "general")
        
        # For docgen, suggest higher k to get more template context
        suggested_k = 12 if is_docgen else max(3, min(complexity_score * 2, 10))
        
        return {
            "complexity_score": complexity_score,
            "is_legal_query": is_legal_query,
            "legal_relevance": legal_relevance,
            "query_type": query_type,
            "keywords": [word for word in words if len(word) > 3],
            "suggested_k": suggested_k,
            "retrieval_strategy": "semantic",
            "mode": "docgen" if is_docgen else "qa"
        }
