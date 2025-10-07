from pydantic import BaseModel
from typing import List, Optional, Dict, Any


class ConfidenceFactors(BaseModel):
    chunk_relevance: float
    keyword_coverage: float
    information_completeness: float
    chunk_consistency: float


class QueryRequest(BaseModel):
    query: str
    conversation_id: Optional[str] = None
    user_id: str = "anon"


class DebugQueryAnalysis(BaseModel):
    complexity_score: int
    is_legal_query: bool
    legal_relevance: float
    query_type: str
    keywords: List[str]
    suggested_k: int
    retrieval_strategy: str


class DebugInfo(BaseModel):
    query_analysis: DebugQueryAnalysis
    retrieval_k: int
    reranking_applied: bool
    total_pages: Optional[int] = None


class QueryResponse(BaseModel):
    success: bool
    answer: str
    answer_markdown: Optional[str] = None
    metadata: Dict[str, Any]
    debug_info: Optional[DebugInfo] = None


