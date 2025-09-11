from pydantic import BaseModel
from typing import List


class QueryAnalysis(BaseModel):
    complexity_score: int
    is_legal_query: bool
    legal_relevance: float
    query_type: str
    keywords: List[str]
    suggested_k: int
    retrieval_strategy: str


