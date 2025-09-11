from typing import List, Dict, Any, Tuple, Optional
import numpy as np
from langchain.schema import Document
import logging
import re
from collections import Counter
import math

logger = logging.getLogger(__name__)

class AdaptiveRetrievalService:
    """
    Advanced retrieval service with re-ranking and adaptive search strategies
    """
    
    def __init__(self):
        # Legal relevance scoring weights
        self.scoring_weights = {
            'semantic_similarity': 0.4,
            'legal_keyword_match': 0.2,
            'citation_relevance': 0.15,
            'page_proximity': 0.1,
            'context_coherence': 0.15
        }
        
        # Legal terminology for boosting
        self.legal_terms = [
            'contract', 'agreement', 'statute', 'regulation', 'case', 'court',
            'plaintiff', 'defendant', 'liability', 'damages', 'jurisdiction',
            'precedent', 'constitutional', 'federal', 'state', 'appeal',
            'motion', 'brief', 'discovery', 'deposition', 'testimony'
        ]
        
        # Citation patterns for legal documents
        self.citation_patterns = [
            r'\b\d+\s+F\.\s*\d*d\s+\d+\b',      # Federal Reporter
            r'\b\d+\s+U\.S\.\s+\d+\b',           # U.S. Reports
            r'\b\d+\s+S\.\s*Ct\.\s+\d+\b',      # Supreme Court Reporter
            r'\b\d+\s+U\.S\.C\.\s*§\s*\d+\b',   # U.S. Code
            r'\bRule\s+\d+\b',                   # Court Rules
            r'\b\d+\s+[A-Z][a-z]+\s+\d+\b'      # General case citation pattern
        ]

    async def adaptive_search_and_rerank(
        self, 
        query: str, 
        initial_results: List[Document], 
        query_analysis: Dict[str, Any]
    ) -> List[Document]:
        """
        Perform adaptive re-ranking based on query analysis
        """
        if not initial_results:
            return initial_results
        
        logger.info(f"Re-ranking {len(initial_results)} documents for query complexity: {query_analysis.get('complexity_score', 0):.2f}")
        
        # Calculate multiple relevance scores for each document
        scored_documents = []
        
        for doc in initial_results:
            scores = await self._calculate_relevance_scores(query, doc, query_analysis)
            total_score = self._combine_scores(scores)
            
            scored_documents.append({
                'document': doc,
                'total_score': total_score,
                'individual_scores': scores
            })
        
        # Sort by total score
        scored_documents.sort(key=lambda x: x['total_score'], reverse=True)
        
        # Log top scores for debugging
        if scored_documents:
            top_scores = [f"{doc['total_score']:.3f}" for doc in scored_documents[:3]]
            logger.info(f"Top 3 re-ranking scores: {top_scores}")
        
        # Return reranked documents
        return [item['document'] for item in scored_documents]

    async def _calculate_relevance_scores(
        self, 
        query: str, 
        document: Document, 
        query_analysis: Dict[str, Any]
    ) -> Dict[str, float]:
        """
        Calculate multiple relevance scores for a document
        """
        doc_text = document.page_content.lower()
        query_lower = query.lower()
        
        scores = {}
        
        # 1. Legal keyword matching score
        scores['legal_keyword_match'] = self._calculate_legal_keyword_score(
            query_lower, doc_text, query_analysis
        )
        
        # 2. Citation relevance score
        scores['citation_relevance'] = self._calculate_citation_score(doc_text, query_analysis)
        
        # 3. Page proximity score (prefer earlier pages for definitions)
        scores['page_proximity'] = self._calculate_page_proximity_score(
            document, query_analysis
        )
        
        # 4. Context coherence score
        scores['context_coherence'] = self._calculate_context_coherence_score(
            query_lower, doc_text
        )
        
        # 5. Semantic similarity score (placeholder - would use actual embeddings in production)
        scores['semantic_similarity'] = self._calculate_semantic_similarity_score(
            query_lower, doc_text
        )
        
        return scores

    def _calculate_legal_keyword_score(
        self, 
        query: str, 
        doc_text: str, 
        query_analysis: Dict[str, Any]
    ) -> float:
        """Calculate score based on legal keyword matches"""
        if not query_analysis.get('is_legal_query', False):
            return 0.5  # Neutral score for non-legal queries
        
        query_terms = set(query.split())
        doc_terms = set(doc_text.split())
        
        # Count legal term matches between query and document
        legal_matches = 0
        total_legal_in_query = 0
        
        for term in self.legal_terms:
            if term in query:
                total_legal_in_query += 1
                if term in doc_text:
                    legal_matches += 1
        
        if total_legal_in_query == 0:
            return 0.5  # Neutral score
        
        # Calculate match ratio with bonus for multiple matches
        match_ratio = legal_matches / total_legal_in_query
        bonus = min(legal_matches * 0.1, 0.3)  # Up to 30% bonus
        
        return min(match_ratio + bonus, 1.0)

    def _calculate_citation_score(self, doc_text: str, query_analysis: Dict[str, Any]) -> float:
        """Calculate score based on legal citations in document"""
        citation_count = 0
        
        for pattern in self.citation_patterns:
            matches = re.findall(pattern, doc_text, re.IGNORECASE)
            citation_count += len(matches)
        
        # Normalize citation count (more citations = higher legal authority)
        if citation_count == 0:
            return 0.3  # Low score for documents without citations
        elif citation_count <= 2:
            return 0.6  # Medium score
        else:
            return min(0.8 + (citation_count - 2) * 0.05, 1.0)  # High score with bonus

    def _calculate_page_proximity_score(
        self, 
        document: Document, 
        query_analysis: Dict[str, Any]
    ) -> float:
        """Calculate score based on page position in document"""
        page_num = document.metadata.get('page', 1)
        total_pages = document.metadata.get('total_pages', 1)
        
        # For definition queries, prefer earlier pages
        if query_analysis.get('query_type', {}).get('definition', False):
            # Higher score for earlier pages
            return max(0.3, 1.0 - (page_num - 1) / max(total_pages, 1) * 0.5)
        
        # For analysis queries, middle pages might be more relevant
        elif query_analysis.get('query_type', {}).get('analysis', False):
            middle_page = total_pages / 2
            distance_from_middle = abs(page_num - middle_page) / max(total_pages / 2, 1)
            return max(0.3, 1.0 - distance_from_middle * 0.4)
        
        # Neutral score for other query types
        return 0.5

    def _calculate_context_coherence_score(self, query: str, doc_text: str) -> float:
        """Calculate how coherent the document context is with the query"""
        query_words = set(query.split())
        doc_words = set(doc_text.split())
        
        # Calculate word overlap
        common_words = query_words.intersection(doc_words)
        
        # Filter out stop words (simplified)
        stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'is', 'are', 'was', 'were'}
        meaningful_common = common_words - stop_words
        
        if not query_words - stop_words:  # Avoid division by zero
            return 0.5
        
        coherence_score = len(meaningful_common) / len(query_words - stop_words)
        return min(coherence_score, 1.0)

    def _calculate_semantic_similarity_score(self, query: str, doc_text: str) -> float:
        """
        Placeholder for semantic similarity score
        In production, this would use actual embedding similarity
        """
        # Simple approximation based on word overlap and length
        query_words = set(query.split())
        doc_words = set(doc_text.split())
        
        if not query_words:
            return 0.5
        
        overlap = len(query_words.intersection(doc_words))
        similarity = overlap / len(query_words)
        
        # Length penalty for very short or very long documents
        doc_length = len(doc_text.split())
        if doc_length < 50:  # Very short
            similarity *= 0.8
        elif doc_length > 1000:  # Very long
            similarity *= 0.9
        
        return min(similarity, 1.0)

    def _combine_scores(self, scores: Dict[str, float]) -> float:
        """Combine individual scores using weighted average"""
        total_score = 0.0
        
        for score_type, weight in self.scoring_weights.items():
            score_value = scores.get(score_type, 0.5)  # Default to neutral
            total_score += weight * score_value
        
        return min(total_score, 1.0)

    def filter_by_relevance_threshold(
        self, 
        documents: List[Document], 
        threshold: float = 0.3
    ) -> List[Document]:
        """Filter documents below relevance threshold"""
        # This would typically use the scoring from reranking
        # For now, return all documents as a placeholder
        return documents

    def diversify_results(
        self, 
        documents: List[Document], 
        max_from_same_page: int = 2
    ) -> List[Document]:
        """Ensure diversity in results by limiting documents from same page"""
        page_counts = {}
        diversified_docs = []
        
        for doc in documents:
            page = doc.metadata.get('page', 1)
            current_count = page_counts.get(page, 0)
            
            if current_count < max_from_same_page:
                diversified_docs.append(doc)
                page_counts[page] = current_count + 1
        
        return diversified_docs
    
    def calculate_advanced_confidence(self, documents: List, query_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """
        Calculate sophisticated confidence based on multiple factors
        Returns detailed confidence analysis
        """
        if not documents:
            return {
                "confidence": "LOW",
                "score": 0.0,
                "factors": {
                    "chunk_relevance": 0.0,
                    "keyword_coverage": 0.0,
                    "information_completeness": 0.0,
                    "chunk_consistency": 0.0
                },
                "rationale": "No documents retrieved"
            }
        
        factors = {}
        
        # 1. Chunk relevance score (average similarity of top chunks)
        similarities = []
        for doc in documents[:3]:
            if hasattr(doc, 'similarity_score'):
                similarities.append(doc.similarity_score)
            else:
                similarities.append(0.5)  # Default similarity
        factors['chunk_relevance'] = np.mean(similarities) if similarities else 0.5
        
        # 2. Keyword coverage (how well documents cover query keywords)
        factors['keyword_coverage'] = self._calculate_keyword_coverage(
            query_analysis.get('keywords', []), documents
        )
        
        # 3. Information completeness (do chunks provide complete context)
        factors['information_completeness'] = self._assess_completeness(documents, query_analysis)
        
        # 4. Chunk consistency (do chunks agree with each other)
        factors['chunk_consistency'] = self._check_consistency(documents)
        
        # Calculate weighted confidence score
        weights = {
            'chunk_relevance': 0.3,
            'keyword_coverage': 0.3,
            'information_completeness': 0.2,
            'chunk_consistency': 0.2
        }
        
        weighted_score = sum(factors[key] * weights[key] for key in factors)
        
        # Determine confidence level
        if weighted_score > 0.8:
            confidence = "HIGH"
        elif weighted_score > 0.5:
            confidence = "MEDIUM"
        else:
            confidence = "LOW"
        
        # Generate rationale
        rationale = self._generate_confidence_rationale(factors, query_analysis)
        
        return {
            "confidence": confidence,
            "score": weighted_score,
            "factors": factors,
            "rationale": rationale
        }
    
    def _calculate_keyword_coverage(self, keywords: List[str], documents: List) -> float:
        """Calculate how well documents cover the query keywords"""
        if not keywords or not documents:
            return 0.5
        
        # Combine all document text
        all_text = " ".join([
            (doc.content if hasattr(doc, 'content') else doc.page_content).lower() 
            for doc in documents
        ])
        
        # Count covered keywords
        covered_keywords = sum(1 for keyword in keywords if keyword.lower() in all_text)
        
        return covered_keywords / len(keywords)
    
    def _assess_completeness(self, documents: List, query_analysis: Dict[str, Any]) -> float:
        """Assess if chunks provide complete context for the query"""
        if not documents:
            return 0.0
        
        # Check page diversity (more pages = more complete picture)
        pages = set(doc.metadata.get('page', 1) for doc in documents)
        page_diversity = min(len(pages) / 3, 1.0)  # Normalize to 3 pages max
        
        # Check total content length (more content = more complete)
        total_chars = sum(len(doc.content if hasattr(doc, 'content') else doc.page_content) for doc in documents)
        content_completeness = min(total_chars / 2000, 1.0)  # Normalize to 2000 chars
        
        # For complex queries, require more completeness
        complexity_factor = query_analysis.get('complexity_score', 0) / 10
        required_completeness = 0.5 + complexity_factor * 0.3
        
        completeness_score = (page_diversity * 0.6 + content_completeness * 0.4)
        
        return min(completeness_score / required_completeness, 1.0)
    
    def _check_consistency(self, documents: List) -> float:
        """Check if documents are consistent with each other"""
        if len(documents) < 2:
            return 1.0  # Single document is always consistent
        
        # Simple consistency check based on common terms
        doc_terms = []
        for doc in documents:
            content = doc.content if hasattr(doc, 'content') else doc.page_content
            terms = set(content.lower().split())
            doc_terms.append(terms)
        
        # Calculate pairwise similarity
        similarities = []
        for i in range(len(doc_terms)):
            for j in range(i + 1, len(doc_terms)):
                if doc_terms[i] and doc_terms[j]:
                    intersection = len(doc_terms[i].intersection(doc_terms[j]))
                    union = len(doc_terms[i].union(doc_terms[j]))
                    similarity = intersection / union if union > 0 else 0
                    similarities.append(similarity)
        
        return np.mean(similarities) if similarities else 0.5
    
    def _generate_confidence_rationale(self, factors: Dict[str, float], query_analysis: Dict[str, Any]) -> str:
        """Generate human-readable rationale for confidence score"""
        rationale_parts = []
        
        if factors['chunk_relevance'] > 0.7:
            rationale_parts.append("high semantic relevance")
        elif factors['chunk_relevance'] < 0.4:
            rationale_parts.append("low semantic relevance")
        
        if factors['keyword_coverage'] > 0.8:
            rationale_parts.append("excellent keyword coverage")
        elif factors['keyword_coverage'] < 0.5:
            rationale_parts.append("limited keyword coverage")
        
        if factors['information_completeness'] > 0.7:
            rationale_parts.append("comprehensive information")
        elif factors['information_completeness'] < 0.4:
            rationale_parts.append("incomplete information")
        
        if factors['chunk_consistency'] > 0.7:
            rationale_parts.append("consistent sources")
        elif factors['chunk_consistency'] < 0.4:
            rationale_parts.append("inconsistent sources")
        
        if query_analysis.get('is_legal_query', False):
            rationale_parts.append("legal domain context")
        
        return "; ".join(rationale_parts) if rationale_parts else "standard retrieval confidence"
