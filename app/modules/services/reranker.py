from typing import List, Dict, Any, Tuple
import numpy as np
from langchain.schema import Document
import re
import logging
from collections import Counter
import math

logger = logging.getLogger(__name__)

class DocumentChunk:
    """Enhanced document chunk with similarity score"""
    def __init__(self, document: Document, similarity_score: float = 0.0):
        self.document = document
        self.content = document.page_content
        self.metadata = document.metadata
        self.similarity_score = similarity_score

class QueryComplexityAnalyzer:
    """Analyze query complexity to determine optimal retrieval strategy"""
    
    def __init__(self):
        self.complex_indicators = [
            'analyze', 'compare', 'contrast', 'evaluate', 'assess', 'examine',
            'relationship', 'implications', 'consequences', 'effect', 'impact',
            'how does', 'why does', 'what are the differences', 'similarities',
            'pros and cons', 'advantages', 'disadvantages', 'benefits', 'risks'
        ]
        
        self.legal_indicators = [
            'contract', 'agreement', 'statute', 'regulation', 'case law',
            'precedent', 'jurisdiction', 'liability', 'damages', 'court',
            'legal', 'law', 'constitutional', 'federal', 'state', 'appeal',
            'plaintiff', 'defendant', 'ruling', 'decision', 'holding'
        ]
        
        self.analytical_keywords = [
            'analyze', 'analysis', 'implications', 'consequences', 'relationship',
            'compare', 'contrast', 'evaluate', 'assessment', 'examination'
        ]
        
        self.factual_keywords = [
            'what is', 'who is', 'when', 'where', 'define', 'definition',
            'list', 'name', 'identify', 'specify', 'state', 'mention'
        ]

    def analyze(self, query: str) -> dict:
        """
        Analyze query to determine optimal retrieval strategy
        Returns comprehensive query analysis
        """
        query_lower = query.lower()
        words = query_lower.split()
        
        # Calculate complexity score
        complexity_indicators = sum(1 for indicator in self.complex_indicators 
                                   if indicator in query_lower)
        query_length_factor = min(len(words) / 10, 1.0)  # Normalize by length
        
        # Calculate legal relevance
        legal_matches = sum(1 for indicator in self.legal_indicators 
                           if indicator in query_lower)
        
        # Determine query type
        is_factual = any(keyword in query_lower for keyword in self.factual_keywords)
        is_analytical = any(keyword in query_lower for keyword in self.analytical_keywords)
        is_comparative = any(word in query_lower for word in ['compare', 'contrast', 'versus', 'vs', 'difference'])
        
        # Extract key terms (remove stop words)
        stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by'}
        keywords = [word for word in words if word not in stop_words and len(word) > 2]
        
        # Calculate final complexity score (1-10 scale)
        complexity_score = int(min(
            (complexity_indicators * 2 + query_length_factor * 3 + (1 if is_analytical else 0) * 2),
            10
        ))
        
        # Determine query classification
        if is_factual:
            query_type = "FACTUAL"
        elif is_analytical:
            query_type = "ANALYTICAL"
        elif is_comparative:
            query_type = "COMPARATIVE"
        else:
            query_type = "EXPLORATORY"
        
        return {
            'complexity_score': complexity_score,
            'is_legal_query': legal_matches > 0,
            'legal_relevance': min(legal_matches / 3, 1.0),
            'query_type': query_type,
            'keywords': keywords[:10],  # Top 10 keywords
            'suggested_k': self._suggest_retrieval_k(complexity_score, legal_matches),
            'retrieval_strategy': 'comprehensive' if complexity_score > 6 else 'precise'
        }
    
    def _suggest_retrieval_k(self, complexity_score: int, legal_matches: int) -> int:
        """Determine optimal number of chunks based on query analysis"""
        base_k = 5
        
        # Adjust based on complexity
        if complexity_score > 7:
            k = base_k * 2  # Complex queries need more context
        elif complexity_score < 3:
            k = max(3, base_k - 2)  # Simple queries need less
        else:
            k = base_k
        
        # Legal queries get 20% more chunks for thoroughness
        if legal_matches > 0:
            k = int(k * 1.2)
        
        return min(k, 15)  # Cap at 15 chunks

def adaptive_chunk_retrieval(query_analysis: dict) -> int:
    """Determine optimal number of chunks based on query analysis"""
    return query_analysis['suggested_k']

class DocumentReranker:
    """Multi-stage document re-ranking pipeline"""
    
    def __init__(self):
        self.weights = {
            'semantic_similarity': 0.4,
            'keyword_overlap': 0.3,
            'positional_relevance': 0.2,
            'information_density': 0.1
        }
        
        # Legal-specific terms for boosting
        self.legal_terms = [
            'contract', 'agreement', 'statute', 'regulation', 'case', 'court',
            'plaintiff', 'defendant', 'liability', 'damages', 'jurisdiction',
            'precedent', 'constitutional', 'federal', 'state', 'appeal'
        ]
    
    async def rerank_chunks(
        self, 
        query: str, 
        chunks: List[Document],
        query_analysis: dict
    ) -> List[DocumentChunk]:
        """
        Multi-factor re-ranking of retrieved chunks
        """
        if not chunks:
            return []
        
        # Convert to DocumentChunk objects
        document_chunks = [DocumentChunk(doc) for doc in chunks]
        
        logger.info(f"Re-ranking {len(document_chunks)} chunks for {query_analysis['query_type']} query")
        
        scored_chunks = []
        
        for chunk in document_chunks:
            scores = self._calculate_relevance_scores(query, chunk, query_analysis)
            
            # Adjust weights based on query type
            adjusted_weights = self._adjust_weights_for_query_type(query_analysis['query_type'])
            
            final_score = sum(scores[k] * adjusted_weights.get(k, self.weights[k]) for k in scores)
            
            chunk.similarity_score = final_score
            scored_chunks.append((chunk, final_score))
        
        # Sort by final score
        ranked = sorted(scored_chunks, key=lambda x: x[1], reverse=True)
        
        # Apply diversity filter
        diverse_chunks = self._apply_diversity_filter(ranked, max_from_same_section=2)
        
        # Log top scores for debugging
        if diverse_chunks:
            top_scores = [f"{chunk.similarity_score:.3f}" for chunk in diverse_chunks[:3]]
            logger.info(f"Top 3 re-ranking scores: {top_scores}")
        
        return diverse_chunks
    
    def _calculate_relevance_scores(self, query: str, chunk: DocumentChunk, query_analysis: dict) -> Dict[str, float]:
        """Calculate multiple relevance scores for a document chunk"""
        scores = {}
        
        # 1. Keyword overlap score
        scores['keyword_overlap'] = self._calculate_keyword_overlap(query, chunk.content)
        
        # 2. Positional relevance (prefer certain positions based on query type)
        scores['positional_relevance'] = self._calculate_positional_score(chunk, query_analysis)
        
        # 3. Information density score
        scores['information_density'] = self._calculate_information_density(chunk.content)
        
        # 4. Semantic similarity (use existing score or calculate)
        scores['semantic_similarity'] = getattr(chunk, 'similarity_score', 0.7)  # Default if not available
        
        return scores
    
    def _calculate_keyword_overlap(self, query: str, content: str) -> float:
        """Calculate keyword overlap score using TF-IDF inspired approach"""
        query_words = set(query.lower().split())
        content_words = content.lower().split()
        content_word_set = set(content_words)
        
        # Remove stop words
        stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by'}
        query_words = query_words - stop_words
        content_word_set = content_word_set - stop_words
        
        if not query_words:
            return 0.5
        
        # Calculate overlap
        overlap = len(query_words.intersection(content_word_set))
        base_score = overlap / len(query_words)
        
        # Boost for legal terms
        legal_boost = 0
        for term in self.legal_terms:
            if term in query.lower() and term in content.lower():
                legal_boost += 0.1
        
        return min(base_score + legal_boost, 1.0)
    
    def _calculate_positional_score(self, chunk: DocumentChunk, query_analysis: dict) -> float:
        """Calculate score based on chunk position in document"""
        page_num = chunk.metadata.get('page', 1)
        total_pages = chunk.metadata.get('total_pages', 1)
        
        if total_pages <= 1:
            return 0.5
        
        # For factual queries, prefer earlier pages (definitions, introductions)
        if query_analysis['query_type'] == 'FACTUAL':
            return max(0.3, 1.0 - (page_num - 1) / total_pages * 0.6)
        
        # For analytical queries, middle sections often contain main content
        elif query_analysis['query_type'] == 'ANALYTICAL':
            middle_page = total_pages / 2
            distance_from_middle = abs(page_num - middle_page) / (total_pages / 2)
            return max(0.3, 1.0 - distance_from_middle * 0.4)
        
        # Neutral score for other types
        return 0.5
    
    def _calculate_information_density(self, content: str) -> float:
        """Calculate information density of the chunk"""
        words = content.split()
        if len(words) == 0:
            return 0.0
        
        # Count unique words (vocabulary richness)
        unique_words = len(set(words))
        vocab_density = unique_words / len(words)
        
        # Count sentences (structural complexity)
        sentences = len([s for s in content.split('.') if s.strip()])
        sentence_density = min(sentences / 10, 1.0)  # Normalize
        
        # Count numbers and specific terms (information richness)
        numbers = len(re.findall(r'\b\d+\b', content))
        number_density = min(numbers / 20, 1.0)  # Normalize
        
        # Combine factors
        density_score = (vocab_density * 0.4 + sentence_density * 0.4 + number_density * 0.2)
        
        return min(density_score, 1.0)
    
    def _adjust_weights_for_query_type(self, query_type: str) -> Dict[str, float]:
        """Adjust scoring weights based on query type"""
        if query_type == 'FACTUAL':
            # Prioritize keyword matching for factual queries
            return {
                'semantic_similarity': 0.3,
                'keyword_overlap': 0.5,
                'positional_relevance': 0.15,
                'information_density': 0.05
            }
        elif query_type == 'ANALYTICAL':
            # Prioritize semantic understanding and information density
            return {
                'semantic_similarity': 0.5,
                'keyword_overlap': 0.2,
                'positional_relevance': 0.1,
                'information_density': 0.2
            }
        else:
            # Use default weights
            return self.weights
    
    def _apply_diversity_filter(self, chunks: List[Tuple[DocumentChunk, float]], max_from_same_section: int = 2) -> List[DocumentChunk]:
        """Ensure diversity in retrieved chunks"""
        page_counts = {}
        diversified_chunks = []
        
        for chunk, score in chunks:
            page = chunk.metadata.get('page', 1)
            current_count = page_counts.get(page, 0)
            
            if current_count < max_from_same_section:
                diversified_chunks.append(chunk)
                page_counts[page] = current_count + 1
        
        return diversified_chunks
