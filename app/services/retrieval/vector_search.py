"""Vector search service using Qdrant with robust error handling."""

from typing import List, Dict, Any, Optional
from langchain.schema import Document
from qdrant_client.models import PointStruct
import logging
import asyncio
import uuid

from app.services.vector_store import get_qdrant
from app.services.embeddings import embed_text
from core.conf import settings

logger = logging.getLogger(__name__)


async def add_documents_to_vector_store(documents: List[Document]) -> int:
    """Add documents to vector store with robust batch processing."""
    if not documents:
        return 0
    
    try:
        logger.info(f"Adding {len(documents)} documents to vector store")
        client = get_qdrant()
        
        # Process in smaller batches to avoid timeouts
        batch_size = 32
        total_added = 0
        
        for i in range(0, len(documents), batch_size):
            batch = documents[i:i + batch_size]
            
            # Generate embeddings for batch
            texts = [doc.page_content for doc in batch]
            embeddings = []
            
            for text in texts:
                try:
                    embedding = await embed_text(text)
                    embeddings.append(embedding)
                except Exception as e:
                    logger.warning(f"Embedding failed for text chunk: {e}")
                    # Use zero vector as fallback
                    embeddings.append([0.0] * 1536)
            
            # Create points
            points = []
            for doc, embedding in zip(batch, embeddings):
                points.append(PointStruct(
                    id=str(uuid.uuid4()),
                    vector=embedding,
                    payload={
                        "text": doc.page_content,
                        "metadata": doc.metadata
                    }
                ))
            
            # Upsert batch
            if points:
                client.upsert(
                    collection_name=settings.QDRANT_COLLECTION,
                    points=points,
                    wait=True
                )
                total_added += len(points)
                logger.debug(f"Processed batch {i//batch_size + 1}: {len(points)} documents")
            
            # Brief pause between batches
            if i + batch_size < len(documents):
                await asyncio.sleep(0.1)
        
        logger.info(f"Successfully added {total_added} documents to vector store")
        return total_added
        
    except Exception as e:
        logger.error(f"Error adding documents to vector store: {str(e)}", exc_info=True)
        raise


async def search_similar_documents(query: str, k: int = 5) -> List[Document]:
    """Search for similar documents with fallback strategies."""
    try:
        client = get_qdrant()
        
        # Generate query embedding
        query_embedding = await embed_text(query)
        
        # Search with primary threshold
        results = client.search(
            collection_name=settings.QDRANT_COLLECTION,
            query_vector=query_embedding,
            limit=k,
            score_threshold=0.2
        )
        
        # If no results, retry with relaxed threshold
        if not results:
            logger.info("No results with threshold 0.2, retrying with no threshold")
            results = client.search(
                collection_name=settings.QDRANT_COLLECTION,
                query_vector=query_embedding,
                limit=k * 2
            )
        
        # Convert to Documents
        documents = []
        for result in results[:k]:  # Limit to requested k
            doc = Document(
                page_content=result.payload["text"],
                metadata=result.payload["metadata"]
            )
            # Add similarity score for downstream processing
            doc.metadata["similarity_score"] = result.score
            documents.append(doc)
        
        logger.info(f"Retrieved {len(documents)} documents for query")
        return documents
        
    except Exception as e:
        logger.error(f"Similarity search failed: {str(e)}", exc_info=True)
        return []
