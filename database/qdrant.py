"""
Qdrant Vector Database Configuration and Management
Handles connection and operations with Qdrant vector database
"""

from typing import List, Dict, Any, Optional
import logging
from qdrant_client import QdrantClient
from qdrant_client.http import models
from qdrant_client.http.models import Distance, VectorParams, PointStruct
from core.config import settings

logger = logging.getLogger(__name__)


class QdrantManager:
    """Manages Qdrant vector database operations"""
    
    def __init__(self):
        """Initialize Qdrant client"""
        self.client = QdrantClient(
            host=settings.QDRANT_HOST,
            port=settings.QDRANT_PORT
        )
        self.collection_name = settings.QDRANT_COLLECTION
        
    async def initialize_collection(self, vector_size: int = 1536):
        """Initialize the collection if it doesn't exist"""
        try:
            # Check if collection exists
            collections = self.client.get_collections()
            collection_names = [col.name for col in collections.collections]
            
            if self.collection_name not in collection_names:
                logger.info(f"Creating collection: {self.collection_name}")
                self.client.create_collection(
                    collection_name=self.collection_name,
                    vectors_config=VectorParams(
                        size=vector_size,
                        distance=Distance.COSINE
                    )
                )
                logger.info(f"Collection {self.collection_name} created successfully")
            else:
                logger.info(f"Collection {self.collection_name} already exists")
                
        except Exception as e:
            logger.error(f"Error initializing collection: {str(e)}")
            raise
    
    async def add_documents(self, documents: List[Dict[str, Any]]) -> bool:
        """Add documents to the collection"""
        try:
            points = []
            for i, doc in enumerate(documents):
                point = PointStruct(
                    id=doc.get('id', i),
                    vector=doc['embedding'],
                    payload={
                        'text': doc['text'],
                        'metadata': doc.get('metadata', {}),
                        'filename': doc.get('filename', ''),
                        'chunk_index': doc.get('chunk_index', 0)
                    }
                )
                points.append(point)
            
            self.client.upsert(
                collection_name=self.collection_name,
                points=points
            )
            
            logger.info(f"Added {len(documents)} documents to collection")
            return True
            
        except Exception as e:
            logger.error(f"Error adding documents: {str(e)}")
            return False
    
    async def search_documents(
        self, 
        query_vector: List[float], 
        limit: int = 5,
        score_threshold: float = 0.7
    ) -> List[Dict[str, Any]]:
        """Search for similar documents"""
        try:
            search_result = self.client.search(
                collection_name=self.collection_name,
                query_vector=query_vector,
                limit=limit,
                score_threshold=score_threshold
            )
            
            results = []
            for point in search_result:
                result = {
                    'id': point.id,
                    'score': point.score,
                    'text': point.payload.get('text', ''),
                    'metadata': point.payload.get('metadata', {}),
                    'filename': point.payload.get('filename', ''),
                    'chunk_index': point.payload.get('chunk_index', 0)
                }
                results.append(result)
            
            logger.info(f"Found {len(results)} similar documents")
            return results
            
        except Exception as e:
            logger.error(f"Error searching documents: {str(e)}")
            return []
    
    async def delete_collection(self) -> bool:
        """Delete the entire collection"""
        try:
            self.client.delete_collection(self.collection_name)
            logger.info(f"Collection {self.collection_name} deleted successfully")
            return True
        except Exception as e:
            logger.error(f"Error deleting collection: {str(e)}")
            return False
    
    async def get_collection_info(self) -> Optional[Dict[str, Any]]:
        """Get information about the collection"""
        try:
            info = self.client.get_collection(self.collection_name)
            return {
                'name': info.name,
                'status': info.status,
                'vectors_count': info.vectors_count,
                'indexed_vectors_count': info.indexed_vectors_count,
                'points_count': info.points_count
            }
        except Exception as e:
            logger.error(f"Error getting collection info: {str(e)}")
            return None


# Global Qdrant manager instance
qdrant_manager = QdrantManager()
