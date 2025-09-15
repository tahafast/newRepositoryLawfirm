from typing import List
import numpy as np
import asyncio
import logging
from openai import AsyncOpenAI
from core.config import settings

logger = logging.getLogger(__name__)

_embed_client: AsyncOpenAI | None = None


def _get_client() -> AsyncOpenAI:
    global _embed_client
    if _embed_client is None:
        _embed_client = AsyncOpenAI(
            api_key=settings.OPENAI_API_KEY, 
            timeout=120.0,  # Increased timeout
            max_retries=3   # Limit retries
        )
    return _embed_client


async def embed_text(text: str) -> List[float]:
    """Generate embedding for a single text with rate limiting."""
    client = _get_client()
    
    # Add rate limiting delay
    await asyncio.sleep(0.05)  # 50ms delay between requests
    
    try:
        logger.debug(f"Generating embedding for text of length {len(text)}")
        response = await client.embeddings.create(
            input=text, 
            model=settings.EMBEDDING_MODEL
        )
        logger.debug("Embedding generated successfully")
        return response.data[0].embedding
    except Exception as e:
        logger.warning(f"Embedding generation failed: {e}")
        raise


async def embed_texts_batch(texts: List[str]) -> List[List[float]]:
    """Generate embeddings for multiple texts in batches with rate limiting."""
    if not texts:
        return []
    
    client = _get_client()
    embeddings = []
    
    # Process in smaller batches to respect OpenAI rate limits
    batch_size = 20  # Reduced batch size for better rate limiting
    
    logger.info(f"Processing {len(texts)} texts in batches of {batch_size}")
    
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        batch_num = (i // batch_size) + 1
        total_batches = (len(texts) + batch_size - 1) // batch_size
        
        logger.info(f"Processing batch {batch_num}/{total_batches} ({len(batch)} texts)")
        
        try:
            # Add delay between batches to respect rate limits
            if i > 0:
                await asyncio.sleep(1.0)  # 1 second delay between batches
            
            response = await client.embeddings.create(
                input=batch,
                model=settings.EMBEDDING_MODEL
            )
            
            batch_embeddings = [data.embedding for data in response.data]
            embeddings.extend(batch_embeddings)
            
            logger.info(f"Successfully processed batch {batch_num}/{total_batches}")
            
        except Exception as e:
            logger.error(f"Failed to process batch {batch_num}: {e}")
            # Add fallback embeddings for failed batch
            fallback_embeddings = [[0.0] * 1536 for _ in batch]
            embeddings.extend(fallback_embeddings)
    
    logger.info(f"Completed embedding generation for {len(embeddings)} texts")
    return embeddings


async def hybrid_embed_text(text: str, *, legal_density: float | None = None) -> List[float]:
    base = np.array(await embed_text(text))
    if legal_density is None:
        return base.tolist()
    legal_weight = min(max(legal_density, 0.0), 1.0) * 0.3
    general_weight = 1.0 - legal_weight
    # simplistic legal vector: boost last 50 dims
    legal = np.zeros_like(base)
    legal[-50:] = 1.0
    if np.linalg.norm(legal) > 0:
        legal = legal / np.linalg.norm(legal)
    vec = general_weight * base + legal_weight * legal
    norm = np.linalg.norm(vec)
    return (vec / norm).tolist() if norm > 0 else vec.tolist()


