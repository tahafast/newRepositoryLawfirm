"""Vector search service using Qdrant with robust error handling."""

from typing import List, Dict, Any, Optional, Sequence
from langchain.schema import Document
from qdrant_client.models import PointStruct
import logging
import asyncio
import uuid
import typing as t

from app.modules.lawfirmchatbot.services.vector_store import get_qdrant, search_similar, upsert_embeddings
from app.modules.lawfirmchatbot.services.embeddings import embed_text, embed_texts_batch

logger = logging.getLogger(__name__)


def _extract_source(payload: dict) -> str:
    """Extract source/document name from payload."""
    if not isinstance(payload, dict):
        return "unknown"
    # common keys where we store file info
    for k in ("source", "file_name", "document", "doc_id"):
        v = payload.get(k)
        if isinstance(v, str) and v.strip():
            return v
    meta = payload.get("metadata")
    if isinstance(meta, dict):
        for k in ("source", "file_name", "document"):
            v = meta.get(k)
            if isinstance(v, str) and v.strip():
                return v
    return "unknown"


def _extract_text(payload: dict) -> str:
    """Extract text from payload, handling various key names and nested structures."""
    if not isinstance(payload, dict):
        return ""
    for k in ("text", "chunk", "page_content", "content", "body"):
        v = payload.get(k)
        if isinstance(v, str) and v.strip():
            return v
    meta = payload.get("metadata")
    if isinstance(meta, dict):
        for k in ("text", "chunk", "page_content", "content"):
            v = meta.get(k)
            if isinstance(v, str) and v.strip():
                return v
    return ""


def normalize_hits(hits: t.Sequence) -> list[dict]:
    """Return [{'text': str, 'source': str, 'score': float, 'id': ...}, ...] with empties filtered."""
    out = []
    for h in hits or []:
        payload = getattr(h, "payload", None) or {}
        text = _extract_text(payload)
        if not text:
            continue
        out.append({
            "text": text,
            "source": _extract_source(payload),
            "score": getattr(h, "score", None),
            "id": getattr(h, "id", None),
            "metadata": payload.get("metadata", {}),
        })
    return out


def build_user_prompt(query: str, contexts: list[dict]) -> str:
    """Build user prompt from contexts - multi-document, restart-safe; no self.current_document."""
    parts = []
    for i, c in enumerate(contexts, 1):
        parts.append(f"[{i}] Source: {c['source']}\n{c['text']}")
    snippets = "\n\n---\n".join(parts)
    return (
        "You are a legal assistant. Answer using ONLY the snippets; cite by [index].\n\n"
        f"{snippets}\n\n"
        f"Question: {query}\nAnswer:"
    )


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
            batch_num = (i // batch_size) + 1
            total_batches = (len(documents) + batch_size - 1) // batch_size
            
            logger.info(f"Processing document batch {batch_num}/{total_batches} ({len(batch)} documents)")
            
            # Generate embeddings for batch using the new batch function
            texts = [doc.page_content for doc in batch]
            try:
                embeddings = await embed_texts_batch(texts)
                logger.info(f"Generated {len(embeddings)} embeddings for batch {batch_num}")
            except Exception as e:
                logger.error(f"Batch embedding failed for batch {batch_num}: {e}")
                # Use zero vectors as fallback
                embeddings = [[0.0] * 1536 for _ in texts]
            
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
            
            # Upsert batch using vector_store helper
            if points:
                # Extract data for the new upsert_embeddings signature
                ids = [p.id for p in points]
                vectors = [p.vector for p in points]
                payloads = [p.payload for p in points]
                
                upsert_embeddings(
                    client=client,
                    ids=ids,
                    vectors=vectors,
                    payloads=payloads
                )
                total_added += len(points)
                logger.info(f"Successfully stored batch {batch_num}/{total_batches}: {len(points)} documents in Qdrant")
            
            # Brief pause between batches for Qdrant
            if i + batch_size < len(documents):
                await asyncio.sleep(0.5)  # Increased pause for better stability
        
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
        
        # Light observability: before search
        logger.info(f"vector_search start top_k={k}")
        
        import time
        start_time = time.time()
        
        # Search with primary threshold using vector_store helper
        results = search_similar(
            client=client,
            query_vector=query_embedding,
            top_k=k,
            filter_=None
        )
        
        # Light observability: after search
        elapsed_ms = int((time.time() - start_time) * 1000)
        logger.info(f"vector_search got hits={len(results or [])} in {elapsed_ms}ms")
        
        # If no results, retry with more results and no threshold
        if not results:
            logger.info("No results with default settings, retrying with more results")
            results = search_similar(
                client=client,
                query_vector=query_embedding,
                top_k=k * 2,
                filter_=None
            )
        
        # Use safe hit normalizer to prevent NoneType subscript errors
        raw_hits = results
        contexts = normalize_hits(raw_hits)
        
        if not contexts:
            # Don't raise; return empty list with logs. Avoid subscripting [].
            logger.warning("no usable contexts from qdrant hits; hits=%d", len(raw_hits or []))
            return []
        
        # Convert normalized hits to Documents
        documents = []
        for context in contexts[:k]:  # Limit to requested k
            doc = Document(
                page_content=context["text"],
                metadata=context["metadata"]
            )
            # Add similarity score for downstream processing
            if context["score"] is not None:
                doc.metadata["similarity_score"] = context["score"]
            documents.append(doc)
        
        logger.info(f"Retrieved {len(documents)} documents for query")
        return documents
        
    except Exception as e:
        logger.error(f"Similarity search failed: {str(e)}", exc_info=True)
        return []
