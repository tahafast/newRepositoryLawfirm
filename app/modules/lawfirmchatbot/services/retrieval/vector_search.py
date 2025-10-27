"""Vector search service using Qdrant with robust error handling."""

from typing import List, Dict, Any, Optional, Sequence
from app.modules.lawfirmchatbot.services._lc_compat import ensure_Document
from qdrant_client.models import PointStruct
import logging
import asyncio
import uuid
import typing as t

Document = ensure_Document()

from app.modules.lawfirmchatbot.services.vector_store import get_qdrant, search_similar, upsert_embeddings
from app.modules.lawfirmchatbot.services.text_cleaner import CLEAN_PIPELINE_VERSION
from app.modules.lawfirmchatbot.services.llm import embed_text_async
from app.modules.lawfirmchatbot.services.qdrant_collections import get_ephemeral_collection
from core.config import settings
from core.utils.perf import profile_stage

logger = logging.getLogger(__name__)

# -------- Retrieval tuning & lightweight cache --------

TOP_K = min(getattr(settings, "QDRANT_TOP_K_DEFAULT", 8), 8)
FETCH_K = settings.QDRANT_TOP_K_LONG_QUERY
MIN_SCORE = getattr(settings, "QDRANT_SCORE_THRESHOLD", 0.18)
CACHE_TTL_SECONDS = 60
_cache: dict = {}


def _normalize_query(q: str) -> str:
    q = (q or "").strip().lower()
    if "summary" in q:
        q += " overview abstract synopsis"
    if "types" in q:
        q += " kinds categories variants"
    if "compare" in q or " vs" in q or "versus" in q:
        q += " comparison differences pros cons"
    return q


def _cache_get(key):
    item = _cache.get(key)
    if not item:
        return None
    import time as _t
    if _t.time() - item[0] > CACHE_TTL_SECONDS:
        _cache.pop(key, None)
        return None
    return item[1]


def _cache_set(key, value):
    import time as _t
    _cache[key] = (_t.time(), value)


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
        # Handle both dict format (from search_similar with score_threshold) and Qdrant object format
        if isinstance(h, dict):
            # Already in dict format
            text = h.get("text", "")
            source = h.get("metadata", {}).get("source", "unknown")
            score = h.get("score")
            item_id = h.get("id")
            metadata = h.get("metadata", {})
        else:
            # Qdrant object format
            payload = getattr(h, "payload", None) or {}
            text = _extract_text(payload)
            source = _extract_source(payload)
            score = getattr(h, "score", None)
            item_id = getattr(h, "id", None)
            metadata = payload.get("metadata", {})
        
        if not text:
            continue
            
        out.append({
            "text": text,
            "source": source,
            "score": score,
            "id": item_id,
            "metadata": metadata,
        })
    return out


def _deduplicate_chunks(chunks: list[dict]) -> tuple[list[dict], list[str], list[int]]:
    """
    Deduplicate chunks by (document_id, page_number) and return clean results.
    
    Returns:
        tuple: (deduplicated_chunks, unique_sources, unique_pages)
    """
    seen = set()
    deduplicated = []
    unique_sources = []
    unique_pages = []
    
    for chunk in chunks:
        # Create deduplication key
        source = chunk.get("source", "unknown")
        metadata = chunk.get("metadata", {})
        
        # Extract page number
        page = None
        for key in ("page", "page_number", "pageIndex", "page_index"):
            p = metadata.get(key)
            if isinstance(p, (int, float)):
                page = int(p)
                break
        
        dedup_key = (source, page)
        
        if dedup_key not in seen:
            seen.add(dedup_key)
            deduplicated.append(chunk)
            
            if source not in unique_sources:
                unique_sources.append(source)
            
            if page is not None and page not in unique_pages:
                unique_pages.append(page)
    
    # Sort pages for consistent output
    unique_pages.sort()
    
    return deduplicated, unique_sources, unique_pages


def build_user_prompt(query: str, contexts: list[dict]) -> str:
    """Build user prompt from contexts - BRAG AI Rich Markdown format."""
    parts = []
    for i, c in enumerate(contexts, 1):
        parts.append(f"[{i}] Source: {c['source']}\n{c['text']}")
    snippets = "\n\n---\n".join(parts)
    return (
        f"USER_QUERY: {query}\n\n"
        f"RETRIEVED_CONTEXT:\n{snippets}\n\n"
        "INSTRUCTIONS - Follow BRAG AI Rich Markdown format:\n"
        "1. Length: 350–400 words\n"
        "2. Opening: direct sentence with takeaway (no meta talk)\n"
        "3. Headings: ## main title, ### sections (NOT 'Understanding'), #### Pros/Cons for comparisons\n"
        "4. Structure:\n"
        "   - COMPARISON: lead-in → mandatory table (3+ rows) → #### Pros/#### Cons per approach (2-4 bullets)\n"
        "   - EXPLAIN: 2-3 ### sections (Key Idea, How It Works, Practical Notes, Examples/Applications)\n"
        "5. Citations: [1], [2] inline when using context; References: <Doc Title, p. X–Y>; <Another Doc, p. Z>\n"
        "6. If no context: 'I couldn't find anything similar in the uploaded documents, but here's what I can share more generally—' + OMIT References\n"
        "7. FORBIDDEN: 'Limited Information Available', 'See available documents', 'As an AI'"
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
            sanitized_batch = []
            for doc in batch:
                content = (getattr(doc, "page_content", "") or "").strip()
                if not content:
                    continue
                doc.page_content = content
                if doc.metadata is None:
                    doc.metadata = {}
                doc.metadata.setdefault("clean_version", CLEAN_PIPELINE_VERSION)
                sanitized_batch.append(doc)

            if not sanitized_batch:
                continue
            batch_num = (i // batch_size) + 1
            total_batches = (len(documents) + batch_size - 1) // batch_size
            
            logger.info(f"Processing document batch {batch_num}/{total_batches} ({len(batch)} documents)")
            
            # Generate embeddings for batch using the optimized function
            texts = [doc.page_content for doc in sanitized_batch]
            try:
                embeddings = []
                for text in texts:
                    embedding = await embed_text_async(text)
                    embeddings.append(embedding)
                logger.info(f"Generated {len(embeddings)} embeddings for batch {batch_num}")
            except Exception as e:
                logger.error(f"Batch embedding failed for batch {batch_num}: {e}")
                # Use zero vectors as fallback
                embeddings = [[0.0] * 1536 for _ in texts]
            
            # Create points
            points = []
            for doc, embedding in zip(sanitized_batch, embeddings):
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


@profile_stage("vector_search")
async def search_similar_documents(query: str, k: int | None = None, score_threshold: float | None = None):
    """Faster but tolerant retrieval — allows partial results."""
    k = k or settings.RETRIEVAL_TOP_K
    score_threshold = score_threshold if score_threshold is not None else settings.RETRIEVAL_SCORE_THRESH

    client = get_qdrant()
    vec = await embed_text_async(query)

    # Use search parameters that are compatible with different Qdrant versions
    search_kwargs = {
        "collection_name": settings.QDRANT_COLLECTION,
        "query_vector": vec,
        "limit": k,
        "with_payload": True,
        "with_vectors": False,
    }
    
    # Add search parameters if supported
    try:
        search_kwargs["search_params"] = {"hnsw_ef": settings.QDRANT_HNSW_EF, "exact": settings.QDRANT_EXACT}
    except Exception:
        # Fallback for older versions that don't support search_params
        pass
    
    res = client.search(**search_kwargs)

    # Remove hard early exit — return partials if available
    if not res:
        return ([], [], [])

    # Keep partials even if scores are low, let LLM decide relevance
    seen = set(); docs=[]; sources=set(); pages=set()
    for p in res:
        pl = p.payload or {}
        # Extract metadata - check both direct payload and nested metadata
        metadata = pl.get("metadata", {})
        doc = (metadata.get("source") or metadata.get("document") or 
               pl.get("source") or pl.get("document") or "Document")
        page = metadata.get("page") or pl.get("page")
        key = (doc, page)
        if key in seen: continue
        seen.add(key)

        class Obj: pass
        o = Obj()
        o.page_content = pl.get("text", "")
        o.metadata = {
            "source": doc,  # Use 'source' as primary key
            "document": doc,
            "page": page,
            "similarity_score": float(p.score)
        }
        docs.append(o)
        sources.add(doc)
        if isinstance(page, (int, float)): pages.add(int(page))

    # Return all hits sorted by score (highest first)
    docs.sort(key=lambda d: d.metadata.get("similarity_score", 0), reverse=True)
    return (docs, list(sources), sorted(pages))


async def warmup():
    """Optional warmup function to run once at startup."""
    from app.modules.lawfirmchatbot.services.llm import embed_text_async
    from app.modules.lawfirmchatbot.services.vector_store import get_qdrant_client
    try:
        await embed_text_async("warmup")
        client = get_qdrant_client()
        client.search(collection_name=settings.QDRANT_COLLECTION, query_vector=[0.0]*settings.EMBED_MODEL_DIM, limit=1)
    except Exception:
        pass


# === Attachment-Scope-Aware Vector Search ===
from typing import Literal

Scope = Literal["ephemeral_only", "hybrid"]


class VectorSearch:
    """
    Unified vector search interface that respects attachment scope.
    Prevents KB pollution when users reference specific attached documents.
    """
    
    def __init__(self, qdrant_client, cfg):
        """
        Initialize with existing Qdrant client and config.
        
        Args:
            qdrant_client: Existing Qdrant client instance
            cfg: Configuration object with QDRANT_COLLECTION attribute
        """
        self.qdrant = qdrant_client
        self.cfg = cfg
        # Memory guard: do not create duplicate collections
        # (assumes collections already exist or are created elsewhere)
        self.kb_collection = getattr(cfg, "QDRANT_COLLECTION", "law_docs_v1")
        self.ephemeral_collection = get_ephemeral_collection()
    
    async def _search_ephemeral(
        self, 
        query: str, 
        conversation_id: str, 
        limit: int, 
        filter_file_ids: Optional[List[str]] = None
    ) -> List[Dict[str, Any]]:
        """
        Search ephemeral documents for a specific conversation.
        
        Args:
            query: Search query text
            conversation_id: Conversation identifier
            limit: Maximum results to return
            filter_file_ids: Optional list of file IDs to filter on
            
        Returns:
            List of search results with text, source, score, metadata
        """
        try:
            # Generate embedding for query
            vec = await embed_text_async(query)
            
            # Search ephemeral collection
            from qdrant_client.models import Filter, FieldCondition, MatchValue, MatchAny
            
            # Build filter conditions
            conditions = [
                FieldCondition(
                    key="conversation_id",
                    match=MatchValue(value=conversation_id)
                )
            ]
            
            if filter_file_ids:
                conditions.append(
                    FieldCondition(
                        key="file_id",
                        match=MatchAny(any=filter_file_ids)
                    )
                )
            
            search_result = self.qdrant.search(
                collection_name=self.ephemeral_collection,
                query_vector=vec,
                query_filter=Filter(must=conditions),
                limit=limit,
                with_payload=True,
                with_vectors=False,
            )
            
            # Normalize results to standard format
            return normalize_hits(search_result)
            
        except Exception as e:
            logger.error(f"Error searching ephemeral collection: {e}", exc_info=True)
            return []
    
    async def _search_main(self, query: str, limit: int) -> List[Dict[str, Any]]:
        """
        Search main knowledge base collection.
        
        Args:
            query: Search query text
            limit: Maximum results to return
            
        Returns:
            List of search results with text, source, score, metadata
        """
        try:
            # Generate embedding for query
            vec = await embed_text_async(query)
            
            # Search main KB collection
            search_result = self.qdrant.search(
                collection_name=self.kb_collection,
                query_vector=vec,
                limit=limit,
                with_payload=True,
                with_vectors=False,
            )
            
            # Normalize results to standard format
            return normalize_hits(search_result)
            
        except Exception as e:
            logger.error(f"Error searching main KB collection: {e}", exc_info=True)
            return []
    
    async def search(
        self,
        *,
        query: str,
        conversation_id: str,
        scope: Scope,
        limit_ephemeral: int,
        limit_main: int,
        filter_file_ids: Optional[List[str]] = None,
    ) -> List[Dict[str, Any]]:
        """
        Unified search that respects attachment scope.
        
        Args:
            query: Search query text
            conversation_id: Conversation identifier
            scope: "ephemeral_only" or "hybrid"
            limit_ephemeral: Max results from ephemeral collection
            limit_main: Max results from main KB collection
            filter_file_ids: Optional file IDs to filter ephemeral search
            
        Returns:
            List of ranked results. If scope == 'ephemeral_only', only ephemeral hits.
            If 'hybrid', ephemeral hits first, followed by main KB hits.
        """
        logger.info(
            f"[vector-search] scope={scope} "
            f"limit_eph={limit_ephemeral} limit_main={limit_main} "
            f"filter_files={filter_file_ids}"
        )
        
        # Always search ephemeral first (for this conversation)
        eph = await self._search_ephemeral(
            query, 
            conversation_id, 
            limit_ephemeral, 
            filter_file_ids
        )
        
        logger.info(f"[vector-search] Found {len(eph)} ephemeral results")
        
        # If any ephemeral results exist, honor priority and do not query main KB
        if eph:
            logger.info("[vector-search] Ephemeral hits found; skipping main KB lookup")
            return eph

        # If scope is ephemeral_only but no hits yet, still skip main KB
        if scope == "ephemeral_only":
            logger.info("[vector-search] Ephemeral-only scope with no hits; returning empty list")
            return eph

        # Hybrid mode: also search main KB
        main = await self._search_main(query, limit_main)
        logger.info(f"[vector-search] Found {len(main)} main KB results")
        
        # Simple merge: ephemeral first (higher priority), then main KB
        combined = eph + main
        logger.info(f"[vector-search] Returning {len(combined)} total results (hybrid)")
        
        return combined
