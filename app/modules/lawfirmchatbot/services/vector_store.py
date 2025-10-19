from typing import Iterable, List, Dict, Any, Optional
import time
import logging
from qdrant_client import QdrantClient
from qdrant_client import models as qmodels
from qdrant_client.http.models import Distance, VectorParams, PointStruct, HnswConfigDiff, PayloadSchemaType
from langchain_core.documents import Document
from core.config import settings

# Initialize logging and settings
import core.logging  # This sets up logging configuration
logger = logging.getLogger("qdrant.client")

_ALIAS_ENABLED = True  # flipped off if alias ops unsupported/fail


_qdrant_client = None


def get_qdrant_client():
    global _qdrant_client
    if _qdrant_client: return _qdrant_client

    # Initialize Qdrant client with compatible parameters
    client_kwargs = {
        "url": settings.QDRANT_URL,
        "api_key": settings.QDRANT_API_KEY,
        "timeout": settings.QDRANT_TIMEOUT_S,
    }
    
    # Only add prefer_grpc if gRPC is enabled (some versions don't support grpc parameter)
    if settings.QDRANT_USE_GRPC:
        client_kwargs["prefer_grpc"] = True
    
    _qdrant_client = QdrantClient(**client_kwargs)

    # Ensure indexes for payload fields
    try:
        _qdrant_client.create_payload_index(settings.QDRANT_COLLECTION, field_name="document", field_schema=PayloadSchemaType.keyword)
    except Exception: pass
    try:
        _qdrant_client.create_payload_index(settings.QDRANT_COLLECTION, field_name="page", field_schema=PayloadSchemaType.integer)
    except Exception: pass

    return _qdrant_client


def get_qdrant() -> QdrantClient:
    """Legacy function for backward compatibility."""
    return get_qdrant_client()


def _physical_name() -> str:
    return settings.QDRANT_COLLECTION


def _alias_name() -> str:
    return settings.QDRANT_COLLECTION_ALIAS or settings.QDRANT_COLLECTION


def get_runtime_collection_name() -> str:
    """Single stable name used by both ingestion and retrieval."""
    return _alias_name()


def get_vector_name() -> str | None:
    """Get vector name for named vectors, returns None for unnamed vectors."""
    n = (settings.QDRANT_VECTOR_NAME or "").strip()
    return n if n else None


def _ensure_collection(client: QdrantClient) -> None:
    """Legacy internal function - delegates to hardened ensure_collection."""
    ensure_collection(client, dim=1536)


def _retry_with_backoff(func, max_retries: int = 3, base_delay: float = 1.0, timeout: float = 30.0):
    """Execute function with exponential backoff retry logic."""
    for attempt in range(max_retries):
        try:
            start_time = time.time()
            result = func()
            elapsed = time.time() - start_time
            if elapsed > timeout:
                raise TimeoutError(f"Operation took {elapsed:.2f}s, exceeding {timeout}s timeout")
            return result
        except Exception as e:
            if attempt == max_retries - 1:
                logger.error(f"Operation failed after {max_retries} attempts: {str(e)}")
                raise
            delay = base_delay * (2 ** attempt)
            logger.warning(f"Attempt {attempt + 1} failed: {str(e)}. Retrying in {delay:.2f}s")
            time.sleep(delay)


# OPTIONAL: tiny logger wrapper (no signature changes to your exported funcs)
def _log(op_name, fn, **kv):
    t0 = time.time()
    res = fn()
    logger.info("qdrant.%s ms=%d %s", op_name, int((time.time()-t0)*1000), kv)
    return res


def use_alias_name() -> str:
    """Return the collection name to use for operations (alias or physical fallback)."""
    return get_runtime_collection_name()


def ensure_collection(client: QdrantClient, dim: int) -> None:
    """Ensure physical collection exists and optionally set up alias if different from physical."""
    physical = _physical_name()
    alias = _alias_name()

    # Ensure the physical collection exists.
    if not client.collection_exists(physical):
        client.recreate_collection(
            physical,
            vectors_config=VectorParams(size=dim, distance=Distance.COSINE),
        )

    # If alias equals physical, do nothing (prevents warnings on older clients)
    if alias == physical:
        return

    # Otherwise try to map alias -> physical with version-safe fallbacks (do not crash)
    try:
        if hasattr(client, "update_aliases"):
            from qdrant_client.http.models import AliasOperations, CreateAlias
            client.update_aliases(
                change_aliases_operations=[AliasOperations.create_alias(
                    CreateAlias(alias_name=alias, collection_name=physical)
                )],
                timeout=30,
            )
        elif hasattr(client, "create_alias"):
            try:
                client.create_alias(collection_name=physical, alias_name=alias)
            except Exception:
                if hasattr(client, "delete_alias"):
                    client.delete_alias(alias_name=alias)
                    client.create_alias(collection_name=physical, alias_name=alias)
    except Exception:
        # Non-fatal; we'll still use the physical name.
        pass


def _safe_filter(f):
    """Drop must clauses that compare to None/"" to avoid filtering everything out after restart."""
    if not f:
        return None
    try:
        must = [c for c in (f.get("must") or []) if not (
            isinstance(c, dict) and c.get("match") and list(c["match"].values())[0] in (None, "", [])
        )]
        out = dict(f)
        if must:
            out["must"] = must
        else:
            out.pop("must", None)
        return out or None
    except Exception:
        return None


def upsert_embeddings(
    client: QdrantClient, 
    ids, 
    vectors, 
    payloads,
    # Legacy parameters for backward compatibility
    embeddings: Optional[List[List[float]]] = None, 
    texts: Optional[List[str]] = None, 
    metadatas: Optional[List[Dict[str, Any]]] = None
) -> int:
    """Upsert embeddings with retry logic and exponential backoff."""
    # Handle legacy calling pattern
    if embeddings is not None and texts is not None:
        if not embeddings or not texts or len(embeddings) != len(texts):
            raise ValueError("embeddings and texts must be non-empty and same length")
        
        if metadatas and len(metadatas) != len(texts):
            raise ValueError("metadatas must be same length as texts if provided")
        
        if ids and len(ids) != len(texts):
            raise ValueError("ids must be same length as texts if provided")
        
        # Convert legacy format to new format
        ids = ids or list(range(len(texts)))
        vectors = embeddings
        payloads = [{"text": texts[i], "metadata": metadatas[i] if metadatas else {}} for i in range(len(texts))]
    
    def _upsert():
        name = get_runtime_collection_name()
        vname = get_vector_name()
        
        if vname:
            pts = [PointStruct(id=ids[i], vector={vname: vectors[i]}, payload=payloads[i]) for i in range(len(ids))]
        else:
            pts = [PointStruct(id=ids[i], vector=vectors[i], payload=payloads[i]) for i in range(len(ids))]
        
        if pts:
            client.upsert(
                collection_name=name, 
                points=pts,
                wait=True
            )
        return len(pts)
    
    return _retry_with_backoff(_upsert, timeout=30.0)


def search_similar(
    client: QdrantClient, 
    query_vector: List[float], 
    top_k: int = 8, 
    filter_: Optional[Dict] = None,
    # Legacy parameters for backward compatibility
    limit: Optional[int] = None,
    score_threshold: Optional[float] = None,
    mmr: bool = False
):
    """Search for similar vectors with retry logic, timeout handling, and optional MMR."""
    # Handle legacy parameters
    if limit is not None:
        top_k = limit
    
    def _search():
        name = get_runtime_collection_name()
        vname = get_vector_name()
        filt = _safe_filter(filter_)
        
        kwargs = dict(
            collection_name=name, 
            query_vector=query_vector, 
            limit=top_k, 
            with_payload=True, 
            with_vectors=False,
            timeout=4  # 4s timeout for search
        )
        
        if vname:
            kwargs["vector_name"] = vname
        if filt:
            kwargs["query_filter"] = filt
        if score_threshold is not None:
            kwargs["score_threshold"] = score_threshold
        
        # Use MMR for diversity if requested (fetch more, then diversify)
        if mmr:
            kwargs["limit"] = min(top_k * 3, 24)  # Fetch 3x for MMR diversity
            
        hits = client.search(**kwargs)
        
        # Apply simple MMR diversification if requested
        if mmr and len(hits) > top_k:
            hits = _apply_mmr(hits, top_k)
        
        # For backward compatibility, return the old format when called with legacy parameters
        if limit is not None or score_threshold is not None:
            return [
                {
                    "id": result.id,
                    "score": result.score,
                    "text": result.payload.get("text", ""),
                    "metadata": result.payload.get("metadata", {})
                }
                for result in hits
            ]
        
        return hits
    
    return _retry_with_backoff(_search, timeout=4.0)


def _apply_mmr(hits, top_k: int, lambda_param: float = 0.5):
    """Apply Maximal Marginal Relevance to diversify results."""
    if not hits or len(hits) <= top_k:
        return hits
    
    selected = [hits[0]]  # Start with highest scoring
    candidates = hits[1:]
    
    while len(selected) < top_k and candidates:
        best_score = float('-inf')
        best_idx = 0
        
        for i, candidate in enumerate(candidates):
            # Relevance score (normalized)
            relevance = candidate.score
            
            # Simple diversity: penalize if payload text is too similar to selected
            max_similarity = 0.0
            cand_text = (candidate.payload or {}).get("text", "")
            for sel in selected:
                sel_text = (sel.payload or {}).get("text", "")
                # Simple word overlap similarity
                if cand_text and sel_text:
                    cand_words = set(cand_text.lower().split())
                    sel_words = set(sel_text.lower().split())
                    if cand_words and sel_words:
                        overlap = len(cand_words & sel_words) / len(cand_words | sel_words)
                        max_similarity = max(max_similarity, overlap)
            
            # MMR formula: lambda * relevance - (1-lambda) * max_similarity
            mmr_score = lambda_param * relevance - (1 - lambda_param) * max_similarity
            
            if mmr_score > best_score:
                best_score = mmr_score
                best_idx = i
        
        selected.append(candidates.pop(best_idx))
    
    return selected


# Legacy wrapper functions - delegate to new hardened versions
def upsert_documents(documents: Iterable[Document]) -> int:
    """Legacy function - wraps new upsert_embeddings for backward compatibility."""
    client = get_qdrant()
    doc_list = list(documents)
    
    if not doc_list:
        return 0
    
    embeddings = [doc.metadata.get("embedding") for doc in doc_list]
    texts = [doc.page_content for doc in doc_list]
    metadatas = [doc.metadata for doc in doc_list]
    ids = [str(i) for i in range(len(doc_list))]
    
    # Filter out documents without embeddings
    valid_data = [(emb, txt, meta, doc_id) for emb, txt, meta, doc_id in zip(embeddings, texts, metadatas, ids) if emb is not None]
    
    if not valid_data:
        logger.warning("No documents with embeddings found")
        return 0
    
    valid_embeddings, valid_texts, valid_metadatas, valid_ids = zip(*valid_data)
    
    return upsert_embeddings(
        client=client,
        embeddings=list(valid_embeddings),
        texts=list(valid_texts),
        metadatas=list(valid_metadatas),
        ids=list(valid_ids)
    )


def points_count(client: QdrantClient) -> int:
    """Get points count using runtime collection name (alias or physical fallback)."""
    name = get_runtime_collection_name()
    try:
        # Prefer exact count if available
        return client.count(name, count_all=True).count
    except Exception:
        info = client.get_collection(name)
        return getattr(info, "points_count", 0) or 0


def _unpack_scroll_result(res):
    """Support both tuple and ScrollResult outputs from qdrant-client."""
    if isinstance(res, tuple):
        points, next_page = res
    else:
        points = getattr(res, "points", []) or []
        next_page = (
            getattr(res, "next_page_offset", None)
            or getattr(res, "next_point_offset", None)
            or getattr(res, "offset", None)
        )
    return points, next_page


def list_documents_in_alias(
    alias: str | None = None,
    limit: int = 100000,
) -> List[Dict[str, Any]]:
    """
    Aggregate by document/source and return:
    [{document, total_chunks, min_page, max_page, first_seen, last_seen}]
    """
    client = get_qdrant_client()
    collection = alias or settings.QDRANT_COLLECTION_ALIAS or settings.QDRANT_COLLECTION

    all_docs: Dict[str, Dict[str, Any]] = {}
    next_page = None
    scanned = 0
    hard_limit = max(limit, 0) if limit is not None else None

    while True:
        if hard_limit is not None and scanned >= hard_limit:
            break

        batch_limit = 1000
        if hard_limit is not None:
            batch_limit = max(1, min(1000, hard_limit - scanned))

        res = client.scroll(
            collection_name=collection,
            with_vectors=False,
            with_payload=True,
            limit=batch_limit,
            offset=next_page,
        )

        points, next_page = _unpack_scroll_result(res)
        if not points:
            break

        scanned += len(points)

        for p in points:
            pl = p.payload or {}
            doc_name = (
                pl.get("document")
                or pl.get("source")
                or pl.get("title")
                or "Unknown Document"
            )
            page = (
                pl.get("page")
                or pl.get("page_number")
                or pl.get("page_index")
            )
            ts = pl.get("ingested_at") or pl.get("timestamp") or pl.get("created_at")

            bucket = all_docs.setdefault(
                doc_name,
                {
                    "document": doc_name,
                    "total_chunks": 0,
                    "min_page": None,
                    "max_page": None,
                    "first_seen": ts,
                    "last_seen": ts,
                },
            )
            bucket["total_chunks"] += 1

            if isinstance(page, (int, float)):
                page_i = int(page)
                bucket["min_page"] = (
                    page_i
                    if bucket["min_page"] is None
                    else min(bucket["min_page"], page_i)
                )
                bucket["max_page"] = (
                    page_i
                    if bucket["max_page"] is None
                    else max(bucket["max_page"], page_i)
                )

            if ts:
                ts_str = str(ts)
                if not bucket["first_seen"] or ts_str < str(bucket["first_seen"]):
                    bucket["first_seen"] = ts
                if not bucket["last_seen"] or ts_str > str(bucket["last_seen"]):
                    bucket["last_seen"] = ts

        if next_page is None:
            break

    docs = list(all_docs.values())
    docs.sort(
        key=lambda d: (
            str(d.get("last_seen") or ""),
            d["document"],
        ),
        reverse=True,
    )
    return docs


def list_document_samples(
    document_name: str,
    alias: str | None = None,
    k: int = 5,
) -> List[Dict[str, Any]]:
    """
    Fetch up to k sample chunks for a single document (ordered by page ASC).
    Works whether your payload uses 'document' or 'source', and tries multiple
    text keys in order: 'text' -> 'page_content' -> 'chunk' -> 'content'.
    """
    client = get_qdrant_client()
    collection = alias or settings.QDRANT_COLLECTION_ALIAS or settings.QDRANT_COLLECTION

    def _search_by_key(key: str):
        res = client.scroll(
            collection_name=collection,
            with_vectors=False,
            with_payload=True,
            limit=2000,
            scroll_filter=qmodels.Filter(
                must=[
                    qmodels.FieldCondition(
                        key=key,
                        match=qmodels.MatchText(text=document_name),
                    )
                ]
            ),
        )
        return _unpack_scroll_result(res)

    points, _ = _search_by_key("document")
    if not points:
        points, _ = _search_by_key("source")

    rows: List[Dict[str, Any]] = []
    for pt in points:
        pl = pt.payload or {}
        page = pl.get("page") or pl.get("page_number") or pl.get("page_index")
        txt = (
            pl.get("text")
            or pl.get("page_content")
            or pl.get("chunk")
            or pl.get("content")
            or ""
        )
        rows.append(
            {
                "page": int(page) if isinstance(page, (int, float)) else None,
                "text": txt,
                "source": pl.get("document") or pl.get("source") or document_name,
            }
        )

    rows.sort(key=lambda r: (999999 if r["page"] is None else r["page"]))
    return rows[:k]
