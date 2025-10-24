from __future__ import annotations

import os
import time
from typing import Callable, Dict, List, Optional, Sequence, Tuple
from uuid import uuid4

import numpy as np
from qdrant_client import QdrantClient
from qdrant_client.http import models as qmodels

from core.config import settings
from app.modules.lawfirmchatbot.services.llm import embed_text

try:  # Prefer a pre-wired Qdrant client if available
    from app.core.qdrant import qdrant_client  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    qdrant_client = None  # type: ignore

_RAW_EPHEMERAL_COLLECTION = os.getenv("EPHEMERAL_COLLECTION_PREFIX", "ephemeral_docs_shared")
EPHEMERAL_COLLECTION_NAME = (
    _RAW_EPHEMERAL_COLLECTION.strip().rstrip("_") or "ephemeral_docs"
)
VECTOR_SIZE = settings.EMBEDDING_DIM  # keep in sync with embedder output
DISTANCE = settings.VECTOR_DISTANCE or "Cosine"


class QdrantUnavailable(RuntimeError):
    """Raised when Qdrant endpoint cannot be reached or configuration is missing."""


_QDRANT_CLIENT: Optional[QdrantClient] = None
_CHECKED_CLIENT_IDS: set[int] = set()


def _distance_metric() -> qmodels.Distance:
    mapping = {
        "COSINE": qmodels.Distance.COSINE,
        "DOT": qmodels.Distance.DOT,
        "EUCLID": qmodels.Distance.EUCLID,
    }
    key = (DISTANCE or "Cosine").upper()
    try:
        return mapping[key]
    except KeyError as exc:  # pragma: no cover - env misconfiguration
        raise ValueError(
            f"Unsupported VECTOR_DISTANCE '{DISTANCE}'. Expected one of: {', '.join(mapping)}"
        ) from exc


def _client() -> QdrantClient:
    """
    Returns an initialized Qdrant client. Validates connectivity once per client instance.
    """
    global _QDRANT_CLIENT

    if qdrant_client is not None:
        client = qdrant_client
    else:
        if _QDRANT_CLIENT is None:
            url = os.getenv("QDRANT_URL")
            if not url:
                raise QdrantUnavailable("QDRANT_URL is not set")
            api_key = os.getenv("QDRANT_API_KEY") or None
            _QDRANT_CLIENT = QdrantClient(url=url, api_key=api_key, timeout=30.0)
        client = _QDRANT_CLIENT

    client_id = id(client)
    if client_id not in _CHECKED_CLIENT_IDS:
        try:
            client.get_collections()
        except Exception as exc:
            raise QdrantUnavailable(f"Qdrant not reachable: {exc}") from exc
        _CHECKED_CLIENT_IDS.add(client_id)

    return client


def ephemeral_collection_name(_: str | None = None) -> str:
    """Return the shared ephemeral collection name."""
    return EPHEMERAL_COLLECTION_NAME


def _conversation_filter(conversation_id: str) -> qmodels.Filter:
    """Build a filter targeting a specific conversation."""
    return qmodels.Filter(
        must=[
            qmodels.FieldCondition(
                key="conversation_id",
                match=qmodels.MatchValue(value=conversation_id),
            )
        ]
    )


def collection_exists(_: str | None = None) -> bool:
    """Check if the shared collection exists."""
    client = _client()
    name = ephemeral_collection_name()
    return client.collection_exists(name)


def create_collection_if_not_exists(_: str | None = None) -> str:
    """Ensure the shared collection exists."""
    client = _client()
    name = ephemeral_collection_name()
    # Create collection if missing
    if not client.collection_exists(name):
        client.create_collection(
            collection_name=name,
            vectors_config=qmodels.VectorParams(
                size=VECTOR_SIZE,
                distance=_distance_metric(),
            ),
        )
    # Ensure payload indexes always exist, even if collection already present
    # CRITICAL: These indexes make autodiscovery fast and reliable
    for field in ("conversation_id", "file_name", "doc_id", "file_id", "quality"):
        try:
            client.create_payload_index(
                collection_name=name,
                field_name=field,
                field_schema=qmodels.PayloadSchemaType.KEYWORD,
                wait=True,
            )
        except Exception as exc:
            print(f"[INFO] Payload index for {field} already exists or skipped: {exc}")
    return name


def _embedder_batched(chunks: List[str]) -> List[List[float]]:
    """
    Embed chunks in controlled batches with light retry/backoff.
    Keeps order intact and avoids overwhelming embedding backends.
    """
    if not chunks:
        return []

    batch_size = max(1, int(settings.EPHEMERAL_EMBED_BATCH))
    max_retries = max(1, int(settings.EPHEMERAL_MAX_RETRIES))
    base_delay = max(0, int(settings.EPHEMERAL_RETRY_BASE_MS)) / 1000.0
    min_delay = 0.05 if base_delay == 0 else base_delay

    vectors: List[List[float]] = []
    total = len(chunks)
    index = 0

    while index < total:
        batch = chunks[index:index + batch_size]
        delay = min_delay

        for attempt in range(max_retries):
            try:
                vectors.extend([embed_text(chunk) for chunk in batch])
                break
            except Exception:
                if attempt == max_retries - 1:
                    raise
                time.sleep(delay)
                delay *= 1.8  # gentle exponential backoff

        index += batch_size

    return vectors


def embed_chunks_batched(chunks: List[str]) -> List[List[float]]:
    """Public helper for callers that want the hardened batched embedder."""
    return _embedder_batched(chunks)


def delete_collection(conversation_id: str) -> bool:
    client = _client()
    name = ephemeral_collection_name()
    if not client.collection_exists(name):
        return False
    client.delete(
        collection_name=name,
        points_selector=qmodels.FilterSelector(filter=_conversation_filter(conversation_id)),
        wait=True,
    )
    return True


def upsert_document(
    *,
    conversation_id: str,
    text: str,
    file_name: Optional[str],
    chunker: Callable[[str], List[str]],
    embedder: Optional[Callable[[List[str]], List[List[float]]]] = None,
) -> Tuple[str, int, str]:
    collection = create_collection_if_not_exists()
    client = _client()

    raw_chunks = chunker(text)
    chunks = [chunk for chunk in raw_chunks if isinstance(chunk, str) and chunk.strip()]
    if not chunks:
        doc_id = str(uuid4())
        return collection, 0, doc_id

    embed_fn = embedder or embed_chunks_batched
    vectors = np.asarray(embed_fn(chunks), dtype=np.float32)
    if vectors.ndim != 2 or vectors.shape[1] != VECTOR_SIZE:
        raise ValueError(
            f"Embedder returned shape {vectors.shape}, expected (*, {VECTOR_SIZE})"
        )
    if vectors.shape[0] != len(chunks):
        raise ValueError(
            f"Embedder returned {vectors.shape[0]} vectors but {len(chunks)} chunks were produced."
        )

    doc_id = str(uuid4())
    points = []

    for idx, (chunk, vector) in enumerate(zip(chunks, vectors)):
        payload: Dict[str, object] = {
            "text": chunk,
            "conversation_id": conversation_id,  # exact snake_case for filtering
            "doc_id": doc_id,
            "file_id": doc_id,  # ADD: alias for autodiscovery compatibility
            "idx": idx,
            "quality": "ok",  # ADD: quality marker for filtering
        }
        if file_name:
            payload["file_name"] = file_name
            payload["filename"] = file_name  # ADD: both variants for compatibility
        points.append(
            qmodels.PointStruct(
                id=str(uuid4()),
                vector=vector.tolist(),
                payload=payload,
            )
        )

    client.upsert(collection_name=collection, points=points, wait=True)
    return collection, len(chunks), doc_id


def delete_document_by_id(conversation_id: str, doc_id: str) -> None:
    """
    Remove all chunks associated with a single uploaded ephemeral document.
    """
    if not doc_id:
        return

    client = _client()
    name = ephemeral_collection_name()
    if not client.collection_exists(name):
        return

    filter_ = qmodels.Filter(
        must=[
            qmodels.FieldCondition(
                key="conversation_id",
                match=qmodels.MatchValue(value=conversation_id),
            ),
            qmodels.FieldCondition(
                key="doc_id",
                match=qmodels.MatchValue(value=doc_id),
            ),
        ]
    )

    client.delete(
        collection_name=name,
        points_selector=qmodels.FilterSelector(filter=filter_),
        wait=True,
    )


def search(
    *,
    conversation_id: str,
    query_embedding: Sequence[float],
    limit: int = 10,
) -> List[Dict]:
    client = _client()
    name = ephemeral_collection_name()
    if not client.collection_exists(name):
        return []

    results = client.search(
        collection_name=name,
        query_vector=list(query_embedding),
        limit=limit,
        with_payload=True,
        with_vectors=False,
        query_filter=_conversation_filter(conversation_id),
    )

    matches: List[Dict] = []
    for res in results:
        payload = res.payload or {}
        matches.append(
            {
                "id": res.id,
                "score": res.score,
                "text": payload.get("text", ""),
                "payload": payload,
            }
        )
    return matches


def backend_health() -> Dict[str, str]:
    """
    Lightweight diagnostic helper describing the active vector backend.
    """
    info: Dict[str, str] = {"backend": "qdrant"}
    try:
        client = _client()
    except QdrantUnavailable as exc:
        info["status"] = f"error: {exc}"
        return info

    info["status"] = "ok"
    info["url"] = "wired-client" if qdrant_client is not None else os.getenv("QDRANT_URL", "")
    return info
