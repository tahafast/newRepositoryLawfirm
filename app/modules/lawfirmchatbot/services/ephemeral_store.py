from __future__ import annotations

import os
from typing import Callable, Dict, List, Optional, Sequence, Tuple
from uuid import uuid4

import numpy as np
from qdrant_client import QdrantClient
from qdrant_client.http import models as qmodels

try:  # Prefer a pre-wired Qdrant client if available
    from app.core.qdrant import qdrant_client  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    qdrant_client = None  # type: ignore

_RAW_EPHEMERAL_COLLECTION = os.getenv("EPHEMERAL_COLLECTION_PREFIX", "ephemeral_docs_shared")
EPHEMERAL_COLLECTION_NAME = (
    _RAW_EPHEMERAL_COLLECTION.strip().rstrip("_") or "ephemeral_docs"
)
VECTOR_SIZE = int(os.getenv("EMBEDDING_DIM", "1536"))  # keep in sync with embedder output
DISTANCE = os.getenv("VECTOR_DISTANCE", "Cosine")


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
    # Ensure payload index always exists, even if collection already present
    try:
        client.create_payload_index(
            collection_name=name,
            field_name="conversation_id",
            field_schema=qmodels.PayloadSchemaType.KEYWORD,
        )
    except Exception as e:
        print(f"[INFO] Payload index for conversation_id already exists or skipped: {e}")
    return name


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
    chunker: Callable[[str], List[str]],
    embedder: Callable[[List[str]], List[List[float]]],
) -> Tuple[str, int]:
    collection = create_collection_if_not_exists()
    client = _client()

    # Remove any existing vectors before inserting replacement chunks.
    client.delete(
        collection_name=collection,
        points_selector=qmodels.FilterSelector(filter=_conversation_filter(conversation_id)),
        wait=True,
    )

    chunks = chunker(text)
    if not chunks:
        return collection, 0

    vectors = np.asarray(embedder(chunks), dtype=np.float32)
    if vectors.ndim != 2 or vectors.shape[1] != VECTOR_SIZE:
        raise ValueError(
            f"Embedder returned shape {vectors.shape}, expected (*, {VECTOR_SIZE})"
        )
    if vectors.shape[0] != len(chunks):
        raise ValueError(
            f"Embedder returned {vectors.shape[0]} vectors but {len(chunks)} chunks were produced."
        )

    points = []

    for idx, (chunk, vector) in enumerate(zip(chunks, vectors)):
        points.append(
            qmodels.PointStruct(
                id=str(uuid4()),
                vector=vector.tolist(),
                payload={
                    "text": chunk,
                    "conversation_id": conversation_id,
                    "idx": idx,
                },
            )
        )

    client.upsert(collection_name=collection, points=points, wait=True)
    return collection, len(chunks)


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
