from typing import Iterable, List
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from langchain.schema import Document
from core.conf import settings


_qdrant: QdrantClient | None = None


def get_qdrant() -> QdrantClient:
    global _qdrant
    if _qdrant is None:
        _qdrant = QdrantClient(host=settings.QDRANT_HOST, port=settings.QDRANT_PORT)
        _ensure_collection(_qdrant)
    return _qdrant


def _ensure_collection(client: QdrantClient) -> None:
    try:
        cols = client.get_collections().collections
        if not any(c.name == settings.QDRANT_COLLECTION for c in cols):
            client.create_collection(
                collection_name=settings.QDRANT_COLLECTION,
                vectors_config=VectorParams(size=1536, distance=Distance.COSINE),
            )
    except Exception:
        raise


def upsert_documents(documents: Iterable[Document]) -> int:
    client = get_qdrant()
    points: List[PointStruct] = []
    for idx, doc in enumerate(documents):
        points.append(
            PointStruct(
                id=idx,
                vector=doc.metadata.get("embedding"),
                payload={"text": doc.page_content, "metadata": doc.metadata},
            )
        )
    if points:
        client.upsert(collection_name=settings.QDRANT_COLLECTION, points=points)
    return len(points)


