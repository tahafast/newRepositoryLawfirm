from qdrant_client import QdrantClient
from core.config import settings

_qdrant_singleton: QdrantClient | None = None


def get_qdrant_client() -> QdrantClient:
    """
    Return a singleton QdrantClient configured for HTTP with an extended timeout.
    This avoids gRPC DEADLINE_EXCEEDED errors on managed Qdrant deployments.
    """
    global _qdrant_singleton
    if _qdrant_singleton is None:
        _qdrant_singleton = QdrantClient(
            url=settings.QDRANT_URL,
            api_key=settings.QDRANT_API_KEY,
            prefer_grpc=False,
            timeout=30.0,
        )
    return _qdrant_singleton
