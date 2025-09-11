from typing import List
import numpy as np
from openai import AsyncOpenAI
from core.conf import settings


_embed_client: AsyncOpenAI | None = None


def _get_client() -> AsyncOpenAI:
    global _embed_client
    if _embed_client is None:
        _embed_client = AsyncOpenAI(api_key=settings.OPENAI_API_KEY, timeout=60.0)
    return _embed_client


async def embed_text(text: str) -> List[float]:
    client = _get_client()
    response = await client.embeddings.create(input=text, model=settings.EMBEDDING_MODEL)
    return response.data[0].embedding


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


