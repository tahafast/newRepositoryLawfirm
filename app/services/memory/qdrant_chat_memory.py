from typing import List, Dict, Optional
from datetime import datetime
from uuid import uuid4

try:
    from app.modules.lawfirmchatbot.services.vector_store import get_qdrant_client
    from qdrant_client.http.models import PointStruct
except Exception:
    get_qdrant_client = None
    PointStruct = None

COLLECTION = "law_chat_memory"

def _ensure_collection(client):
    from qdrant_client.http.models import Distance, VectorParams
    try:
        client.get_collection(COLLECTION)
    except Exception:
        client.recreate_collection(
            collection_name=COLLECTION,
            vectors_config=VectorParams(size=1536, distance=Distance.COSINE)  # match your embedding dims
        )

async def upsert_turn(user_id: str, conversation_id: str, role: str, text: str, embedding_fn):
    if not get_qdrant_client:
        return
    client = get_qdrant_client()
    _ensure_collection(client)
    vec = await embedding_fn(text)
    point = PointStruct(id=str(uuid4()), vector=vec, payload={
        "user_id": user_id, "conversation_id": conversation_id, "role": role, "text": text, "ts": datetime.utcnow().isoformat()
    })
    client.upsert(collection_name=COLLECTION, points=[point])

async def search_memory(query: str, user_id: str, top_k: int, embed_fn) -> List[Dict]:
    if not get_qdrant_client:
        return []
    client = get_qdrant_client()
    vec = await embed_fn(query)
    res = client.search(collection_name=COLLECTION, query_vector=vec, limit=top_k, query_filter={
        "must": [{"key": "user_id", "match": {"value": user_id}}]
    })
    snippets = []
    for r in res:
        p = r.payload or {}
        snippets.append({"text": p.get("text",""), "role": p.get("role","assistant")})
    return snippets

