from typing import Optional, List
from qdrant_client import QdrantClient
from qdrant_client.http import models as qm

import os

from app.modules.lawfirmchatbot.services.qdrant_collections import get_ephemeral_collection

EPHEMERAL_COLLECTION = get_ephemeral_collection()
EMBEDDING_DIM = int(os.getenv("EMBEDDING_DIM", "1536"))


def _pluck_text(payload: dict) -> str:
    for k in ("text", "chunk", "content", "page_text"):
        v = payload.get(k)
        if isinstance(v, str) and v.strip():
            return v.strip()
    return ""


def fetch_ephemeral_priority_context(
    client: QdrantClient,
    conversation_id: str,
    max_points: int = 20,
    max_chars: int = 2000,
) -> str:
    """
    Returns a compact string from the ephemeral collection for this conversation.
    If nothing is found, returns an empty string.
    Designed for the [Priority] section, not full retrieval.
    """
    # Fast filter by conversation_id (you already added an index)
    flt = qm.Filter(should=[
        qm.FieldCondition(key="conversation_id", match=qm.MatchValue(value=conversation_id))
    ])

    # We don't need vectors here; pull payload only.
    try:
        resp = client.scroll(
            collection_name=EPHEMERAL_COLLECTION,
            scroll_filter=flt,
            limit=max_points,
            with_vectors=False,
            with_payload=True,
        )
    except Exception:
        return ""

    points = resp[0] if isinstance(resp, tuple) else resp
    if not points:
        return ""

    snippets: List[str] = []
    seen = set()
    for p in points:
        payload = getattr(p, "payload", {}) or {}
        t = _pluck_text(payload)
        if not t:
            continue
        t = " ".join(t.split())  # compact whitespace
        # de-dupe short repeats
        key = t[:120]
        if key in seen:
            continue
        seen.add(key)
        snippets.append(t)

        # stop when we reach max_chars
        if sum(len(s) for s in snippets) > max_chars:
            break

    if not snippets:
        return ""

    header = "Attached file(s) context is available and MUST be prioritized. Key excerpts:\n"
    bullet = "\n".join(f"• {s[:300]}" for s in snippets)  # keep bullets short
    return f"{header}{bullet}\n"
