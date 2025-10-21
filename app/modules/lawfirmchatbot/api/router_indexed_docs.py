import asyncio
from datetime import datetime
from typing import Any, Dict, List, Optional

from fastapi import APIRouter

from core.config import settings
from app.modules.lawfirmchatbot.services.qdrant_service import get_qdrant_client

router = APIRouter()


async def async_scroll_all_points(
    collection_name: str,
    batch_size: int = 200,
    max_points: int = 5000,
) -> List[Any]:
    """
    Iterate through a collection lazily, yielding to the event loop between batches
    so the FastAPI server thread stays responsive.
    """
    client = get_qdrant_client()
    all_points: List[Any] = []
    next_offset: Optional[int] = None
    batch_index = 0

    while True:
        points, next_offset = client.scroll(
            collection_name=collection_name,
            with_payload=True,
            with_vectors=False,
            limit=batch_size,
            offset=next_offset,
        )
        if points:
            all_points.extend(points)

        batch_index += 1
        print(
            f"[async_scroll] batch={batch_index}, total={len(all_points)}, next_offset={next_offset}"
        )

        if not next_offset or len(all_points) >= max_points:
            print(f"[async_scroll] stopping after {len(all_points)} points")
            break

        await asyncio.sleep(0.05)

    return all_points


def aggregate_docs(points: List[Any]) -> List[Dict[str, Any]]:
    """
    Aggregate Qdrant point payloads by document name.
    """
    now_iso = datetime.utcnow().isoformat()
    docs: Dict[str, Dict[str, Any]] = {}

    for point in points:
        payload = getattr(point, "payload", None) or {}
        name = (
            payload.get("document_name")
            or payload.get("file_name")
            or payload.get("filename")
            or payload.get("name")
            or payload.get("source")
            or payload.get("sourcefile")
            or payload.get("metadata", {}).get("source")
            or payload.get("metadata", {}).get("file_name")
            or payload.get("metadata", {}).get("filename")
            or payload.get("metadata", {}).get("document_name")
            or payload.get("document")
            or "unknown"
        )

        page = (
            payload.get("page_number")
            or payload.get("page")
            or payload.get("metadata", {}).get("page_number")
            or 0
        )
        try:
            page_int = int(page)
        except (TypeError, ValueError):
            page_int = 0

        last_seen = (
            payload.get("timestamp")
            or payload.get("last_seen")
            or payload.get("metadata", {}).get("timestamp")
            or now_iso
        )
        last_seen_str = (
            last_seen.isoformat() if hasattr(last_seen, "isoformat") else str(last_seen)
        )

        if name not in docs:
            docs[name] = {
                "document": name,
                "chunks": 1,
                "min_page": page_int,
                "max_page": page_int,
                "last_seen": last_seen_str,
            }
        else:
            entry = docs[name]
            entry["chunks"] += 1
            entry["min_page"] = min(entry["min_page"], page_int)
            entry["max_page"] = max(entry["max_page"], page_int)
            if last_seen_str > entry["last_seen"]:
                entry["last_seen"] = last_seen_str

    print(f"[aggregate_docs] total documents aggregated: {len(docs)}")
    return sorted(
        docs.values(),
        key=lambda entry: (entry["last_seen"], entry["document"]),
        reverse=True,
    )


@router.get("/docs/indexed")
async def get_indexed_documents():
    alias = settings.QDRANT_COLLECTION_ALIAS or "law_docs_current"
    print(f"[docs.indexed] scanning alias={alias}")
    try:
        points = await async_scroll_all_points(alias)
        docs = aggregate_docs(points)
        print(f"[docs.indexed] returning {len(docs)} aggregated docs")
        return docs
    except Exception as exc:
        print(f"[docs.indexed] error: {exc}")
        return []
