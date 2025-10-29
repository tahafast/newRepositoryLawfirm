from __future__ import annotations

import hashlib
import io
import logging
import os
from typing import Callable, Dict, List, Optional

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile, status
from starlette.concurrency import run_in_threadpool

from core.config import settings
from app.modules.lawfirmchatbot.schema.ephemeral import (
    EphemeralDeleteResponse,
    EphemeralExistsResponse,
    EphemeralUploadResponse,
)
from app.modules.lawfirmchatbot.services.ephemeral_store import (
    QdrantUnavailable,
    backend_health,
    collection_exists,
    delete_collection,
    delete_document_by_id,
    embed_chunks_batched,
    ephemeral_collection_name,
    upsert_document,
)
from app.modules.lawfirmchatbot.services.text_cleaner import (
    clean_text,
    normalize_text,
    looks_garbled,
    CleanStats,
)

router = APIRouter(prefix="/api/v1/lawfirm/ephemeral", tags=["Ephemeral Docs"])

# Allowed file extensions (server-side)
ALLOWED_EXTS = {".pdf", ".doc", ".docx", ".txt"}

# Cleaner toggles (feature flagged for deploy safety)
ENABLE_TEXT_CLEANER = os.getenv("ENABLE_TEXT_CLEANER", "true").lower() != "false"
_PIPELINE_ENV_FALLBACK = os.getenv("CLEAN_PIPELINE_VERSION") or os.getenv("TEXT_PIPELINE_VERSSION")
try:
    TEXT_PIPELINE_VERSION = int(os.getenv("TEXT_PIPELINE_VERSION", _PIPELINE_ENV_FALLBACK or "2"))
except ValueError:
    TEXT_PIPELINE_VERSION = 2

# Dependency hook types
ChunkerFn = Callable[[str], List[str]]
EmbedderFn = Callable[[List[str]], List[List[float]]]


def _default_chunker(text: str) -> List[str]:
    """
    Minimal text splitter (~1000 chars with 10% overlap) to keep router self-sufficient.
    Callers can override via dependency injection for richer chunking.
    """
    max_len = 1000
    overlap = 100
    if not text:
        return []
    if len(text) <= max_len:
        return [text]

    chunks: List[str] = []
    start = 0
    length = len(text)
    while start < length:
        end = min(length, start + max_len)
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        if end >= length:
            break
        start = max(0, end - overlap)
        if start >= length:
            break
    return chunks


def _default_embedder(chunks: List[str]) -> List[List[float]]:
    """
    Default embedder built on the project-wide embedding helper.
    Override via dependency injection to batch or reuse cached vectors.
    """
    return embed_chunks_batched(chunks)


def get_chunker() -> ChunkerFn:
    return _default_chunker


def get_embedder() -> EmbedderFn:
    return _default_embedder


async def _read_text_from_upload(file: UploadFile) -> tuple[str, dict]:
    """
    Read upload, extract raw text, and optionally run the cleaner before embedding.
    """
    logger = logging.getLogger(__name__)
    original_filename = file.filename or ""
    ext = os.path.splitext(original_filename.lower())[1]
    if ext not in ALLOWED_EXTS:
        raise HTTPException(
            status_code=415,
            detail="Unsupported file type. Allowed: .pdf, .doc, .docx, .txt",
        )

    raw = await file.read()

    # Extract raw text based on file type
    if ext == ".txt":
        raw_text = raw.decode(errors="ignore")
    elif ext == ".pdf":
        try:
            import pypdf  # type: ignore

            reader = pypdf.PdfReader(io.BytesIO(raw))
            raw_text = "\n".join((page.extract_text() or "") for page in reader.pages)
        except Exception as e:
            logger.warning(f"PDF extraction failed for {original_filename}, falling back to raw decode: {e}")
            raw_text = raw.decode(errors="ignore")
    elif ext == ".docx":
        try:
            import docx  # type: ignore

            doc = docx.Document(io.BytesIO(raw))
            raw_text = "\n".join(paragraph.text for paragraph in doc.paragraphs)
        except Exception as e:
            logger.warning(f"DOCX extraction failed for {original_filename}, falling back to raw decode: {e}")
            raw_text = raw.decode(errors="ignore")
    elif ext == ".doc":
        logger.warning(f"Old .doc file detected ({original_filename}) - attempting binary-safe extraction.")
        raw_text = raw.decode(errors="ignore")
    else:
        raw_text = raw.decode(errors="ignore")

    raw_text = raw_text or ""

    clean_meta: Dict[str, object] = {
        "pipeline_version": TEXT_PIPELINE_VERSION,
        "clean_version": TEXT_PIPELINE_VERSION,
        "kept": True,
    }

    stats: CleanStats | None = None

    if ENABLE_TEXT_CLEANER:
        cleaned_text, stats = clean_text(raw_text, filename=original_filename)
        clean_meta["clean_stats"] = stats.__dict__
        clean_meta["clean_removed_ratio"] = stats.removed_ratio
        clean_meta["clean_alpha_ratio"] = stats.alpha_ratio
        if not stats.kept or not cleaned_text.strip():
            logger.error(
                "[ephemeral-upload] Cleaner rejected %s (alpha=%.2f, removed=%.2f)",
                original_filename,
                stats.alpha_ratio,
                stats.removed_ratio,
            )
            raise HTTPException(
                status_code=422,
                detail=f"Unreadable or corrupted file: {original_filename}",
            )
        cleaned_text = normalize_text(cleaned_text)
        if looks_garbled(cleaned_text):
            logger.error(f"[ephemeral-upload] Garbled text detected in {original_filename} after cleaning")
            raise HTTPException(
                status_code=422,
                detail=f"Unreadable or corrupted file: {original_filename}",
            )
        clean_meta["quality"] = "ok"
    else:
        cleaned_text = normalize_text(raw_text)
        clean_meta["clean_stats"] = None
        clean_meta["clean_removed_ratio"] = None
        clean_meta["clean_alpha_ratio"] = None
        clean_meta["quality"] = "unchecked"

    cleaned_text = (cleaned_text or "").strip()
    if not cleaned_text:
        logger.warning(f"[ephemeral-upload] No readable text extracted from {original_filename}")
        raise HTTPException(
            status_code=422,
            detail=f"Unreadable or corrupted file: {original_filename}",
        )

    return cleaned_text, clean_meta



@router.post("/upload", response_model=EphemeralUploadResponse)
async def upload_ephemeral_document(
    conversation_id: str = Form(...),
    user_id: str = Form(...),  # retained for compatibility, not used here
    mode: str = Form("replace"),
    remove_doc_ids: Optional[List[str]] = Form(None),
    files: List[UploadFile] = File(...),
    chunker: ChunkerFn = Depends(get_chunker),
    embedder: EmbedderFn = Depends(get_embedder),
) -> EphemeralUploadResponse:
    """
    Multi-file upload: extract, chunk, embed, and upsert each file for the conversation.
    """
    if not conversation_id.strip():
        raise HTTPException(status_code=400, detail="conversation_id is required")
    if not files:
        raise HTTPException(status_code=400, detail="At least one file is required.")
    if len(files) > settings.EPHEMERAL_MAX_FILES:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=f"Too many files. Max is {settings.EPHEMERAL_MAX_FILES}.",
        )
    total_bytes = 0
    for f in files:
        size = getattr(f, "size", None)
        if isinstance(size, int):
            total_bytes += size
    if total_bytes and (total_bytes / (1024 * 1024)) > settings.EPHEMERAL_MAX_TOTAL_MB:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=f"Attachments exceed {settings.EPHEMERAL_MAX_TOTAL_MB}MB total.",
        )

    mode_normalized = (mode or "replace").lower()
    if mode_normalized not in {"replace", "append"}:
        mode_normalized = "replace"

    processed_files: List[Dict[str, object]] = []
    collection_name: Optional[str] = None

    doc_ids_to_remove = [doc_id.strip() for doc_id in (remove_doc_ids or []) if isinstance(doc_id, str) and doc_id.strip()]
    for doc_id in doc_ids_to_remove:
        await run_in_threadpool(delete_document_by_id, conversation_id, doc_id)

    for upload in files:
        text_content, clean_meta = await _read_text_from_upload(upload)
        if not text_content.strip():
            continue

        digest_input = f"{conversation_id}{upload.filename}{TEXT_PIPELINE_VERSION}"
        doc_digest = hashlib.sha256(digest_input.encode("utf-8")).hexdigest()
        clean_meta["pipeline_version"] = TEXT_PIPELINE_VERSION
        clean_meta["digest"] = doc_digest

        try:
            collection, chunks, stored_doc_id = await run_in_threadpool(
                upsert_document,
                conversation_id=conversation_id,
                text=text_content,
                file_name=upload.filename,
                chunker=chunker,
                embedder=embedder,
                clean_metadata=clean_meta,
                doc_id=doc_digest,
                pipeline_version=TEXT_PIPELINE_VERSION,
            )
        except QdrantUnavailable as exc:
            raise HTTPException(
                status_code=503,
                detail=f"Vector store unavailable: {exc}",
            ) from exc

        collection_name = collection
        processed_files.append(
            {
                "doc_id": stored_doc_id,
                "file_name": upload.filename or "document",
                "chunks": chunks,
            }
        )

    if not processed_files:
        raise HTTPException(
            status_code=400,
            detail="Could not extract any text from the provided files.",
        )

    # === PATCH 1: Conversation-Scoped Cleanup (max 5 attachments per chat) ===
    try:
        from qdrant_client import QdrantClient
        from qdrant_client.models import Filter, FieldCondition, MatchValue
        from datetime import datetime
        import logging
        
        logger = logging.getLogger(__name__)
        qdrant = QdrantClient(url=os.getenv("QDRANT_URL"), api_key=os.getenv("QDRANT_API_KEY"))
        
        # Fetch all points for this conversation
        collection_name_to_use = collection_name or ephemeral_collection_name(conversation_id)
        existing_result = qdrant.scroll(
            collection_name=collection_name_to_use,
            scroll_filter=Filter(
                must=[FieldCondition(key="conversation_id", match=MatchValue(value=conversation_id))]
            ),
            with_payload=True,
            limit=999
        )
        existing_points = existing_result[0] if isinstance(existing_result, tuple) else existing_result
        
        # Extract unique file_ids
        unique_files = {}
        for p in existing_points:
            if hasattr(p, 'payload') and p.payload:
                file_id = p.payload.get("file_id") or p.payload.get("doc_id")
                if file_id:
                    # Store point with timestamp for sorting
                    ts = p.payload.get("ts", "")
                    if file_id not in unique_files or ts > unique_files[file_id].get("ts", ""):
                        unique_files[file_id] = {"ts": ts, "point_id": p.id}
        
        # If more than 5 unique files, delete oldest ones
        if len(unique_files) > 5:
            sorted_files = sorted(unique_files.items(), key=lambda x: x[1].get("ts", ""))
            files_to_delete = sorted_files[:-5]  # Keep last 5, delete rest
            
            for file_id, info in files_to_delete:
                # Delete all points for this file_id in this conversation
                qdrant.delete(
                    collection_name=collection_name_to_use,
                    points_selector={"filter": Filter(
                        must=[
                            FieldCondition(key="conversation_id", match=MatchValue(value=conversation_id)),
                            FieldCondition(key="file_id", match=MatchValue(value=file_id))
                        ]
                    )}
                )
            logger.info(f"[ephemeral-cleanup] Trimmed to 5 files for conversation {conversation_id}")
    except Exception as cleanup_err:
        logger.warning(f"[ephemeral-cleanup] Failed to enforce max-5 cleanup: {cleanup_err}")
        # Don't fail the upload due to cleanup failure - log but continue

    return EphemeralUploadResponse(
        status="ok",
        collection=collection_name or ephemeral_collection_name(conversation_id),
        files=processed_files,
        mode=mode_normalized,
        tokens_used=None,
    )


@router.get("/{conversation_id}/exists", response_model=EphemeralExistsResponse)
def ephemeral_exists(conversation_id: str) -> EphemeralExistsResponse:
    exists = collection_exists(conversation_id)
    return EphemeralExistsResponse(
        exists=exists,
        collection=ephemeral_collection_name(conversation_id),
    )


@router.delete("/{conversation_id}", response_model=EphemeralDeleteResponse)
def delete_ephemeral_collection(conversation_id: str) -> EphemeralDeleteResponse:
    deleted = delete_collection(conversation_id)
    return EphemeralDeleteResponse(
        status="ok",
        collection=ephemeral_collection_name(conversation_id),
        deleted=deleted,
    )


@router.delete("/{conversation_id}/doc/{doc_id}")
async def delete_ephemeral_document(conversation_id: str, doc_id: str):
    try:
        await run_in_threadpool(delete_document_by_id, conversation_id, doc_id)
    except QdrantUnavailable as exc:
        raise HTTPException(
            status_code=503,
            detail=f"Vector store unavailable: {exc}",
        ) from exc
    return {
        "status": "ok",
        "deleted": {"conversation_id": conversation_id, "doc_id": doc_id},
    }


@router.get("/ping")
def ephemeral_ping() -> Dict[str, str]:
    """
    Lightweight health endpoint that reports which backend is active.
    """
    return backend_health()
