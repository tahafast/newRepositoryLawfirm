from __future__ import annotations

import io
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

router = APIRouter(prefix="/api/v1/lawfirm/ephemeral", tags=["Ephemeral Docs"])

# Allowed file extensions (server-side)
ALLOWED_EXTS = {".pdf", ".doc", ".docx", ".txt"}

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


async def _read_text_from_upload(file: UploadFile) -> str:
    """
    Lightweight extractor supporting txt/pdf/docx without adding hard dependencies.
    Replace via dependency overrides if richer extraction is required.
    Includes text cleaning to remove binary garbage and unreadable characters.
    """
    from app.modules.lawfirmchatbot.services.text_cleaner import clean_and_validate_text
    import logging
    
    logger = logging.getLogger(__name__)
    filename = (file.filename or "").lower()
    _, ext = os.path.splitext(filename)
    if ext not in ALLOWED_EXTS:
        raise HTTPException(
            status_code=415,
            detail="Unsupported file type. Allowed: .pdf, .doc, .docx, .txt",
        )
    raw = await file.read()

    # Extract raw text based on file type
    raw_text = ""
    
    if filename.endswith(".txt"):
        raw_text = raw.decode(errors="ignore")

    elif filename.endswith(".pdf"):
        try:
            import pypdf  # type: ignore

            reader = pypdf.PdfReader(io.BytesIO(raw))
            raw_text = "\n".join((page.extract_text() or "") for page in reader.pages)
        except Exception as e:
            logger.warning(f"PDF extraction failed for {filename}, falling back to raw decode: {e}")
            raw_text = raw.decode(errors="ignore")

    elif filename.endswith(".docx"):
        try:
            import docx  # type: ignore

            doc = docx.Document(io.BytesIO(raw))
            raw_text = "\n".join(paragraph.text for paragraph in doc.paragraphs)
        except Exception as e:
            logger.warning(f"DOCX extraction failed for {filename}, falling back to raw decode: {e}")
            raw_text = raw.decode(errors="ignore")
    
    elif filename.endswith(".doc") and not filename.endswith(".docx"):
        logger.warning(f"Old .doc file detected ({filename}) — attempting binary-safe extraction.")
        raw_text = raw.decode(errors="ignore")
    
    else:
        raw_text = raw.decode(errors="ignore")

    # Clean the extracted text to remove binary garbage and detect quality
    cleaned_text, quality_metadata = clean_and_validate_text(raw_text, filename, mark_garbled=True)
    
    # Log quality issues
    if quality_metadata.get("quality") == "garbled":
        logger.error(f"[ephemeral-upload] Garbled text detected in {filename} - embeddings may be poor quality")
    elif quality_metadata.get("quality") == "empty":
        logger.warning(f"[ephemeral-upload] Empty text extracted from {filename}")
    
    return cleaned_text


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
        text_content = await _read_text_from_upload(upload)
        if not text_content.strip():
            continue

        try:
            collection, chunks, doc_id = await run_in_threadpool(
                upsert_document,
                conversation_id=conversation_id,
                text=text_content,
                file_name=upload.filename,
                chunker=chunker,
                embedder=embedder,
            )
        except QdrantUnavailable as exc:
            raise HTTPException(
                status_code=503,
                detail=f"Vector store unavailable: {exc}",
            ) from exc

        collection_name = collection
        processed_files.append(
            {
                "doc_id": doc_id,
                "file_name": upload.filename or "document",
                "chunks": chunks,
            }
        )

    if not processed_files:
        raise HTTPException(
            status_code=400,
            detail="Could not extract any text from the provided files.",
        )

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
