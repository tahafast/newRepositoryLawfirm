from __future__ import annotations

import io
import os
from typing import Callable, Dict, List

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile
from starlette.concurrency import run_in_threadpool

from app.modules.lawfirmchatbot.schemas.ephemeral import (
    EphemeralDeleteResponse,
    EphemeralExistsResponse,
    EphemeralUploadResponse,
)
from app.modules.lawfirmchatbot.services.ephemeral_store import (
    QdrantUnavailable,
    backend_health,
    collection_exists,
    delete_collection,
    ephemeral_collection_name,
    upsert_document,
)
from app.modules.lawfirmchatbot.services.llm import embed_text

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
    vectors: List[List[float]] = []
    for chunk in chunks:
        if not chunk.strip():
            continue
        vectors.append(embed_text(chunk))
    return vectors


def get_chunker() -> ChunkerFn:
    return _default_chunker


def get_embedder() -> EmbedderFn:
    return _default_embedder


async def _read_text_from_upload(file: UploadFile) -> str:
    """
    Lightweight extractor supporting txt/pdf/docx without adding hard dependencies.
    Replace via dependency overrides if richer extraction is required.
    """
    filename = (file.filename or "").lower()
    _, ext = os.path.splitext(filename)
    if ext not in ALLOWED_EXTS:
        raise HTTPException(
            status_code=415,
            detail="Unsupported file type. Allowed: .pdf, .doc, .docx, .txt",
        )
    raw = await file.read()

    if filename.endswith(".txt"):
        return raw.decode(errors="ignore")

    if filename.endswith(".pdf"):
        try:
            import pypdf  # type: ignore

            reader = pypdf.PdfReader(io.BytesIO(raw))
            return "\n".join((page.extract_text() or "") for page in reader.pages)
        except Exception:
            return raw.decode(errors="ignore")

    if filename.endswith(".docx"):
        try:
            import docx  # type: ignore

            doc = docx.Document(io.BytesIO(raw))
            return "\n".join(paragraph.text for paragraph in doc.paragraphs)
        except Exception:
            return raw.decode(errors="ignore")

    return raw.decode(errors="ignore")


@router.post("/upload", response_model=EphemeralUploadResponse)
async def upload_ephemeral_document(
    conversation_id: str = Form(...),
    user_id: str = Form(...),  # retained for compatibility, not used here
    file: UploadFile = File(...),
    chunker: ChunkerFn = Depends(get_chunker),
    embedder: EmbedderFn = Depends(get_embedder),
) -> EphemeralUploadResponse:
    """
    Extract → chunk → embed → upsert into a per-chat collection.
    No ranking, merging, or prompting logic lives here.
    """
    if not conversation_id.strip():
        raise HTTPException(status_code=400, detail="conversation_id is required")

    text = await _read_text_from_upload(file)
    if not text.strip():
        raise HTTPException(status_code=400, detail="Could not extract any text from file.")

    # Run blocking upsert work (and any nested event loop usage) off the main event loop.
    try:
        collection, chunks = await run_in_threadpool(
            upsert_document,
            conversation_id=conversation_id,
            text=text,
            chunker=chunker,
            embedder=embedder,
        )
    except QdrantUnavailable as exc:
        raise HTTPException(
            status_code=503,
            detail=f"Vector store unavailable: {exc}",
        ) from exc

    return EphemeralUploadResponse(
        status="ok",
        chunks=chunks,
        collection=collection,
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


@router.get("/ping")
def ephemeral_ping() -> Dict[str, str]:
    """
    Lightweight health endpoint that reports which backend is active.
    """
    return backend_health()
