from fastapi import APIRouter, UploadFile, File, Depends, HTTPException, Query
from typing import List, Dict, Any
from pydantic import BaseModel
import logging
import uuid

# Import DTOs only
from app.modules.lawfirmchatbot.schema.query import QueryRequest, QueryResponse
from app.modules.lawfirmchatbot.schema.documents import (
    DocumentUploadResponse,
    DocumentUploadBatchResponse,
)
from core.config import get_legacy_services, Services

# Import service functions
from app.modules.lawfirmchatbot.services.document_service import (
    process_document_upload,
    process_document_query,
)
from app.modules.lawfirmchatbot.services.vector_store import list_document_samples

# Import memory services
from app.services.memory.db import get_db
from app.services.memory.models import Conversation, ChatMessage
from app.services.memory.repo import list_conversations, ensure_conversation, last_messages
from sqlalchemy.ext.asyncio import AsyncSession

logger = logging.getLogger(__name__)

v1 = APIRouter(prefix="/api/v1/lawfirm", tags=["Law Firm Chatbot"])
# IMPORTANT: my existing endpoints are registered on `v1`
# Example (keep your real handlers; do NOT change their signatures):
# @v1.post("/upload-document") ...
# @v1.post("/query") ...
router = v1  # optional alias for external imports


async def _resolve_conversation_id(
    conversation_id: str | None,
    user_id: str,
    db: AsyncSession
) -> str:
    """
    PATCH_1: Stable conversation_id resolution with fallback to latest open chat.
    
    Strategy:
    1. If conversation_id provided and exists → use it
    2. If missing → find latest conversation for user
    3. Else → create new conversation
    
    This prevents accidental chat fragmentation from frontend reinit.
    """
    if conversation_id:
        try:
            conv = await db.get(Conversation, conversation_id)
            if conv and conv.user_id == user_id:
                logger.info(f"[PATCH_1] Using existing conversation: {conversation_id}")
                return conversation_id
            else:
                logger.warning(f"[PATCH_1] Conversation {conversation_id} not found or belongs to different user")
        except Exception as e:
            logger.warning(f"[PATCH_1] Failed to fetch conversation {conversation_id}: {e}")
    
    # Try to find latest conversation for this user
    from sqlalchemy import select
    try:
        q = select(Conversation).where(
            Conversation.user_id == user_id
        ).order_by(Conversation.created_at.desc()).limit(1)
        res = await db.execute(q)
        rows = res.all()
        
        if rows:
            conv = rows[0][0]
            logger.info(f"[PATCH_1] Using latest conversation for user {user_id}: {conv.id}")
            return conv.id
    except Exception as e:
        logger.warning(f"[PATCH_1] Failed to find latest conversation: {e}")
    
    # No conversation found, create new one
    try:
        new_conv = await ensure_conversation(db, user_id, None)
        await db.flush()
        logger.info(f"[PATCH_1] Created new conversation: {new_conv.id}")
        return new_conv.id
    except Exception as e:
        logger.error(f"[PATCH_1] Failed to create conversation: {e}")
        # Fallback: generate UUID
        return str(uuid.uuid4())


@v1.get("/docs/indexed/samples")
async def get_indexed_doc_samples(
    document: str = Query(..., description="Exact document name to preview"),
    alias: str | None = Query(None),
    k: int = Query(5, ge=1, le=10),
) -> Dict[str, Any]:
    try:
        samples = list_document_samples(document_name=document, alias=alias, k=k)
        return {"success": True, "document": document, "items": samples}
    except Exception as e:
        return {"success": False, "error": str(e)}


@v1.post(
    "/upload-document",
    response_model=DocumentUploadResponse | DocumentUploadBatchResponse,
)
async def upload_document(
    file: UploadFile | None = File(default=None),
    files: List[UploadFile] | None = File(default=None),
    services: Services = Depends(get_legacy_services)
) -> DocumentUploadResponse | DocumentUploadBatchResponse:
    """Upload and process a document for RAG indexing."""
    uploads: List[UploadFile] = []
    if file is not None:
        uploads.append(file)
    if files:
        uploads.extend(files)

    if not uploads:
        raise HTTPException(status_code=400, detail="No file uploaded")

    results: List[DocumentUploadResponse] = []
    for item in uploads:
        result = await process_document_upload(item, services)
        results.append(result)

    if len(results) == 1:
        return results[0]

    total_chunks = sum(r.chunks for r in results)
    return DocumentUploadBatchResponse(
        message=f"Successfully processed {len(results)} files",
        total_chunks=total_chunks,
        files=results,
    )


@v1.post("/query", response_model=QueryResponse)
async def query_document(
    req: QueryRequest,
    services: Services = Depends(get_legacy_services),
    db: AsyncSession = Depends(get_db)
) -> QueryResponse:
    """
    Process a query against the uploaded documents.
    
    PATCH_1: Stable conversation_id handling - ensures consistent conversation tracking.
    """
    import time
    start_time = time.time()
    
    try:
        # PATCH_1: Resolve conversation_id with fallback logic
        resolved_conv_id = await _resolve_conversation_id(
            conversation_id=req.conversation_id,
            user_id=req.user_id or "anon",
            db=db
        )
        
        # Update request with resolved conversation_id
        req.conversation_id = resolved_conv_id
        
        logger.info(f"[PATCH_1] Query with resolved conversation: {resolved_conv_id}, query_len={len(req.query)}")
        
        # Process the query
        response = await process_document_query(req, services)
        
        # Ensure conversation_id is in metadata for UI coherence
        if response.metadata:
            response.metadata["conversation_id"] = resolved_conv_id
        else:
            response.metadata = {"conversation_id": resolved_conv_id}
        
        elapsed_ms = int((time.time() - start_time) * 1000)
        logger.info(f"[PATCH_1] Query processed successfully in {elapsed_ms}ms with conversation {resolved_conv_id}")
        return response
    
    except Exception as e:
        logger.error(f"Error processing query: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail={"success": False, "answer": str(e), "metadata": {}}
        )


# Conversation management endpoints

class NewConversationRequest(BaseModel):
    user_id: str = "anon"

class ConversationResponse(BaseModel):
    conversation_id: str
    title: str

class ConversationListItem(BaseModel):
    id: str
    title: str
    created_at: str

class MessageResponse(BaseModel):
    role: str
    content: str
    created_at: str

@v1.post("/conversations/new", response_model=ConversationResponse)
async def new_conversation(
    req: NewConversationRequest = NewConversationRequest(),
    db: AsyncSession = Depends(get_db)
) -> ConversationResponse:
    """Create a new conversation."""
    conv = await ensure_conversation(db, req.user_id, None)
    await db.commit()
    return ConversationResponse(conversation_id=conv.id, title=conv.title)

@v1.get("/conversations", response_model=List[ConversationListItem])
async def get_conversations(
    user_id: str = "anon",
    db: AsyncSession = Depends(get_db)
) -> List[ConversationListItem]:
    """List all conversations for a user."""
    rows = await list_conversations(db, user_id, limit=50)
    return [
        ConversationListItem(
            id=r.id,
            title=r.title,
            created_at=r.created_at.isoformat()
        ) for r in rows
    ]

@v1.get("/conversations/{conversation_id}/messages", response_model=List[MessageResponse])
async def get_messages(
    conversation_id: str,
    db: AsyncSession = Depends(get_db)
) -> List[MessageResponse]:
    """Get all messages in a conversation."""
    rows = await last_messages(db, conversation_id, limit=200)
    return [
        MessageResponse(
            role=r.role,
            content=r.content,
            created_at=r.created_at.isoformat()
        ) for r in rows
    ]

@v1.delete("/conversations/{conversation_id}")
async def delete_conversation(
    conversation_id: str,
    db: AsyncSession = Depends(get_db)
) -> Dict[str, Any]:
    """Delete a conversation and all its messages."""
    from sqlalchemy import delete as sql_delete
    
    # Delete all messages in the conversation
    await db.execute(
        sql_delete(ChatMessage).where(ChatMessage.conversation_id == conversation_id)
    )
    
    # Delete the conversation
    await db.execute(
        sql_delete(Conversation).where(Conversation.id == conversation_id)
    )
    
    await db.commit()
    
    return {"success": True, "deleted_id": conversation_id}


