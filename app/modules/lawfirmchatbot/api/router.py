from fastapi import APIRouter, UploadFile, File, Depends, HTTPException
from typing import List, Dict, Any
from pydantic import BaseModel
import logging

# Import DTOs only
from app.modules.lawfirmchatbot.schema.query import QueryRequest, QueryResponse
from app.modules.lawfirmchatbot.schema.documents import DocumentUploadResponse
from core.config import get_legacy_services, Services

# Import service functions
from app.modules.lawfirmchatbot.services.document_service import (
    process_document_upload,
    process_document_query
)

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

@v1.post("/upload-document", response_model=DocumentUploadResponse)
async def upload_document(
    file: UploadFile = File(...),
    services: Services = Depends(get_legacy_services)
) -> DocumentUploadResponse:
    """Upload and process a document for RAG indexing."""
    return await process_document_upload(file, services)

@v1.post("/query", response_model=QueryResponse)
async def query_document(
    req: QueryRequest,
    services: Services = Depends(get_legacy_services)
) -> QueryResponse:
    """Process a query against the uploaded documents."""
    import time
    start_time = time.time()
    logger.info(f"Received query request: {req.query[:120]}")
    try:
        response = await process_document_query(req, services)
        elapsed_ms = int((time.time() - start_time) * 1000)
        logger.info(f"Query processed successfully in {elapsed_ms}ms")
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


