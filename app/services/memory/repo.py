from typing import List, Optional
from sqlalchemy import select, func, delete
from sqlalchemy.ext.asyncio import AsyncSession
from .models import Conversation, ChatMessage

async def ensure_conversation(db: AsyncSession, user_id: str, conversation_id: Optional[str]) -> Conversation:
    if conversation_id:
        row = await db.get(Conversation, conversation_id)
        if row:
            return row
    conv = Conversation(user_id=user_id)
    db.add(conv)
    await db.flush()
    return conv

async def add_message(db: AsyncSession, conversation_id: str, role: str, content: str) -> ChatMessage:
    msg = ChatMessage(conversation_id=conversation_id, role=role, content=content)
    db.add(msg)
    await db.flush()
    return msg

async def last_messages(db: AsyncSession, conversation_id: str, limit: int = 10) -> List[ChatMessage]:
    q = select(ChatMessage).where(ChatMessage.conversation_id == conversation_id).order_by(ChatMessage.created_at.desc()).limit(limit)
    res = await db.execute(q)
    return list(reversed([r[0] for r in res.all()]))

async def list_conversations(db: AsyncSession, user_id: str, limit: int = 20) -> List[Conversation]:
    q = select(Conversation).where(Conversation.user_id == user_id).order_by(Conversation.created_at.desc()).limit(limit)
    res = await db.execute(q)
    return [r[0] for r in res.all()]

