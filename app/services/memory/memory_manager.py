from typing import List, Optional, Dict
from sqlalchemy.ext.asyncio import AsyncSession
from .db import get_db
from .repo import ensure_conversation, add_message, last_messages
from .redis_buffer import EphemeralBuffer
from .qdrant_chat_memory import upsert_turn, search_memory

class ChatMemoryManager:
    """Coordinates persistent history, ephemeral buffer, and semantic recall."""
    def __init__(self, embed_text):
        self.buffer = EphemeralBuffer(max_turns=5)
        self.embed = embed_text  # callable: str -> List[float]

    async def start_or_get_conversation(self, db: AsyncSession, user_id: str, conversation_id: Optional[str]) -> str:
        conv = await ensure_conversation(db, user_id, conversation_id)
        return conv.id

    async def append(self, db: AsyncSession, user_id: str, conversation_id: str, role: str, content: str):
        await add_message(db, conversation_id, role, content)
        await self.buffer.append(conversation_id, role, content)
        # best-effort Qdrant upsert (async fire-and-forget; ignore failures)
        try:
            await upsert_turn(user_id, conversation_id, role, content, self.embed)
        except Exception:
            pass

    async def get_prompt_messages(self, db: AsyncSession, conversation_id: str, recent_pairs: int = 5) -> List[Dict]:
        # persistent last messages as fallback
        recent = await last_messages(db, conversation_id, limit=recent_pairs*2)
        return [{"role": m.role, "content": m.content} for m in recent]

    async def semantic_recall(self, query: str, user_id: str, k: int = 3) -> List[Dict]:
        try:
            return await search_memory(query, user_id, k, self.embed)
        except Exception:
            return []

