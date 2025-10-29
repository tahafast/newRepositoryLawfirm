from typing import List, Optional, Dict
from sqlalchemy.ext.asyncio import AsyncSession
from .db import get_db
from .repo import ensure_conversation, add_message, last_messages
from .redis_buffer import EphemeralBuffer
from .qdrant_chat_memory import upsert_turn, search_memory
from .models import Conversation
import logging
from datetime import datetime
import uuid

logger = logging.getLogger(__name__)

class ChatMemoryManager:
    """Coordinates persistent history, ephemeral buffer, and semantic recall."""
    def __init__(self, embed_text):
        self.buffer = EphemeralBuffer(max_turns=5)
        self.embed = embed_text  # callable: str -> List[float]
        self.qdrant_client = None  # Lazy-initialized

    async def _get_qdrant_client(self):
        """Lazy initialize Qdrant client."""
        if self.qdrant_client is None:
            try:
                import os
                from qdrant_client import QdrantClient
                self.qdrant_client = QdrantClient(
                    url=os.getenv("QDRANT_URL", "http://localhost:6333"),
                    api_key=os.getenv("QDRANT_API_KEY")
                )
            except Exception as e:
                logger.warning(f"[memory-manager] Failed to initialize Qdrant client: {e}")
                return None
        return self.qdrant_client

    async def start_or_get_conversation(self, db: AsyncSession, user_id: str, conversation_id: Optional[str]) -> str:
        """
        Resolve conversation_id with fallback to latest open conversation.
        
        PATCH_1: Stable conversation_id resolution
        - If conversation_id provided and exists → use it
        - Else if missing → find latest conversation for user
        - Else → create new conversation
        """
        if conversation_id:
            # Try to get existing conversation
            conv = await db.get(Conversation, conversation_id)
            if conv:
                logger.info(f"[memory-manager] Resolved existing conversation: {conversation_id}")
                return conv.id
            logger.warning(f"[memory-manager] Conversation {conversation_id} not found, finding latest...")
        
        # Try to find latest conversation for this user
        from sqlalchemy import select
        q = select(Conversation).where(
            Conversation.user_id == user_id
        ).order_by(Conversation.created_at.desc()).limit(1)
        res = await db.execute(q)
        rows = res.all()
        
        if rows:
            conv = rows[0][0]
            logger.info(f"[memory-manager] Using latest conversation for user {user_id}: {conv.id}")
            return conv.id
        
        # No conversation found, create new one
        conv = await ensure_conversation(db, user_id, None)
        logger.info(f"[memory-manager] Created new conversation: {conv.id}")
        return conv.id

    async def append(self, db: AsyncSession, user_id: str, conversation_id: str, role: str, content: str):
        """
        Append message to both SQL storage and Qdrant (best-effort).
        Ensures both user and assistant messages use the same conversation_id.
        """
        # Always write to SQL first (transactional)
        await add_message(db, conversation_id, role, content)
        await db.flush()  # Ensure SQL commit before attempting Qdrant
        
        # Best-effort Qdrant upsert (async fire-and-forget)
        await self.buffer.append(conversation_id, role, content)
        try:
            await upsert_turn(user_id, conversation_id, role, content, self.embed)
            logger.debug(f"[memory-manager] Upserted {role} message to Qdrant for conversation {conversation_id}")
        except Exception as e:
            logger.warning(f"[memory-manager] Qdrant upsert failed (non-fatal): {e}")

    async def atomic_upsert_messages(self, db: AsyncSession, messages: List[Dict[str, str]]):
        """
        PATCH_3: Atomically upsert multiple messages (user + assistant) to ensure
        they share the same conversation_id and are persisted together.
        
        Args:
            db: Database session
            messages: List of {"role": "user"|"assistant", "text": str, "conversation_id": str}
        """
        try:
            # Batch add to SQL (transactional)
            for msg in messages:
                conversation_id = msg.get("conversation_id")
                role = msg.get("role", "user")
                text = msg.get("text", "")
                
                if not conversation_id:
                    logger.warning(f"[atomic-upsert] Skipping message without conversation_id: {msg}")
                    continue
                
                await add_message(db, conversation_id, role, text)
            
            # Flush all to ensure SQL consistency
            await db.flush()
            logger.info(f"[atomic-upsert] Successfully persisted {len(messages)} messages atomically to SQL")
            
            # Best-effort Qdrant upsert (non-blocking)
            for msg in messages:
                try:
                    conversation_id = msg.get("conversation_id")
                    role = msg.get("role")
                    text = msg.get("text")
                    user_id = msg.get("user_id", "anon")
                    
                    if conversation_id and role and text:
                        await upsert_turn(user_id, conversation_id, role, text, self.embed)
                except Exception as e:
                    logger.debug(f"[atomic-upsert] Qdrant upsert failed for message (non-fatal): {e}")
            
            logger.info(f"[atomic-upsert] Completed atomic upsert for {len(messages)} messages")
        
        except Exception as e:
            logger.error(f"[atomic-upsert] CRITICAL: Atomic upsert failed: {e}", exc_info=True)
            raise

    async def get_prompt_messages(self, db: AsyncSession, conversation_id: str, recent_pairs: int = 5) -> List[Dict]:
        """Get recent messages for prompt building, ordered by timestamp."""
        recent = await last_messages(db, conversation_id, limit=recent_pairs*2)
        messages = [{"role": m.role, "content": m.content} for m in recent]
        
        # Ensure consistent ordering by creation time
        messages.sort(key=lambda m: recent[[i for i, r in enumerate(recent) if r.role == m["role"] and r.content == m["content"]][0]].created_at if recent else datetime.utcnow())
        
        return messages

    async def semantic_recall(self, query: str, user_id: str, k: int = 3) -> List[Dict]:
        """Recall semantically similar messages from conversation history."""
        try:
            return await search_memory(query, user_id, k, self.embed)
        except Exception as e:
            logger.warning(f"[memory-manager] Semantic recall failed (non-fatal): {e}")
            return []

