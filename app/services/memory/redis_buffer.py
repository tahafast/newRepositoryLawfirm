import os, json
from typing import List, Dict
try:
    import redis.asyncio as redis  # type: ignore
except Exception:
    redis = None

REDIS_URL = os.getenv("REDIS_URL")

class EphemeralBuffer:
    """Stores the last N messages per conversation. Falls back to in-proc dict if Redis missing."""
    def __init__(self, max_turns: int = 5):
        self.max_turns = max_turns
        self._mem: Dict[str, List[Dict]] = {}
        self._r = redis.from_url(REDIS_URL) if REDIS_URL and redis else None

    async def append(self, conversation_id: str, role: str, content: str):
        item = {"role": role, "content": content}
        if self._r:
            key = f"chatbuf:{conversation_id}"
            await self._r.rpush(key, json.dumps(item))
            await self._r.ltrim(key, -2*self.max_turns, -1)  # pairs of (user, assistant)
        else:
            self._mem.setdefault(conversation_id, []).append(item)
            self._mem[conversation_id] = self._mem[conversation_id][-2*self.max_turns:]

    async def recent(self, conversation_id: str) -> List[Dict]:
        if self._r:
            key = f"chatbuf:{conversation_id}"
            vals = await self._r.lrange(key, 0, -1)
            return [json.loads(v) for v in vals]
        return self._mem.get(conversation_id, [])

