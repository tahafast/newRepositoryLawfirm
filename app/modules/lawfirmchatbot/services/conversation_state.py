from __future__ import annotations

import asyncio
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, Optional


@dataclass
class ConversationState:
    summary: str = ""
    pinned_facts: Dict[str, str] = field(default_factory=dict)
    doc_fields: Dict[str, str] = field(default_factory=dict)
    last_document: str = ""
    last_doc_type: Optional[str] = None
    last_mode: Optional[str] = None  # "docgen" | "qa" | "tweak"
    last_response_text: str = ""
    last_references: list[str] = field(default_factory=list)


_CONVERSATION_CACHE: Dict[str, ConversationState] = defaultdict(ConversationState)
_LOCK = asyncio.Lock()


async def get_conversation_state(conversation_id: str) -> ConversationState:
    """
    Return mutable conversation state object for the given conversation id.
    Creates a default entry if none exists.
    """
    async with _LOCK:
        return _CONVERSATION_CACHE[conversation_id]


async def reset_conversation_state(conversation_id: str) -> None:
    async with _LOCK:
        if conversation_id in _CONVERSATION_CACHE:
            _CONVERSATION_CACHE[conversation_id] = ConversationState()


async def update_summary(conversation_id: str, summary: str) -> None:
    async with _LOCK:
        state = _CONVERSATION_CACHE[conversation_id]
        state.summary = summary


async def update_last_document(
    conversation_id: str,
    document_html: str,
    doc_type: Optional[str] = None,
    references: Optional[list[str]] = None,
) -> None:
    async with _LOCK:
        state = _CONVERSATION_CACHE[conversation_id]
        state.last_document = document_html or ""
        if doc_type:
            state.last_doc_type = doc_type
        state.last_mode = "docgen"
        state.last_response_text = document_html or ""
        if references is not None:
            state.last_references = list(references)


async def record_response(
    conversation_id: str,
    text: str,
    mode: Optional[str] = None,
) -> None:
    async with _LOCK:
        state = _CONVERSATION_CACHE[conversation_id]
        state.last_response_text = text or ""
        if mode:
            state.last_mode = mode


async def merge_pinned_facts(conversation_id: str, facts: Dict[str, str]) -> None:
    if not facts:
        return
    async with _LOCK:
        state = _CONVERSATION_CACHE[conversation_id]
        state.pinned_facts.update({k: v for k, v in facts.items() if v})
