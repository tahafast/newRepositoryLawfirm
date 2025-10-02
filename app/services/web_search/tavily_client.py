from __future__ import annotations

"""Tavily web search client (preferred).

Use via search_and_summarize(). If TAVILY_API_KEY is missing, callers should
skip web usage gracefully.
"""

import os
import logging
from dataclasses import dataclass
from typing import List, Tuple

import aiohttp

logger = logging.getLogger(__name__)


@dataclass
class WebSource:
    url: str
    title: str
    snippet: str
    score: float


async def _fetch_json(session: aiohttp.ClientSession, url: str, payload: dict) -> dict:
    async with session.post(url, json=payload, timeout=20) as resp:
        resp.raise_for_status()
        return await resp.json()


async def search_and_summarize(query: str, top_k: int = 3, max_chars: int = 3000) -> Tuple[List[WebSource], str]:
    api_key = os.getenv("TAVILY_API_KEY")
    if not api_key:
        logger.info("Tavily disabled (no API key)")
        return [], ""

    # Tavily single-call research endpoint keeps costs predictable
    url = "https://api.tavily.com/search"
    payload = {
        "api_key": api_key,
        "query": query,
        "include_answer": True,
        "max_results": max(1, min(top_k, 5)),
    }

    try:
        async with aiohttp.ClientSession() as session:
            data = await _fetch_json(session, url, payload)
    except Exception as e:
        logger.error(f"Tavily search failed: {e}")
        return [], ""

    results = []
    for item in (data.get("results") or [])[:top_k]:
        title = (item.get("title") or "").strip()
        content = (item.get("content") or "").strip()
        href = (item.get("url") or "").strip()
        if not href or len(content) < 400 or not title:
            continue
        snippet = content[: max_chars]
        results.append(WebSource(url=href, title=title, snippet=snippet, score=1.0))

    summary = (data.get("answer") or "").strip()
    return results, summary


