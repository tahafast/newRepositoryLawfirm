from __future__ import annotations

"""Lightweight SERP fallback (no API dependency placeholder).

For now, we do nothing and return empty results to avoid breaking flows.
This file exists to keep a stable interface.
"""

import logging
from dataclasses import dataclass
from typing import List, Tuple

logger = logging.getLogger(__name__)


@dataclass
class WebSource:
    url: str
    title: str
    snippet: str
    score: float


async def search_and_summarize(query: str, top_k: int = 3, max_chars: int = 3000) -> Tuple[List[WebSource], str]:
    logger.info("SERP fallback disabled (no implementation)")
    return [], ""


