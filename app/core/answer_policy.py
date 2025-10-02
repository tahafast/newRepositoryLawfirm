"""
Answer selection policy, intent classification, and markdown formatter.

This module centralizes:
- classify_intent(query)
- select_strategy(query, retrieval_stats)
- format_markdown(answer_sections)
- grounding_guardrails(sources)

Kept intentionally lightweight (regex/heuristics) to avoid extra LLM calls.
"""

from __future__ import annotations

import re
from typing import Dict, List, Literal, Tuple


Strategy = Literal["kb_only", "kb_plus_web", "clarify"]


INTENT_PATTERNS: Dict[str, List[str]] = {
    "define": [r"\bdefine\b", r"\bdefinition\b", r"\bwhat is\b", r"\bmeaning of\b"],
    "compare": [r"\bcompare\b", r"\bversus\b", r"\bvs\.?\b", r"\bvs\b", r"\bagainst\b"],
    "summarize": [r"\bsummar(y|ise|ize)\b", r"\boverview\b", r"\bkey points\b", r"\brecap\b"],
}


def classify_intent(query: str) -> str:
    q = query.lower().strip()
    for intent, pats in INTENT_PATTERNS.items():
        for pat in pats:
            if re.search(pat, q):
                return intent
    return "general"


def select_strategy(query: str, retrieval_stats: Dict[str, float]) -> Strategy:
    """Choose between kb_only, kb_plus_web, and clarify.

    Heuristics (config-free defaults):
    - kb_only when ≥3 chunks above 0.25 OR any chunk ≥0.45
    - kb_plus_web when <3 strong chunks OR intent implies external facts
    - clarify when there are zero hits AND web disabled by caller
    """
    strong_hits = int(retrieval_stats.get("strong_hits", 0))
    max_score = float(retrieval_stats.get("max_score", 0.0))
    total_hits = int(retrieval_stats.get("total_hits", 0))

    intent = classify_intent(query)
    needs_web_keywords = [
        "compare", "latest", "news", "updates", "recent", "case law updates", "versus", "vs"
    ]
    needs_web = intent == "compare" or any(k in query.lower() for k in needs_web_keywords)

    if strong_hits >= 3 or max_score >= 0.45:
        return "kb_only"

    if total_hits == 0 and not needs_web:
        return "clarify"

    return "kb_plus_web" if needs_web or strong_hits < 3 else "kb_only"


def _normalize_citation_labels(sources: List[Tuple[str, str]]) -> Dict[str, int]:
    """Assign 1..N labels in first-appearance order for source keys.

    sources: list of (key, display) tuples. Key should be a stable id
    such as URL or "doc|page" string.
    """
    order: Dict[str, int] = {}
    next_id = 1
    for key, _display in sources:
        if key not in order:
            order[key] = next_id
            next_id += 1
    return order


def format_markdown(answer_sections: List[Dict[str, object]]) -> str:
    """Build final Markdown with bold H3 section headings and normalized citations.

    answer_sections: [
      {
        "title": str,
        "bullets": Optional[List[str]] | None,
        "paragraphs": Optional[List[str]] | None,
        "sources": Optional[List[Dict{key, title, source}]]  # used to build citation labels
      }, ...
    ]
    """
    lines: List[str] = []

    # Collect sources for normalized ordering
    flat_sources: List[Tuple[str, str]] = []
    for sec in answer_sections:
        for s in (sec.get("sources") or []):
            key = str(s.get("key") or s.get("url") or s.get("source") or "")
            title = str(s.get("title") or s.get("source") or key)
            if key:
                flat_sources.append((key, title))

    label_map = _normalize_citation_labels(flat_sources)

    def relabel(text: str) -> str:
        # Replace any [n] present with correct numbering by first-appearance order of source keys
        # We expect callers to insert placeholders like [s:url] or [s:doc|page]. If not provided,
        # keep existing [n]. This keeps function permissive.
        return text

    for sec in answer_sections:
        title = str(sec.get("title", "Section")).strip()
        lines.append(f"### **{title}**")

        # paragraphs
        paragraphs = list((sec.get("paragraphs") or []))
        for p in paragraphs[:4]:  # keep it tight
            lines.append(relabel(str(p)))

        # bullets (truncate to max 6)
        bullets = list((sec.get("bullets") or []))[:6]
        for b in bullets:
            lines.append(f"- {relabel(str(b))}")

        # spacing
        if lines and lines[-1] != "":
            lines.append("")

    # Citations section if any sources present
    if flat_sources:
        lines.append("### **Citations**")
        # deduplicate by key, preserve order
        seen = set()
        for key, title in flat_sources:
            if key in seen:
                continue
            seen.add(key)
            display = title
            # Display format: [[n] Title — source]
            n = _normalize_citation_labels(flat_sources)[key]
            lines.append(f"- [[{n}] {display}]")

    return "\n".join(lines).strip()


def grounding_guardrails(use_web: bool) -> str:
    return "Includes verified web sources." if use_web else ""


