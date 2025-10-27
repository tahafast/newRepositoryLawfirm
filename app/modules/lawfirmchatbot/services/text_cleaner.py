"""
Context:
This cleaner runs before embeddings to ensure Qdrant never receives binary junk
or legacy DOC encoding artifacts. It is environment-aware and versioned so
prod and local extractors stay consistent.
"""

from __future__ import annotations

import logging
import os
import re
import unicodedata
from dataclasses import dataclass
from typing import Tuple

logger = logging.getLogger(__name__)

_PIPELINE_VERSION_KEYS = (
    "CLEAN_PIPELINE_VERSION",
    "TEXT_PIPELINE_VERSION",
    "TEXT_PIPELINE_VERSSION",  # legacy typo
)
for _key in _PIPELINE_VERSION_KEYS:
    _val = os.getenv(_key)
    if _val:
        CLEAN_PIPELINE_VERSION = _val
        break
else:
    CLEAN_PIPELINE_VERSION = "lawfirm-cleaner-v1"

# Keep \t, \n, \r but remove other control characters
NON_PRINTABLE = dict.fromkeys(i for i in range(32) if i not in (9, 10, 13))


@dataclass
class CleanStats:
    removed_ratio: float
    alpha_ratio: float
    kept: bool


def clean_text(raw_text: str, filename: str = "") -> Tuple[str, CleanStats]:
    """Return cleaned text + metadata. Never breaks logic if cleaner is off."""
    if not raw_text:
        return "", CleanStats(0.0, 0.0, False)

    alpha_threshold = float(os.getenv("CLEAN_MIN_ALPHA_RATIO", "0.55"))
    min_words = int(os.getenv("CLEAN_MIN_WORDS", "12"))

    text = re.sub(r"[\x00-\x1F\x7F]", " ", raw_text)
    text = re.sub(r"\s+", " ", text).strip()

    alpha_ratio = sum(c.isalpha() for c in text) / max(1, len(text))
    words = len(text.split())
    kept = alpha_ratio >= alpha_threshold and words >= min_words

    removed_ratio = 1 - (len(text) / max(1, len(raw_text)))
    if removed_ratio > 0.25:
        logger.warning(f"[cleaner] {filename}: removed {removed_ratio*100:.1f}% junk")
    if not kept:
        logger.warning(
            f"[cleaner] {filename}: rejected (alpha={alpha_ratio:.2f}, words={words})"
        )

    return (text if kept else ""), CleanStats(removed_ratio, alpha_ratio, kept)


def normalize_text(text: str) -> str:
    """
    Normalize text with proper Unicode handling and smart punctuation replacement.
    """
    if not text:
        return ""

    text = unicodedata.normalize("NFKC", text)
    text = text.translate(NON_PRINTABLE)

    replacements = {
        "\u2010": "-",  # hyphen
        "\u2011": "-",
        "\u2012": "-",
        "\u2013": "-",
        "\u2014": "-",
        "\u2018": "'",
        "\u2019": "'",
        "\u201C": '"',
        "\u201D": '"',
        "\u2026": "...",
    }
    for old, new in replacements.items():
        text = text.replace(old, new)

    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)

    return text.strip()


def looks_garbled(text: str) -> bool:
    """
    Detect if text is garbled/unreadable binary garbage.
    """
    if not text:
        return True

    printable = sum(ch.isprintable() for ch in text)
    printable_ratio = printable / max(1, len(text))

    alpha = sum(ch.isalpha() for ch in text)
    alpha_ratio = alpha / max(1, len(text))

    is_garbled = printable_ratio < 0.90 and alpha_ratio < 0.40

    if is_garbled:
        logger.warning(
            "[text-cleaner] Detected garbled text: "
            f"printable_ratio={printable_ratio:.2f}, alpha_ratio={alpha_ratio:.2f}"
        )

    return is_garbled


def clean_extracted_text(text: str, filename: str = "") -> str:
    """
    Clean extracted text and return safe, normalized content.
    """
    cleaned, stats = clean_text(text, filename)
    if not stats.kept:
        return ""
    return normalize_text(cleaned)


def clean_and_validate_text(
    text: str,
    filename: str = "",
    mark_garbled: bool = True,
) -> tuple[str, dict]:
    """
    Enhanced cleaning with garbled detection and quality metadata.
    """
    metadata = {
        "quality": "ok",
        "clean_version": CLEAN_PIPELINE_VERSION,
    }

    if not text:
        logger.warning(f"Empty text received for file: {filename}")
        metadata["quality"] = "empty"
        metadata["kept"] = False
        return "", metadata

    cleaned_raw, stats = clean_text(text, filename)
    metadata.update(
        {
            "alpha_ratio": stats.alpha_ratio,
            "removed_ratio": stats.removed_ratio,
            "kept": stats.kept,
        }
    )

    if not stats.kept:
        metadata["quality"] = "garbled"
        return "", metadata

    cleaned = normalize_text(cleaned_raw)
    if not cleaned:
        metadata["quality"] = "empty"
        metadata["kept"] = False
        return "", metadata

    if mark_garbled and looks_garbled(cleaned):
        logger.warning(f"Garbled text detected for {filename}: text appears unreadable")
        metadata["quality"] = "garbled"
        metadata["kept"] = False
        return "", metadata

    metadata["cleanup_ratio"] = stats.removed_ratio
    return cleaned, metadata
