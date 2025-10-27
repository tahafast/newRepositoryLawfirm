import re
from typing import Literal

Intent = Literal["qa", "docgen"]

_ATTACHMENT_REFERENCES = re.compile(
    r"\b(attached|attachment|this\s+document|the\s+document|this\s+file|the\s+file|above\s+file)\b",
    re.I,
)

DOCGEN_VERBS = {
    "draft",
    "prepare",
    "generate",
    "write",
    "compose",
    "create",
    "make",
    "produce",
}

FORMAT_REQUEST_VERBS = {"give", "provide", "share", "supply", "send", "offer", "show"}

FORMAT_WORDS = {
    "template",
    "format",
    "skeleton",
    "sample",
    "outline",
    "proforma",
    "framework",
}

LEGAL_DOC_NOUNS = {
    "plaint",
    "petition",
    "notice",
    "legal notice",
    "application",
    "affidavit",
    "suit",
    "reply",
    "rejoinder",
    "writ",
    "complaint",
    "agreement",
    "contract",
    "bail application",
}

INTERROGATIVE_OPENERS = {
    "what",
    "who",
    "when",
    "where",
    "why",
    "how",
    "which",
    "does",
    "do",
    "did",
    "is",
    "are",
    "can",
    "could",
    "should",
    "may",
}


def _contains_any(text: str, terms) -> bool:
    return any(term in text for term in terms)


def decide_intent(message: str, has_ephemeral: bool) -> Intent:
    """
    Conservative docgen gate:
      - Interrogatives (who/what/when/...) or queries ending with '?' default to QA
        unless they explicitly ask to draft/prepare/create something.
      - Docgen requires a drafting verb or an explicit request to provide a template/format.
      - Legal document nouns alone are insufficient for docgen when phrased as questions.
      - Attachment mentions still bias to QA unless a docgen cue is present.
    """
    msg = (message or "").strip().lower()
    if not msg:
        return "qa"

    words = msg.split()
    first_word = words[0] if words else ""
    is_question = msg.endswith("?") or first_word in INTERROGATIVE_OPENERS

    has_docgen_verb = _contains_any(msg, DOCGEN_VERBS)
    asks_for_format = _contains_any(msg, FORMAT_WORDS)
    mentions_doc_noun = _contains_any(msg, LEGAL_DOC_NOUNS)
    template_with_request = asks_for_format and (
        has_docgen_verb or _contains_any(msg, FORMAT_REQUEST_VERBS)
    )
    strong_docgen_signal = has_docgen_verb or template_with_request

    # Interrogatives default to QA unless the user still clearly asks us to draft.
    if is_question and not strong_docgen_signal:
        return "qa"

    references_attachment = bool(_ATTACHMENT_REFERENCES.search(msg))
    if has_ephemeral and references_attachment and not strong_docgen_signal:
        return "qa"

    if strong_docgen_signal or (asks_for_format and mentions_doc_noun):
        return "docgen"

    return "qa"
