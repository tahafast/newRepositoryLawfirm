import re
from typing import Literal

Intent = Literal["qa", "docgen"]

_DOCGEN_TRIGGERS = re.compile(
    r"\b(draft|compose|generate|prepare|write|format|create)\b", re.I
)
_ATTACHMENT_REFERENCES = re.compile(
    r"\b(attached|attachment|this\s+document|the\s+document|this\s+file|the\s+file|above\s+file)\b",
    re.I,
)

def decide_intent(message: str, has_ephemeral: bool) -> Intent:
    """
    Very small biasing rule:
      - If the user references the attached file and we HAVE ephemeral chunks,
        and they did NOT use clear doc-generation verbs -> go QA.
      - Otherwise fall back to docgen triggers, else QA by default.
    """
    msg = (message or "").strip()
    references_attachment = bool(_ATTACHMENT_REFERENCES.search(msg))

    if has_ephemeral and references_attachment and not _DOCGEN_TRIGGERS.search(msg):
        return "qa"

    if _DOCGEN_TRIGGERS.search(msg):
        return "docgen"

    return "qa"
