from typing import Dict, Sequence, Any


COMMON_DIRECTIVE = (
    " Always output the finished document as filing-ready HTML (use <div>, <p>, <ol>, <table> as needed)"
    " without markdown code fences or meta commentary. Include appropriate headings, verification/attestation,"
    " and signature blocks when customary."
)

DOCGEN_PROMPTS: Dict[str, str] = {
    "affidavit": (
        "You are to draft a formal affidavit compliant with Pakistani courts. "
        "Use verified, declarative language, include verification and attestation blocks, "
        "and mirror the numbering/style from the provided references."
    ),
    "synopsis": (
        "You are to prepare a concise case synopsis summarizing key facts, questions presented, "
        "and relief sought. Highlight procedural posture and supporting authorities."
    ),
    "rejoinder": (
        "You are to write a rejoinder replying to the opposite party's affidavit. "
        "Address each contested allegation, rebut with clarity, and restate the relief requested."
    ),
    "legal_notice": (
        "You are to draft a formal legal notice to be served upon the respondent. "
        "State background facts, legal breaches, the demand being made, and a clear compliance deadline."
    ),
    "general": (
        "You are to prepare the exact document type requested by the user following Pakistani legal practice. "
        "Maintain professional tone, ensure headings, body, and closing sections match the intent."
    ),
}


def get_docgen_prompt(doc_type: str) -> str:
    key = (doc_type or "").lower()
    base_prompt = DOCGEN_PROMPTS.get(key, DOCGEN_PROMPTS["general"])
    return base_prompt + COMMON_DIRECTIVE


def build_docgen_prompt(
    user_query: str,
    answers: Dict[str, str],
    context: Sequence[Any],
) -> str:
    """
    Prepare the user prompt for doc generation with explicit references.
    """
    details = "\n".join(
        f"- {key}: {value}"
        for key, value in (answers or {}).items()
        if value
    ) or "- user_request: " + user_query.strip()

    context_snippets = []
    for chunk in list(context or [])[:3]:
        text = getattr(chunk, "page_content", None) or ""
        text = text.strip()
        if text:
            context_snippets.append(text)

    refs = "\n\n---\n\n".join(context_snippets) if context_snippets else "[No retrieved context available]"

    return (
        f'You are drafting based on this user intent: "{user_query}"\n\n'
        f"DETAILS:\n{details}\n\n"
        "REFERENCE DOCUMENTS (use their structure, tone, and style as guidance):\n"
        f"{refs}\n\n"
        "If any fact or field is missing, insert a clear placeholder like [PLACEHOLDER: CASE_TITLE] and keep the structure intact.\n"
        "Do not fabricate names, dates, or case numbers. Only omit placeholders if the user explicitly said they do not want them."
    )
