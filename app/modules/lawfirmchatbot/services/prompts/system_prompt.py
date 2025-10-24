from textwrap import dedent
from typing import Optional
from qdrant_client import QdrantClient

# A single source of truth for the system prompt.
# We'll inject PRIORITY_CONTEXT dynamically.
SYSTEM_PROMPT_TMPL = dedent("""
You are BRAG AI — a legal research & drafting copilot.

[Priority]
{PRIORITY_CONTEXT}

[Role]
- Be precise, cite retrieved snippets when relevant.
- If the user asks about an attached file, answer from that file first.
- If the user asks to draft/generate a document, produce a structured, clean draft.

[Answering policy]
- Prefer exact passages over speculation.
- If info is not in context, say so and ask for the missing details.
- For QA: keep answers concise, list statutes/sections with short quotes where appropriate.
- For doc generation: use placeholders when details are missing; never invent facts.

[Safety]
- No personal legal advice. Provide general information and options.

""").strip()


def build_system_prompt(priority_context: str) -> str:
    """
    Injects the computed priority_context into [Priority] section.
    """
    return SYSTEM_PROMPT_TMPL.format(PRIORITY_CONTEXT=priority_context or "Always consider law_docs_v1 (main db).")


def generate_priority_context(ephemeral_chunks: list, has_ephemeral: bool = False) -> str:
    """
    Generate priority context based on ephemeral chunks availability.
    
    Args:
        ephemeral_chunks: List of ephemeral chunks for this conversation
        has_ephemeral: Boolean indicating if ephemeral chunks exist
        
    Returns:
        Priority context string to inject into system prompt
    """
    if not has_ephemeral or not ephemeral_chunks:
        return "Always consider law_docs_v1 (main db)."
    
    # Create compact summary of ephemeral chunks
    chunk_summaries = []
    for i, chunk in enumerate(ephemeral_chunks[:3], 1):  # Limit to first 3 for brevity
        # Extract key info from chunk
        text = chunk.get("text", "") or chunk.get("chunk", "") or ""
        if text:
            # Take first 100 chars as summary
            summary = text[:100].strip()
            if len(text) > 100:
                summary += "..."
            chunk_summaries.append(f"{i}. {summary}")
    
    if chunk_summaries:
        summaries_text = "\n".join(chunk_summaries)
        return f"ATTACHED FILES AVAILABLE - prioritize these for answers:\n{summaries_text}\n\nBias toward QA mode unless user explicitly asks to draft/generate/format."
    else:
        return "Always consider law_docs_v1 (main db)."


def generate_priority_context_from_qdrant(
    client: QdrantClient, 
    conversation_id: str, 
    max_points: int = 20, 
    max_chars: int = 2000
) -> str:
    """
    Generate priority context using the ephemeral priority fetcher service.
    
    Args:
        client: QdrantClient instance
        conversation_id: Conversation ID to filter ephemeral chunks
        max_points: Maximum number of points to fetch
        max_chars: Maximum characters for the context
        
    Returns:
        Priority context string to inject into system prompt
    """
    from app.modules.lawfirmchatbot.services.ephemeral_priority import fetch_ephemeral_priority_context
    
    ephemeral_context = fetch_ephemeral_priority_context(
        client=client,
        conversation_id=conversation_id,
        max_points=max_points,
        max_chars=max_chars
    )
    
    if ephemeral_context:
        return ephemeral_context + "Bias toward QA mode unless user explicitly asks to draft/generate/format."
    else:
        return "Always consider law_docs_v1 (main db)."
