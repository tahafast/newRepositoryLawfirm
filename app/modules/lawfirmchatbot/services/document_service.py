"""
Document Service Module
Handles document upload and processing business logic
"""

from typing import Optional, Dict, Any
from tempfile import NamedTemporaryFile
from html import escape
import os
import logging
import re
from fastapi import UploadFile, HTTPException

from app.modules.lawfirmchatbot.schema.documents import DocumentUploadResponse
from app.modules.lawfirmchatbot.schema.query import QueryRequest, QueryResponse
from core.config import Services

logger = logging.getLogger(__name__)

ALLOWED_EXTENSIONS = {'.pdf', '.docx', '.doc', '.txt'}


def render_html_document(doc_text: str) -> str:
    """
    Normalize raw LLM output into the HTML card format expected by the UI.
    Preserves existing HTML where possible and upgrades plain/markdown text.
    """
    if not doc_text:
        body = "<p>[Empty document generated]</p>"
    else:
        text = doc_text.strip()

        # Strip fenced code blocks if the model responded with ```html
        if text.startswith("```"):
            fence = text.splitlines()
            if fence:
                # Drop first line (``` or ```html) and trailing ```
                text = "\n".join(fence[1:])
                if text.endswith("```"):
                    text = text[:-3].rstrip()

        lower = text.lower()
        # If output already includes our wrapper, return as-is
        if "class=\"legal-doc\"" in lower or "class='legal-doc'" in lower:
            return text

        # Treat existing HTML as ready content
        html_indicators = ("<html", "<body", "<section", "<article", "<div")
        if any(ind in lower for ind in html_indicators):
            body = text
        else:
            lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
            if not lines:
                body = "<p>[Empty document generated]</p>"
            else:
                body = "".join(f"<p>{escape(ln)}</p>" for ln in lines)

    return (
        "<div class='legal-doc'>"
        "<div class='doc-body'>"
        f"{body}"
        "</div>"
        "</div>"
    )


def extract_case_info(text: str) -> dict:
    """
    Extract a minimal set of legal document fields from free-form text.
    Returns both the captured data and a list of missing fields.
    """
    fields = {
        "case_type_number": None,
        "case_title": None,
        "representing": None,
        "deponent": None,
        "address": None,
        "capacity": None,
    }

    text = text or ""
    lowered = text.lower()

    case_match = re.search(r"(w\.p\.|suit|application|bail)\s*no\.?\s*[\w/ -]+", lowered)
    title_match = re.search(r"([a-z\s]+v\.?\s+[a-z\s]+)", lowered)
    deponent_match = re.search(r"(mr\.|mrs\.|ms\.)\s*[\w\s]+", text, re.IGNORECASE)
    address_match = re.search(r"(house|flat|street|town|phase|road)[^.,\n]+", text, re.IGNORECASE)
    capacity_match = re.search(r"(authorized representative|petitioner|respondent|applicant|plaintiff|defendant)", lowered)

    if case_match:
        fields["case_type_number"] = case_match.group(0).strip().title()
    if title_match:
        fields["case_title"] = title_match.group(0).strip().title()
    if "petitioner" in lowered:
        fields["representing"] = "Petitioner"
    elif "respondent" in lowered:
        fields["representing"] = "Respondent"
    if deponent_match:
        fields["deponent"] = deponent_match.group(0).strip()
    if address_match:
        fields["address"] = address_match.group(0).strip()
    if capacity_match:
        fields["capacity"] = capacity_match.group(0).strip().title()

    missing = [key for key, value in fields.items() if not value]
    return {"data": fields, "missing": missing}


async def process_document_upload(
    file: UploadFile,
    services: Services
) -> DocumentUploadResponse:
    """
    Process document upload with validation and chunking.
    
    Args:
        file: Uploaded file from FastAPI
        services: Service container with injected dependencies
        
    Returns:
        DocumentUploadResponse: Upload result with metadata
        
    Raises:
        HTTPException: For validation errors or processing failures
    """
    # Validate filename exists
    if not file.filename:
        raise HTTPException(
            status_code=400,
            detail="File must have a filename"
        )
    
    # Validate file extension
    if not any(file.filename.lower().endswith(ext) for ext in ALLOWED_EXTENSIONS):
        raise HTTPException(
            status_code=400,
            detail=f"Invalid file type. Allowed types: {', '.join(sorted(ALLOWED_EXTENSIONS))}"
        )

    temp_file_path: Optional[str] = None
    try:
        suffix = os.path.splitext(file.filename)[1]
        # Stream in chunks to avoid memory spikes on large files
        with NamedTemporaryFile(delete=False, suffix=suffix) as temp_file:
            temp_file_path = temp_file.name
            while True:
                chunk = await file.read(1024 * 1024)
                if not chunk:
                    break
                temp_file.write(chunk)

        # Ensure collection exists before processing (fixes upload crash by passing client)
        from app.modules.lawfirmchatbot.services.vector_store import ensure_collection
        ensure_collection(services.qdrant_client, dim=1536)
        
        # Process document using RAG orchestrator
        from app.modules.lawfirmchatbot.services.rag.rag_orchestrator import get_rag_orchestrator
        rag_orchestrator = get_rag_orchestrator()
        chunks_count = await rag_orchestrator.process_document(temp_file_path, file.filename)

        return DocumentUploadResponse(
            message=f"Successfully processed {file.filename}",
            chunks=chunks_count,
            filename=file.filename
        )
        
    except HTTPException:
        # Re-raise HTTP exceptions as-is
        raise
    except Exception as e:
        logger.error(f"Document upload failed: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e)) from e
    finally:
        # Clean up temporary file
        if temp_file_path and os.path.exists(temp_file_path):
            try:
                os.unlink(temp_file_path)
            except OSError as e:
                logger.warning(f"Failed to delete temp file {temp_file_path}: {e}")


async def process_document_query(
    request: QueryRequest,
    services: Services
) -> QueryResponse:
    """
    Process a query against uploaded documents.
    
    Args:
        request: Query request with user query
        services: Service container with injected dependencies
        
    Returns:
        QueryResponse: Query result with answer and metadata
        
    Raises:
        HTTPException: For processing failures
    """
    try:
        from app.modules.lawfirmchatbot.services.rag.rag_orchestrator import get_rag_orchestrator
        rag_orchestrator = get_rag_orchestrator()
        # New high-level orchestrator path with strategy routing; fallback to legacy if needed
        try:
            result = await rag_orchestrator.answer_query(
                query=request.query,
                conversation_id=request.conversation_id,
                user_id=request.user_id
            )
            from app.modules.lawfirmchatbot.schema.query import QueryResponse
            return QueryResponse(
                success=bool(result.get("success", True)),
                answer=str(result.get("answer", "")),
                answer_markdown=result.get("answer_markdown"),
                metadata=result.get("metadata", {}),
                debug_info=None,
            )
        except AttributeError:
            # Older orchestrator without answer_query
            return await rag_orchestrator.get_answer(request.query)
    except Exception as e:
        logger.error(f"Query processing failed: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e)) from e
