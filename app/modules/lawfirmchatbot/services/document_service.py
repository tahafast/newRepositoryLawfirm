"""
Document Service Module
Handles document upload and processing business logic
"""

from typing import Optional, Dict, Any
from tempfile import NamedTemporaryFile
import os
import logging
from fastapi import UploadFile, HTTPException

from app.modules.lawfirmchatbot.services.rag.rag_orchestrator import get_rag_orchestrator
from app.modules.lawfirmchatbot.schema.documents import DocumentUploadResponse
from app.modules.lawfirmchatbot.schema.query import QueryRequest, QueryResponse
from core.config import Services

logger = logging.getLogger(__name__)

ALLOWED_EXTENSIONS = {'.pdf', '.docx', '.txt'}


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
        rag_orchestrator = get_rag_orchestrator()
        # New high-level orchestrator path with strategy routing; fallback to legacy if needed
        try:
            result = await rag_orchestrator.answer_query(request.query)
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
