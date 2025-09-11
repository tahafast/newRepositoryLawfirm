from fastapi import APIRouter, HTTPException, UploadFile, File
from typing import Optional, Dict, Any
from tempfile import NamedTemporaryFile
import os
import logging

# Import modular services
from app.services.rag.rag_orchestrator import get_rag_orchestrator
from app.schemas.query import QueryRequest, QueryResponse

logger = logging.getLogger(__name__)
router = APIRouter()
rag_orchestrator = get_rag_orchestrator()

ALLOWED_EXTENSIONS = {'.pdf', '.docx', '.txt'}

@router.post("/upload-document")
async def upload_document(file: UploadFile = File(...)) -> Dict[str, Any]:
    """Upload and process a document for RAG indexing."""
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

        chunks_count = await rag_orchestrator.process_document(temp_file_path, file.filename)

        return {
            "message": f"Successfully processed {file.filename}", 
            "chunks": chunks_count,
            "filename": file.filename
        }
    except Exception as e:
        logger.error(f"Document upload failed: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e)) from e
    finally:
        if temp_file_path and os.path.exists(temp_file_path):
            try:
                os.unlink(temp_file_path)
            except OSError as e:
                logger.warning(f"Failed to delete temp file {temp_file_path}: {e}")

@router.post("/query", response_model=QueryResponse)
async def query_document(req: QueryRequest) -> QueryResponse:
    """Process a query against the uploaded documents."""
    try:
        return await rag_orchestrator.get_answer(req.query)
    except Exception as e:
        logger.error(f"Query processing failed: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e)) from e


