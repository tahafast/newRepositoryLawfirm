from fastapi import APIRouter, UploadFile, File, Depends
import logging

# Import DTOs only
from app.modules.lawfirmchatbot.schema.query import QueryRequest, QueryResponse
from app.modules.lawfirmchatbot.schema.documents import DocumentUploadResponse
from core.config import get_legacy_services, Services

# Import service functions
from app.modules.lawfirmchatbot.services.document_service import (
    process_document_upload,
    process_document_query
)

logger = logging.getLogger(__name__)

v1 = APIRouter(prefix="/api/v1/lawfirm", tags=["Law Firm Chatbot"])
# IMPORTANT: my existing endpoints are registered on `v1`
# Example (keep your real handlers; do NOT change their signatures):
# @v1.post("/upload-document") ...
# @v1.post("/query") ...
router = v1  # optional alias for external imports

@v1.post("/upload-document", response_model=DocumentUploadResponse)
async def upload_document(
    file: UploadFile = File(...),
    services: Services = Depends(get_legacy_services)
) -> DocumentUploadResponse:
    """Upload and process a document for RAG indexing."""
    return await process_document_upload(file, services)

@v1.post("/query", response_model=QueryResponse)
async def query_document(
    req: QueryRequest,
    services: Services = Depends(get_legacy_services)
) -> QueryResponse:
    """Process a query against the uploaded documents."""
    logger.info(f"Received query request: {req.query}")
    try:
        response = await process_document_query(req, services)
        logger.info("Query processed successfully")
        return response
    except Exception as e:
        logger.error(f"Error processing query: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail={"success": False, "answer": str(e), "metadata": {}}
        )


