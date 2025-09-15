from pydantic import BaseModel, field_validator
from typing import Optional, Dict, Any
import re


class UploadedDocMeta(BaseModel):
    name: str
    type: str
    total_pages: int
    timestamp: float
    chunks_count: int


class DocumentUploadRequest(BaseModel):
    """Request model for document upload (handled via UploadFile in endpoint)."""
    pass


class DocumentUploadResponse(BaseModel):
    """Response model for document upload - matches original contract."""
    message: str
    chunks: int
    filename: str


class DocumentChunk(BaseModel):
    content: str
    metadata: Dict[str, Any]
    similarity_score: Optional[float] = None
    
    @field_validator('content')
    @classmethod
    def clean_content(cls, v: str) -> str:
        """Ensure content is JSON-serializable by cleaning problematic Unicode."""
        if not v:
            return v
        
        try:
            # Remove surrogate pairs and problematic characters
            v = re.sub(r'[\ud800-\udfff]', '', v)  # Remove surrogate pairs
            v = re.sub(r'[\ufffe\uffff]', '', v)   # Remove non-characters
            
            # Replace mathematical symbols that cause issues
            v = re.sub(r'[\U0001d400-\U0001d7ff]', '[MATH]', v)  # Mathematical symbols
            v = re.sub(r'[\U00010000-\U0010ffff]', '[SYMBOL]', v)  # High Unicode planes
            
            # Ensure the text can be encoded to UTF-8
            v = v.encode('utf-8', errors='ignore').decode('utf-8')
            
            return v
        except Exception:
            # Fallback: keep only ASCII characters
            return v.encode('ascii', errors='ignore').decode('ascii')


