from typing import Dict, Any
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import logging
from app.api.router import router

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Law Firm Chatbot API",
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
async def root() -> Dict[str, str]:
    """Root endpoint with basic API information."""
    logger.info("Root endpoint accessed")
    return {"message": "Welcome to Law Firm Chatbot API", "version": "1.0.0"}

# Health check endpoint
@app.get("/health")
async def health_check() -> Dict[str, str]:
    """Health check endpoint for monitoring."""
    return {"status": "healthy"}

# Include router with prefix
app.include_router(router, prefix="/api/v1")
