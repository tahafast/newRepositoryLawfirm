# app/main.py
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from core.config import settings, wire_services, perform_warmup
from core.logging_conf import logging  # side-effect: configures logging
from app.modules.router import router as modules_router

logger = logging.getLogger(__name__)


def create_app():
    app = FastAPI(title="Law Firm Chatbot")
    
    # Add CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.CORS_ALLOWED_ORIGINS,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    
    wire_services(app)   # keeps Qdrant ensure + warmup + singletons
    app.include_router(modules_router)
    
    # (append in app/main.py after include_router)
    for r in app.routes:
        try:
            logging.getLogger("router.map").info("ROUTE %s %s", ",".join(r.methods or []), r.path)
        except Exception:
            pass
    
    return app


app = create_app()


@app.on_event("startup")
async def startup_event():
    """Application startup event handler."""
    logger.info("Starting Law Firm Chatbot API...")
    
    # Optional HTTP wire logging for debugging
    if settings.LOG_QDRANT_HTTP == "1":
        logging.getLogger("httpx").setLevel(logging.INFO)
    
    # Perform async warmup operations
    await perform_warmup(app)
    
    logger.info("Application startup completed successfully")
