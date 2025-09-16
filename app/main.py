# app/main.py
import os
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from core.config import settings, wire_services, perform_warmup
from core.logging_conf import logging  # side-effect: configures logging
from app.modules.router import router as modules_router

logger = logging.getLogger(__name__)


def create_app():
    app = FastAPI(title="Law Firm Chatbot")
    wire_services(app)
    app.include_router(modules_router)

    # --- CORS (allow local static servers & file:// testing) ---
    allow_origins = os.getenv("CORS_ALLOW_ORIGINS", "*")
    origins = ["*"] if allow_origins.strip() == "*" else [o.strip() for o in allow_origins.split(",") if o.strip()]
    app.add_middleware(
        CORSMiddleware,
        allow_origins=origins,          # e.g. ["http://127.0.0.1:5500","http://localhost:5173","*"]
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
        expose_headers=["*"],
        max_age=86400,
    )

    # --- Optional: serve the frontend from FastAPI to avoid CORS completely ---
    # Set SERVE_FRONTEND=1 (default) to mount /ui -> ./frontend
    if os.getenv("SERVE_FRONTEND", "1") == "1":
        # mounting at /ui ensures /api/* keeps working
        app.mount("/ui", StaticFiles(directory="frontend", html=True), name="frontend")
    
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
