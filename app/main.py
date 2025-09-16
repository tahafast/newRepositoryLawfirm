# app/main.py
import os
import logging
from pathlib import Path
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import RedirectResponse, PlainTextResponse
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
        FRONTEND_DIR = Path(__file__).resolve().parents[1] / "frontend"
        if FRONTEND_DIR.exists():
            app.mount("/ui", StaticFiles(directory=str(FRONTEND_DIR), html=True), name="frontend")
    
    # (append in app/main.py after include_router)
    for r in app.routes:
        try:
            logging.getLogger("router.map").info("ROUTE %s %s", ",".join(sorted(r.methods or [])), r.path)
        except Exception:
            pass
    
    return app


app = create_app()


@app.middleware("http")
async def _no_cache_ui(request, call_next):
    resp = await call_next(request)
    p = request.url.path
    if p.startswith("/ui/config") or p.endswith("/chat.html") or p.endswith("/ingest.html"):
        resp.headers["Cache-Control"] = "no-store"
        resp.headers["Pragma"] = "no-cache"
        resp.headers["Expires"] = "0"
    return resp


@app.api_route("/", methods=["GET", "HEAD"], include_in_schema=False)
async def root():
    # Redirect root to static UI if present, otherwise return a tiny OK
    FRONTEND_DIR = Path(__file__).resolve().parents[1] / "frontend"
    if FRONTEND_DIR.exists():
        return RedirectResponse(url="/ui/")
    return PlainTextResponse("ok", status_code=200)


@app.api_route("/healthz", methods=["GET", "HEAD"], include_in_schema=False)
async def healthz():
    return PlainTextResponse("ok", status_code=200)


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
