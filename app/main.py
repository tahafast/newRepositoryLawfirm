# app/main.py
import logging
import os
from pathlib import Path

import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles

from app.modules.lawfirmchatbot.api.router_indexed_docs import (
    router as indexed_docs_router,
)
from app.modules.router import router as modules_router
from core.config import perform_warmup, settings, wire_services
from core.logging import get_logger

logger = get_logger(__name__)

_UI_CANDIDATES = [
    Path("./ui"),
    Path("./app/templates"),
    Path(__file__).resolve().parents[1] / "frontend",
]
_UI_ROOT = next((p.resolve() for p in _UI_CANDIDATES if p.exists()), None)


def create_app():
    app = FastAPI(title="Law Firm Chatbot")
    wire_services(app)
    app.include_router(modules_router)
    app.include_router(indexed_docs_router, prefix="/api/v1/lawfirm")

    # --- Ephemeral router wiring (no business logic change) ---
    from app.modules.lawfirmchatbot.api import ephemeral_router as _ephemeral_router
    from app.modules.lawfirmchatbot.services._lc_compat import get_recursive_splitter
    from app.modules.lawfirmchatbot.services.llm import embed_text

    def _chunker(text: str) -> list[str]:
        splitter = get_recursive_splitter(
            chunk_size=4000,
            chunk_overlap=480,
            length_function=len,
            add_start_index=False,
            separators=["\n\n", "\n", " ", ""],
        )
        docs = splitter.create_documents([text])
        return [doc.page_content for doc in docs if getattr(doc, "page_content", "").strip()]

    def _embedder(chunks: list[str]) -> list[list[float]]:
        return [embed_text(chunk) for chunk in chunks if chunk.strip()]

    app.dependency_overrides[_ephemeral_router.get_chunker] = lambda: _chunker
    app.dependency_overrides[_ephemeral_router.get_embedder] = lambda: _embedder
    app.include_router(_ephemeral_router.router)

    # --- CORS (allow local static servers & file:// testing) ---
    allow_origins = os.getenv("CORS_ALLOW_ORIGINS", "*")
    origins = (
        ["*"]
        if allow_origins.strip() == "*"
        else [o.strip() for o in allow_origins.split(",") if o.strip()]
    )
    app.add_middleware(
        CORSMiddleware,
        allow_origins=origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
        expose_headers=["*"],
        max_age=86400,
    )

    # --- Optional: serve the frontend from FastAPI to avoid CORS completely ---
    if os.getenv("SERVE_FRONTEND", "1") == "1" and _UI_ROOT:
        app.mount("/ui", StaticFiles(directory=str(_UI_ROOT), html=True), name="frontend")

    for route in app.routes:
        try:
            logging.getLogger("router.map").info(
                "ROUTE %s %s", ",".join(sorted(route.methods or [])), route.path
            )
        except Exception:
            pass

    return app


app = create_app()


@app.api_route("/", methods=["GET", "HEAD"], include_in_schema=False)
async def root():
    if _UI_ROOT:
        return RedirectResponse(url="/ui/chat.html", status_code=302)
    return JSONResponse({"status": "ok", "ui": "not-mounted"})


@app.api_route("/healthz", methods=["GET", "HEAD"], include_in_schema=False)
async def healthz():
    return JSONResponse({"status": "ok"})


@app.on_event("startup")
async def startup_event():
    """Application startup event handler."""
    logger.info("Starting Law Firm Chatbot API...")

    try:
        from app.services.memory.init_db import init_database

        logger.info("Initializing chat memory database...")
        await init_database()
        logger.info("Chat memory database initialized successfully")
    except Exception as exc:
        logger.warning(f"Failed to initialize chat memory database: {exc}")
        logger.warning("Chat memory features may not work properly")

    if settings.LOG_QDRANT_HTTP == "1":
        logging.getLogger("httpx").setLevel(logging.INFO)

    await perform_warmup(app)

    logger.info("Application startup completed successfully")


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("app.main:app", host="0.0.0.0", port=port)
