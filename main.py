from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from app.modules.router import router
from core.conf import settings
from core.logging_conf import logger
from version import build
import time

app = FastAPI(
    title=settings.PROJECT_NAME,
    version="1.0.0",
    docs_url="/docs" if settings.ENVIRONMENT == "dev" else None,
    redoc_url="/redoc" if settings.ENVIRONMENT == "dev" else None,
    openapi_url="/openapi.json" if settings.ENVIRONMENT == "dev" else None,
)

# Middleware to log every request
@app.middleware("http")
async def log_requests(request: Request, call_next):
    start_time = time.time()
    logger.info(f"➡️ Incoming request: {request.method} {request.url}")
    
    response = await call_next(request)
    
    process_time = (time.time() - start_time) * 1000
    logger.info(f"⬅️ Response status: {response.status_code} | Time: {process_time:.2f}ms")
    
    return response

@app.on_event("startup")
async def startup_event():
    logger.info("🚀 FastAPI application started")

@app.on_event("shutdown")
async def shutdown_event():
    logger.info("🛑 FastAPI application stopped")

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=settings.CORS_EXPOSE_HEADERS,
)

# Register global router
app.include_router(router, prefix=settings.FASTAPI_API_V1_PATH)

# Root endpoint
@app.get("/")
async def root():
    return {
        "message": f"Welcome to {settings.PROJECT_NAME} API!",
        "build": build
    }
