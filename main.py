from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.modules.router import router
from core.conf import settings
from version import build

app = FastAPI(
    title="My Modular FastAPI App",
    version="1.0.0",
    docs_url="/docs" if settings.ENVIRONMENT == "dev" else None,
    redoc_url="/redoc" if settings.ENVIRONMENT == "dev" else None,
    openapi_url="/openapi.json" if settings.ENVIRONMENT == "dev" else None,
)

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

@app.get("/")
async def root():
    return {"message": "Welcome to Law Firm Chatbot API!", "build": build}
