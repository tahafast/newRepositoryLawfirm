"""
Minimal server test to validate FastAPI startup
"""
import os
from fastapi import FastAPI
from fastapi.responses import JSONResponse

# Set environment variables
os.environ["QDRANT_MODE"] = "embedded"
os.environ["LOG_LEVEL"] = "INFO"

print("Creating minimal FastAPI app...")

app = FastAPI(
    title="Law Firm Chatbot Test",
    version="1.0.0"
)

@app.get("/health")
async def health_check():
    return JSONResponse({
        "status": "healthy", 
        "service": "Law Firm Chatbot",
        "llm_model": "gpt-3.5-turbo",
        "modes": ["docker", "cloud", "embedded"]
    })

@app.get("/")
async def root():
    return JSONResponse({
        "message": "Law Firm Chatbot API",
        "status": "running",
        "version": "1.0.0"
    })

if __name__ == "__main__":
    import uvicorn
    print("Starting minimal server on http://127.0.0.1:8002...")
    uvicorn.run(app, host="127.0.0.1", port=8002, log_level="info")
