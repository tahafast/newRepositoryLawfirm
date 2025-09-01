from fastapi import APIRouter, Query
from app.modules.lawfirmchatbot.services.testingservice import say_hello


v1 = APIRouter(prefix="/hello", tags=["Hello World"])

@v1.get("/")
def hello(name: str = Query("World", description="Your name")):
    """
    A simple hello world endpoint
    """
    return say_hello(name)
