from fastapi import APIRouter

# Import all routers
from app.modules.lawfirmchatbot.api.testing import v1 as testing_router

router = APIRouter()

# Register routers
router.include_router(testing_router)
