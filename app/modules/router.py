# app/modules/router.py
from fastapi import APIRouter
from app.modules.lawfirmchatbot.api.router import v1 as lawfirm_router

router = APIRouter()
router.include_router(lawfirm_router)
