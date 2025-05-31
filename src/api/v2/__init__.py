"""
API v2 Package
Version 2 of the Chat Service API endpoints.
"""

from fastapi import APIRouter

from .chat_routes import router as chat_router
from .conversation_routes import router as conversation_router
from .session_routes import router as session_router
from .health_routes import router as health_router
from .webhook_routes import router as webhook_router

# Create main v2 router
router = APIRouter(prefix="/api/v2", tags=["v2"])

# Include all sub-routers
router.include_router(chat_router)
router.include_router(conversation_router)
router.include_router(session_router)
router.include_router(health_router)
router.include_router(webhook_router)

__all__ = [
    "router",
    "chat_router",
    "conversation_router",
    "session_router",
    "health_router",
    "webhook_router"
]