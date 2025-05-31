"""
Chat Service - Multi-tenant AI chatbot platform.

This package provides message processing, conversation management,
and multi-channel communication capabilities for the chatbot platform.
"""

__version__ = "2.0.0"
__author__ = "Development Team"
__email__ = "dev@company.com"
__description__ = "Multi-tenant AI chatbot platform - Chat Service"

# Package metadata
__title__ = "chat-service"
__url__ = "https://github.com/company/chatbot-platform"
__license__ = "MIT"

# Semantic version components
VERSION_INFO = (2, 0, 0)

# Service identification
SERVICE_NAME = "chat-service"
SERVICE_COMPONENT = "message-processing"
API_VERSION = "v2"

# Export commonly used components for convenience
from src.config.settings import get_settings
from src.utils.logger import get_logger

__all__ = [
    "__version__",
    "__author__",
    "__email__",
    "__description__",
    "VERSION_INFO",
    "SERVICE_NAME",
    "SERVICE_COMPONENT",
    "API_VERSION",
    "get_settings",
    "get_logger",
]