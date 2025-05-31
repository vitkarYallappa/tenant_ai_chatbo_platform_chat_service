"""
Application constants and enumerations.

This module defines all constant values, enumerations, and
configuration defaults used throughout the Chat Service.
"""

from enum import Enum
from typing import Dict, List, Tuple

# Service Information
SERVICE_NAME = "chat-service"
API_VERSION = "v2"
SERVICE_VERSION = "2.0.0"
SERVICE_DESCRIPTION = "Multi-tenant AI chatbot platform - Chat Service"

# API Configuration
API_PREFIX = f"/api/{API_VERSION}"
DEFAULT_PAGE_SIZE = 20
MAX_PAGE_SIZE = 100
MIN_PAGE_SIZE = 1

# Timeout Configuration (in seconds)
DEFAULT_TIMEOUT_MS = 30000
SHORT_TIMEOUT_MS = 5000
LONG_TIMEOUT_MS = 60000
STARTUP_TIMEOUT = 60
SHUTDOWN_TIMEOUT = 30

# Health Check Configuration
HEALTH_CHECK_INTERVAL = 30  # seconds
HEALTH_CHECK_TIMEOUT = 5    # seconds
MAX_CONSECUTIVE_FAILURES = 3

# Cache TTL Configuration (in seconds)
CACHE_TTL = {
    "session": 3600,        # 1 hour
    "user_profile": 900,    # 15 minutes
    "config": 300,          # 5 minutes
    "response": 1800,       # 30 minutes
    "rate_limit": 60,       # 1 minute
    "health_check": 10,     # 10 seconds
}

# Rate Limiting Configuration
RATE_LIMIT_WINDOWS = {
    "second": 1,
    "minute": 60,
    "hour": 3600,
    "day": 86400,
}

# Message Size Limits (in bytes)
MAX_MESSAGE_SIZE = 4096      # 4KB for text messages
MAX_MEDIA_SIZE = 52428800    # 50MB for media files
MAX_FILE_SIZE = 104857600    # 100MB for document files
MAX_AUDIO_DURATION = 600     # 10 minutes in seconds
MAX_VIDEO_DURATION = 1800    # 30 minutes in seconds

# Supported Channel Types
class ChannelType(str, Enum):
    """Supported communication channels."""
    WEB = "web"
    WH