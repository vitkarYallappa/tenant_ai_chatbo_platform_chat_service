"""
Configuration package for Chat Service.

This package provides centralized configuration management with
environment-based settings, validation, and constants.
"""

from src.config.settings import get_settings, Settings
from src.config.constants import (
    SERVICE_NAME,
    API_VERSION,
    DEFAULT_PAGE_SIZE,
    MAX_PAGE_SIZE,
    DEFAULT_TIMEOUT_MS,
    HEALTH_CHECK_INTERVAL,
    CACHE_TTL,
    SUPPORTED_CHANNELS,
    MESSAGE_TYPES,
    CONVERSATION_STATUSES,
)

__all__ = [
    "get_settings",
    "Settings",
    "SERVICE_NAME",
    "API_VERSION",
    "DEFAULT_PAGE_SIZE",
    "MAX_PAGE_SIZE",
    "DEFAULT_TIMEOUT_MS",
    "HEALTH_CHECK_INTERVAL",
    "CACHE_TTL",
    "SUPPORTED_CHANNELS",
    "MESSAGE_TYPES",
    "CONVERSATION_STATUSES",
]