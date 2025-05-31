"""
Utilities package for Chat Service.

This package provides common utility functions, decorators, and helpers
used throughout the Chat Service application.
"""

from src.utils.logger import setup_logging, get_logger
from src.utils.dependencies import (
    get_health_checker,
    get_request_id,
    get_tenant_id,
    get_correlation_id,
)

# Re-export commonly used utilities
__all__ = [
    # Logging utilities
    "setup_logging",
    "get_logger",

    # Dependency injection utilities
    "get_health_checker",
    "get_request_id",
    "get_tenant_id",
    "get_correlation_id",
]

# Package metadata
__version__ = "2.0.0"
__description__ = "Chat Service utility functions and helpers"