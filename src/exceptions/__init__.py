"""
Custom exceptions package for Chat Service.

This package provides custom exception classes and error handling
utilities for the Chat Service application.
"""

from src.exceptions.base_exceptions import (
    ChatServiceException,
    ValidationError,
    AuthenticationError,
    AuthorizationError,
    NotFoundError,
    RateLimitError,
    InternalServerError,
    ExternalServiceError,
    TimeoutError,
    ConfigurationError,
    setup_exception_handlers,
)

# Re-export all custom exceptions
__all__ = [
    # Base exception classes
    "ChatServiceException",
    "ValidationError",
    "AuthenticationError",
    "AuthorizationError",
    "NotFoundError",
    "RateLimitError",
    "InternalServerError",
    "ExternalServiceError",
    "TimeoutError",
    "ConfigurationError",

    # Exception handling setup
    "setup_exception_handlers",
]

# Package metadata
__version__ = "2.0.0"
__description__ = "Chat Service custom exceptions and error handling"
