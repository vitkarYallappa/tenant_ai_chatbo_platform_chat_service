"""
Base exception classes and error handling for Chat Service.

This module provides the foundation for all custom exceptions
and centralized error handling throughout the application.
"""

import traceback
from typing import Any, Dict, Optional, List, Union
from datetime import datetime

from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
from starlette.exceptions import HTTPException as StarletteHTTPException
import structlog

from src.config.constants import ERROR_MESSAGES, ErrorCategory
from src.utils.logger import get_logger

# Initialize logger
logger = get_logger(__name__)


class ChatServiceException(Exception):
    """
    Base exception class for all Chat Service custom exceptions.

    This provides a consistent interface for error handling with
    structured error information and logging integration.
    """

    def __init__(
            self,
            message: str,
            error_code: str = "INTERNAL_ERROR",
            status_code: int = 500,
            details: Optional[Dict[str, Any]] = None,
            category: ErrorCategory = ErrorCategory.INTERNAL,
            user_message: Optional[str] = None,
            correlation_id: Optional[str] = None,
            tenant_id: Optional[str] = None,
            user_id: Optional[str] = None,
            retryable: bool = False,
            caused_by: Optional[Exception] = None
    ):
        """
        Initialize Chat Service exception.

        Args:
            message: Internal error message for logging
            error_code: Machine-readable error code
            status_code: HTTP status code
            details: Additional error details
            category: Error category for monitoring
            user_message: User-friendly error message
            correlation_id: Request correlation ID
            tenant_id: Tenant ID for multi-tenancy
            user_id: User ID if available
            retryable: Whether the operation can be retried
            caused_by: Original exception that caused this error
        """
        super().__init__(message)

        self.message = message
        self.error_code = error_code
        self.status_code = status_code
        self.details = details or {}
        self.category = category
        self.user_message = user_message or message
        self.correlation_id = correlation_id
        self.tenant_id = tenant_id
        self.user_id = user_id
        self.retryable = retryable
        self.caused_by = caused_by
        self.timestamp = datetime.utcnow()

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert exception to dictionary representation.

        Returns:
            Dictionary with error information
        """
        error_dict = {
            "error": {
                "code": self.error_code,
                "message": self.user_message,
                "category": self.category.value,
                "timestamp": self.timestamp.isoformat(),
                "retryable": self.retryable,
            }
        }

        if self.details:
            error_dict["error"]["details"] = self.details

        if self.correlation_id:
            error_dict["error"]["correlation_id"] = self.correlation_id

        return error_dict

    def log_error(self, logger: Optional[structlog.BoundLogger] = None) -> None:
        """
        Log the error with appropriate level and context.

        Args:
            logger: Logger instance to use
        """
        if logger is None:
            logger = get_logger(__name__)

        log_data = {
            "error_code": self.error_code,
            "error_category": self.category.value,
            "status_code": self.status_code,
            "retryable": self.retryable,
        }

        if self.correlation_id:
            log_data["correlation_id"] = self.correlation_id

        if self.tenant_id:
            log_data["tenant_id"] = self.tenant_id

        if self.user_id:
            log_data["user_id"] = self.user_id

        if self.details:
            log_data["details"] = self.details

        if self.caused_by:
            log_data["caused_by"] = str(self.caused_by)
            log_data["caused_by_type"] = type(self.caused_by).__name__

        # Choose appropriate log level based on status code
        if self.status_code >= 500:
            logger.error(self.message, **log_data, exc_info=True)
        elif self.status_code >= 400:
            logger.warning(self.message, **log_data)
        else:
            logger.info(self.message, **log_data)


class ValidationError(ChatServiceException):
    """Exception for request validation errors."""

    def __init__(
            self,
            message: str = "Request validation failed",
            field: Optional[str] = None,
            value: Optional[Any] = None,
            **kwargs
    ):
        details = kwargs.pop("details", {})
        if field:
            details["field"] = field
        if value is not None:
            details["value"] = str(value)

        super().__init__(
            message=message,
            error_code="VALIDATION_ERROR",
            status_code=400,
            category=ErrorCategory.VALIDATION,
            details=details,
            user_message="The request contains invalid data. Please check your input and try again.",
            **kwargs
        )


class AuthenticationError(ChatServiceException):
    """Exception for authentication failures."""

    def __init__(
            self,
            message: str = "Authentication failed",
            **kwargs
    ):
        super().__init__(
            message=message,
            error_code="AUTHENTICATION_FAILED",
            status_code=401,
            category=ErrorCategory.AUTHENTICATION,
            user_message="Authentication is required to access this resource.",
            **kwargs
        )


class AuthorizationError(ChatServiceException):
    """Exception for authorization failures."""

    def __init__(
            self,
            message: str = "Authorization failed",
            required_permission: Optional[str] = None,
            **kwargs
    ):
        details = kwargs.pop("details", {})
        if required_permission:
            details["required_permission"] = required_permission

        super().__init__(
            message=message,
            error_code="AUTHORIZATION_FAILED",
            status_code=403,
            category=ErrorCategory.AUTHORIZATION,
            details=details,
            user_message="You don't have permission to access this resource.",
            **kwargs
        )


class NotFoundError(ChatServiceException):
    """Exception for resource not found errors."""

    def __init__(
            self,
            message: str = "Resource not found",
            resource_type: Optional[str] = None,
            resource_id: Optional[str] = None,
            **kwargs
    ):
        details = kwargs.pop("details", {})
        if resource_type:
            details["resource_type"] = resource_type
        if resource_id:
            details["resource_id"] = resource_id

        super().__init__(
            message=message,
            error_code="RESOURCE_NOT_FOUND",
            status_code=404,
            category=ErrorCategory.NOT_FOUND,
            details=details,
            user_message="The requested resource could not be found.",
            **kwargs
        )


class RateLimitError(ChatServiceException):
    """Exception for rate limit exceeded errors."""

    def __init__(
            self,
            message: str = "Rate limit exceeded",
            limit: Optional[int] = None,
            window: Optional[int] = None,
            reset_time: Optional[int] = None,
            **kwargs
    ):
        details = kwargs.pop("details", {})
        if limit:
            details["limit"] = limit
        if window:
            details["window_seconds"] = window
        if reset_time:
            details["reset_time"] = reset_time

        super().__init__(
            message=message,
            error_code="RATE_LIMIT_EXCEEDED",
            status_code=429,
            category=ErrorCategory.RATE_LIMIT,
            details=details,
            user_message="Too many requests. Please wait before trying again.",
            retryable=True,
            **kwargs
        )


class InternalServerError(ChatServiceException):
    """Exception for internal server errors."""

    def __init__(
            self,
            message: str = "Internal server error",
            **kwargs
    ):
        super().__init__(
            message=message,
            error_code="INTERNAL_SERVER_ERROR",
            status_code=500,
            category=ErrorCategory.INTERNAL,
            user_message="An unexpected error occurred. Please try again later.",
            retryable=True,
            **kwargs
        )


class ExternalServiceError(ChatServiceException):
    """Exception for external service errors."""

    def __init__(
            self,
            message: str = "External service error",
            service_name: Optional[str] = None,
            service_error: Optional[str] = None,
            **kwargs
    ):
        details = kwargs.pop("details", {})
        if service_name:
            details["service_name"] = service_name
        if service_error:
            details["service_error"] = service_error

        super().__init__(
            message=message,
            error_code="EXTERNAL_SERVICE_ERROR",
            status_code=502,
            category=ErrorCategory.EXTERNAL,
            details=details,
            user_message="A required service is currently unavailable. Please try again later.",
            retryable=True,
            **kwargs
        )


class TimeoutError(ChatServiceException):
    """Exception for timeout errors."""

    def __init__(
            self,
            message: str = "Request timeout",
            timeout_seconds: Optional[float] = None,
            operation: Optional[str] = None,
            **kwargs
    ):
        details = kwargs.pop("details", {})
        if timeout_seconds:
            details["timeout_seconds"] = timeout_seconds
        if operation:
            details["operation"] = operation

        super().__init__(
            message=message,
            error_code="TIMEOUT_ERROR",
            status_code=504,
            category=ErrorCategory.TIMEOUT,
            details=details,
            user_message="The request took too long to complete. Please try again.",
            retryable=True,
            **kwargs
        )


class ConfigurationError(ChatServiceException):
    """Exception for configuration errors."""

    def __init__(
            self,
            message: str = "Configuration error",
            config_key: Optional[str] = None,
            **kwargs
    ):
        details = kwargs.pop("details", {})
        if config_key:
            details["config_key"] = config_key

        super().__init__(
            message=message,
            error_code="CONFIGURATION_ERROR",
            status_code=500,
            category=ErrorCategory.INTERNAL,
            details=details,
            user_message="Service configuration error. Please contact support.",
            **kwargs
        )


async def chat_service_exception_handler(
        request: Request,
        exc: ChatServiceException
) -> JSONResponse:
    """
    Handler for Chat Service custom exceptions.

    Args:
        request: FastAPI request object
        exc: Chat Service exception instance

    Returns:
        JSON response with error details
    """
    # Log the error
    exc.log_error()

    # Create response
    response_data = {
        "status": "error",
        **exc.to_dict(),
        "meta": {
            "timestamp": exc.timestamp.isoformat(),
            "path": str(request.url.path),
            "method": request.method,
        }
    }

    # Add request ID if available
    request_id = getattr(request.state, "request_id", None)
    if request_id:
        response_data["meta"]["request_id"] = request_id

    return JSONResponse(
        status_code=exc.status_code,
        content=response_data
    )


async def http_exception_handler(
        request: Request,
        exc: HTTPException
) -> JSONResponse:
    """
    Handler for standard HTTP exceptions.

    Args:
        request: FastAPI request object
        exc: HTTP exception instance

    Returns:
        JSON response with error details
    """
    # Map status codes to error categories
    category_map = {
        400: ErrorCategory.VALIDATION,
        401: ErrorCategory.AUTHENTICATION,
        403: ErrorCategory.AUTHORIZATION,
        404: ErrorCategory.NOT_FOUND,
        429: ErrorCategory.RATE_LIMIT,
        500: ErrorCategory.INTERNAL,
        502: ErrorCategory.EXTERNAL,
        503: ErrorCategory.NETWORK,
        504: ErrorCategory.TIMEOUT,
    }

    category = category_map.get(exc.status_code, ErrorCategory.INTERNAL)

    # Create structured error response
    error_data = {
        "status": "error",
        "error": {
            "code": f"HTTP_{exc.status_code}",
            "message": exc.detail,
            "category": category.value,
            "timestamp": datetime.utcnow().isoformat(),
        },
        "meta": {
            "timestamp": datetime.utcnow().isoformat(),
            "path": str(request.url.path),
            "method": request.method,
        }
    }

    # Add request ID if available
    request_id = getattr(request.state, "request_id", None)
    if request_id:
        error_data["meta"]["request_id"] = request_id

    # Log the error
    logger.warning(
        "HTTP exception occurred",
        status_code=exc.status_code,
        detail=exc.detail,
        path=str(request.url.path),
        method=request.method,
        request_id=request_id
    )

    return JSONResponse(
        status_code=exc.status_code,
        content=error_data
    )


async def validation_exception_handler(
        request: Request,
        exc: RequestValidationError
) -> JSONResponse:
    """
    Handler for request validation errors.

    Args:
        request: FastAPI request object
        exc: Request validation error instance

    Returns:
        JSON response with validation error details
    """
    # Extract validation errors
    validation_errors = []
    for error in exc.errors():
        validation_errors.append({
            "field": ".".join(str(x) for x in error["loc"]),
            "message": error["msg"],
            "type": error["type"],
            "input": error.get("input"),
        })

    error_data = {
        "status": "error",
        "error": {
            "code": "VALIDATION_ERROR",
            "message": "Request validation failed",
            "category": ErrorCategory.VALIDATION.value,
            "timestamp": datetime.utcnow().isoformat(),
            "details": {
                "validation_errors": validation_errors
            }
        },
        "meta": {
            "timestamp": datetime.utcnow().isoformat(),
            "path": str(request.url.path),
            "method": request.method,
        }
    }

    # Add request ID if available
    request_id = getattr(request.state, "request_id", None)
    if request_id:
        error_data["meta"]["request_id"] = request_id

    # Log validation error
    logger.warning(
        "Request validation failed",
        validation_errors=validation_errors,
        path=str(request.url.path),
        method=request.method,
        request_id=request_id
    )

    return JSONResponse(
        status_code=400,
        content=error_data
    )


async def generic_exception_handler(
        request: Request,
        exc: Exception
) -> JSONResponse:
    """
    Handler for unexpected exceptions.

    Args:
        request: FastAPI request object
        exc: Generic exception instance

    Returns:
        JSON response with generic error message
    """
    # Generate error ID for tracking
    import uuid
    error_id = str(uuid.uuid4())

    error_data = {
        "status": "error",
        "error": {
            "code": "INTERNAL_SERVER_ERROR",
            "message": "An unexpected error occurred",
            "category": ErrorCategory.INTERNAL.value,
            "timestamp": datetime.utcnow().isoformat(),
            "error_id": error_id,
        },
        "meta": {
            "timestamp": datetime.utcnow().isoformat(),
            "path": str(request.url.path),
            "method": request.method,
        }
    }

    # Add request ID if available
    request_id = getattr(request.state, "request_id", None)
    if request_id:
        error_data["meta"]["request_id"] = request_id

    # Log the unexpected error with full details
    logger.error(
        "Unexpected exception occurred",
        error_id=error_id,
        error_type=type(exc).__name__,
        error_message=str(exc),
        path=str(request.url.path),
        method=request.method,
        request_id=request_id,
        traceback=traceback.format_exc(),
        exc_info=True
    )

    return JSONResponse(
        status_code=500,
        content=error_data
    )


def setup_exception_handlers(app: FastAPI) -> None:
    """
    Setup exception handlers for the FastAPI application.

    Args:
        app: FastAPI application instance
    """
    # Custom exception handlers
    app.add_exception_handler(ChatServiceException, chat_service_exception_handler)
    app.add_exception_handler(HTTPException, http_exception_handler)
    app.add_exception_handler(StarletteHTTPException, http_exception_handler)
    app.add_exception_handler(RequestValidationError, validation_exception_handler)

    # Generic exception handler (catch-all)
    app.add_exception_handler(Exception, generic_exception_handler)

    logger.info("Exception handlers configured successfully")


# Export all exception classes and utilities
__all__ = [
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
    "setup_exception_handlers",
    "chat_service_exception_handler",
    "http_exception_handler",
    "validation_exception_handler",
    "generic_exception_handler",
]