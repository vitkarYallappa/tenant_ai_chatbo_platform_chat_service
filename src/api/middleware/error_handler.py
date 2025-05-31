"""
Error Handling Middleware
Centralized error handling with standardized responses and logging.
"""

import traceback
from typing import Dict, Any, Optional
from fastapi import Request, HTTPException, status
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
from pydantic import ValidationError
import structlog
from uuid import uuid4

from src.api.responses.api_response import (
    create_error_response,
    create_validation_error_response
)
from src.api.responses.error_responses import (
    create_validation_error,
    create_authentication_error,
    create_authorization_error,
    create_not_found_error,
    create_rate_limit_error,
    create_internal_error
)
from src.services.exceptions import (
    ServiceError,
    ValidationError as ServiceValidationError,
    NotFoundError,
    UnauthorizedError,
    ForbiddenError,
    ConflictError,
    RateLimitError,
    TimeoutError,
    ExternalServiceError,
    QuotaExceededError
)

logger = structlog.get_logger()


class ErrorHandlingMiddleware:
    """Middleware for centralized error handling"""

    def __init__(self, app):
        self.app = app

    async def __call__(self, request: Request, call_next):
        """Process request through error handling"""

        # Generate trace ID for request tracking
        trace_id = str(uuid4())
        request.state.trace_id = trace_id

        try:
            response = await call_next(request)
            return response

        except Exception as e:
            return await self.handle_exception(request, e, trace_id)

    async def handle_exception(
            self,
            request: Request,
            exc: Exception,
            trace_id: str
    ) -> JSONResponse:
        """
        Handle different types of exceptions with appropriate responses

        Args:
            request: FastAPI request object
            exc: Exception that occurred
            trace_id: Trace ID for tracking

        Returns:
            JSONResponse with error details
        """
        # Extract request context for logging
        request_context = self._extract_request_context(request)

        # Handle different exception types
        if isinstance(exc, HTTPException):
            return await self._handle_http_exception(exc, trace_id, request_context)
        elif isinstance(exc, RequestValidationError):
            return await self._handle_validation_error(exc, trace_id, request_context)
        elif isinstance(exc, ServiceValidationError):
            return await self._handle_service_validation_error(exc, trace_id, request_context)
        elif isinstance(exc, NotFoundError):
            return await self._handle_not_found_error(exc, trace_id, request_context)
        elif isinstance(exc, UnauthorizedError):
            return await self._handle_unauthorized_error(exc, trace_id, request_context)
        elif isinstance(exc, ForbiddenError):
            return await self._handle_forbidden_error(exc, trace_id, request_context)
        elif isinstance(exc, ConflictError):
            return await self._handle_conflict_error(exc, trace_id, request_context)
        elif isinstance(exc, RateLimitError):
            return await self._handle_rate_limit_error(exc, trace_id, request_context)
        elif isinstance(exc, TimeoutError):
            return await self._handle_timeout_error(exc, trace_id, request_context)
        elif isinstance(exc, ExternalServiceError):
            return await self._handle_external_service_error(exc, trace_id, request_context)
        elif isinstance(exc, QuotaExceededError):
            return await self._handle_quota_exceeded_error(exc, trace_id, request_context)
        elif isinstance(exc, ServiceError):
            return await self._handle_service_error(exc, trace_id, request_context)
        else:
            return await self._handle_unexpected_error(exc, trace_id, request_context)

    def _extract_request_context(self, request: Request) -> Dict[str, Any]:
        """Extract relevant request context for logging"""
        return {
            "method": request.method,
            "url": str(request.url),
            "path": request.url.path,
            "client_ip": request.client.host if request.client else "unknown",
            "user_agent": request.headers.get("user-agent"),
            "tenant_id": getattr(request.state, "tenant_id", None),
            "trace_id": getattr(request.state, "trace_id", None)
        }

    async def _handle_http_exception(
            self,
            exc: HTTPException,
            trace_id: str,
            context: Dict[str, Any]
    ) -> JSONResponse:
        """Handle FastAPI HTTP exceptions"""

        logger.warning(
            "HTTP exception occurred",
            status_code=exc.status_code,
            detail=exc.detail,
            trace_id=trace_id,
            **context
        )

        # Map HTTP status codes to error codes
        error_code_mapping = {
            400: "BAD_REQUEST",
            401: "AUTHENTICATION_FAILED",
            403: "ACCESS_DENIED",
            404: "RESOURCE_NOT_FOUND",
            405: "METHOD_NOT_ALLOWED",
            409: "RESOURCE_CONFLICT",
            413: "PAYLOAD_TOO_LARGE",
            415: "UNSUPPORTED_MEDIA_TYPE",
            422: "VALIDATION_ERROR",
            429: "RATE_LIMIT_EXCEEDED",
            500: "INTERNAL_SERVER_ERROR",
            502: "BAD_GATEWAY",
            503: "SERVICE_UNAVAILABLE",
            504: "GATEWAY_TIMEOUT"
        }

        error_code = error_code_mapping.get(exc.status_code, "HTTP_ERROR")

        error_response = create_error_response(
            error_code=error_code,
            error_message=str(exc.detail),
            trace_id=trace_id
        )

        headers = getattr(exc, "headers", {})

        return JSONResponse(
            status_code=exc.status_code,
            content=error_response.dict(),
            headers=headers
        )

    async def _handle_validation_error(
            self,
            exc: RequestValidationError,
            trace_id: str,
            context: Dict[str, Any]
    ) -> JSONResponse:
        """Handle Pydantic validation errors"""

        logger.warning(
            "Request validation error",
            errors=exc.errors(),
            trace_id=trace_id,
            **context
        )

        # Format validation errors
        validation_errors = []
        for error in exc.errors():
            field_path = " -> ".join(str(loc) for loc in error["loc"])
            validation_errors.append({
                "field": field_path,
                "message": error["msg"],
                "type": error["type"],
                "input": error.get("input")
            })

        error_response = create_validation_error_response(
            validation_errors={"errors": validation_errors},
            request_id=trace_id
        )

        return JSONResponse(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            content=error_response.dict()
        )

    async def _handle_service_validation_error(
            self,
            exc: ServiceValidationError,
            trace_id: str,
            context: Dict[str, Any]
    ) -> JSONResponse:
        """Handle service layer validation errors"""

        logger.warning(
            "Service validation error",
            error=str(exc),
            trace_id=trace_id,
            **context
        )

        error_response = create_error_response(
            error_code="VALIDATION_ERROR",
            error_message=str(exc),
            error_details=getattr(exc, "details", None),
            trace_id=trace_id
        )

        return JSONResponse(
            status_code=status.HTTP_400_BAD_REQUEST,
            content=error_response.dict()
        )

    async def _handle_not_found_error(
            self,
            exc: NotFoundError,
            trace_id: str,
            context: Dict[str, Any]
    ) -> JSONResponse:
        """Handle not found errors"""

        logger.info(
            "Resource not found",
            error=str(exc),
            trace_id=trace_id,
            **context
        )

        error_response = create_error_response(
            error_code="RESOURCE_NOT_FOUND",
            error_message=str(exc),
            error_details=getattr(exc, "details", None),
            trace_id=trace_id
        )

        return JSONResponse(
            status_code=status.HTTP_404_NOT_FOUND,
            content=error_response.dict()
        )

    async def _handle_unauthorized_error(
            self,
            exc: UnauthorizedError,
            trace_id: str,
            context: Dict[str, Any]
    ) -> JSONResponse:
        """Handle unauthorized errors"""

        logger.warning(
            "Unauthorized access attempt",
            error=str(exc),
            trace_id=trace_id,
            **context
        )

        error_response = create_error_response(
            error_code="AUTHENTICATION_FAILED",
            error_message=str(exc),
            error_details=getattr(exc, "details", None),
            trace_id=trace_id
        )

        return JSONResponse(
            status_code=status.HTTP_401_UNAUTHORIZED,
            content=error_response.dict(),
            headers={"WWW-Authenticate": "Bearer"}
        )

    async def _handle_forbidden_error(
            self,
            exc: ForbiddenError,
            trace_id: str,
            context: Dict[str, Any]
    ) -> JSONResponse:
        """Handle forbidden errors"""

        logger.warning(
            "Forbidden access attempt",
            error=str(exc),
            trace_id=trace_id,
            **context
        )

        error_response = create_error_response(
            error_code="ACCESS_DENIED",
            error_message=str(exc),
            error_details=getattr(exc, "details", None),
            trace_id=trace_id
        )

        return JSONResponse(
            status_code=status.HTTP_403_FORBIDDEN,
            content=error_response.dict()
        )

    async def _handle_conflict_error(
            self,
            exc: ConflictError,
            trace_id: str,
            context: Dict[str, Any]
    ) -> JSONResponse:
        """Handle conflict errors"""

        logger.warning(
            "Resource conflict",
            error=str(exc),
            trace_id=trace_id,
            **context
        )

        error_response = create_error_response(
            error_code="RESOURCE_CONFLICT",
            error_message=str(exc),
            error_details=getattr(exc, "details", None),
            trace_id=trace_id
        )

        return JSONResponse(
            status_code=status.HTTP_409_CONFLICT,
            content=error_response.dict()
        )

    async def _handle_rate_limit_error(
            self,
            exc: RateLimitError,
            trace_id: str,
            context: Dict[str, Any]
    ) -> JSONResponse:
        """Handle rate limit errors"""

        logger.warning(
            "Rate limit exceeded",
            error=str(exc),
            trace_id=trace_id,
            **context
        )

        error_details = getattr(exc, "details", {})
        retry_after = error_details.get("retry_after_seconds", 60)

        error_response = create_error_response(
            error_code="RATE_LIMIT_EXCEEDED",
            error_message=str(exc),
            error_details=error_details,
            trace_id=trace_id
        )

        headers = {
            "Retry-After": str(retry_after)
        }

        # Add rate limit headers if available
        if "limit" in error_details:
            headers["X-RateLimit-Limit"] = str(error_details["limit"])
        if "remaining" in error_details:
            headers["X-RateLimit-Remaining"] = str(error_details["remaining"])
        if "reset" in error_details:
            headers["X-RateLimit-Reset"] = str(error_details["reset"])

        return JSONResponse(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            content=error_response.dict(),
            headers=headers
        )

    async def _handle_timeout_error(
            self,
            exc: TimeoutError,
            trace_id: str,
            context: Dict[str, Any]
    ) -> JSONResponse:
        """Handle timeout errors"""

        logger.warning(
            "Request timeout",
            error=str(exc),
            trace_id=trace_id,
            **context
        )

        error_response = create_error_response(
            error_code="REQUEST_TIMEOUT",
            error_message=str(exc),
            error_details=getattr(exc, "details", None),
            trace_id=trace_id
        )

        return JSONResponse(
            status_code=status.HTTP_504_GATEWAY_TIMEOUT,
            content=error_response.dict()
        )

    async def _handle_external_service_error(
            self,
            exc: ExternalServiceError,
            trace_id: str,
            context: Dict[str, Any]
    ) -> JSONResponse:
        """Handle external service errors"""

        logger.error(
            "External service error",
            error=str(exc),
            service=getattr(exc, "service_name", "unknown"),
            trace_id=trace_id,
            **context
        )

        # Don't expose internal service details to users
        user_message = "An external service is temporarily unavailable. Please try again later."

        error_response = create_error_response(
            error_code="EXTERNAL_SERVICE_ERROR",
            error_message=user_message,
            error_details={"service_error": True},
            trace_id=trace_id
        )

        return JSONResponse(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            content=error_response.dict(),
            headers={"Retry-After": "60"}
        )

    async def _handle_quota_exceeded_error(
            self,
            exc: QuotaExceededError,
            trace_id: str,
            context: Dict[str, Any]
    ) -> JSONResponse:
        """Handle quota exceeded errors"""

        logger.warning(
            "Quota exceeded",
            error=str(exc),
            trace_id=trace_id,
            **context
        )

        error_details = getattr(exc, "details", {})

        error_response = create_error_response(
            error_code="QUOTA_EXCEEDED",
            error_message=str(exc),
            error_details=error_details,
            trace_id=trace_id
        )

        headers = {}
        if "reset_time" in error_details:
            headers["X-Quota-Reset"] = str(error_details["reset_time"])

        return JSONResponse(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            content=error_response.dict(),
            headers=headers
        )

    async def _handle_service_error(
            self,
            exc: ServiceError,
            trace_id: str,
            context: Dict[str, Any]
    ) -> JSONResponse:
        """Handle generic service errors"""

        logger.error(
            "Service error",
            error=str(exc),
            error_type=type(exc).__name__,
            trace_id=trace_id,
            **context
        )

        error_response = create_error_response(
            error_code="SERVICE_ERROR",
            error_message="A service error occurred. Please try again.",
            error_details={"internal_error": True},
            trace_id=trace_id
        )

        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content=error_response.dict()
        )

    async def _handle_unexpected_error(
            self,
            exc: Exception,
            trace_id: str,
            context: Dict[str, Any]
    ) -> JSONResponse:
        """Handle unexpected errors"""

        # Log full traceback for unexpected errors
        logger.error(
            "Unexpected error occurred",
            error=str(exc),
            error_type=type(exc).__name__,
            traceback=traceback.format_exc(),
            trace_id=trace_id,
            **context
        )

        # Don't expose internal error details to users
        error_response = create_error_response(
            error_code="INTERNAL_SERVER_ERROR",
            error_message="An unexpected error occurred. Please try again later.",
            error_details={
                "internal_error": True,
                "support_reference": trace_id
            },
            trace_id=trace_id
        )

        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content=error_response.dict()
        )


# Exception handlers for specific FastAPI exception types
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    """Handle FastAPI validation exceptions"""
    trace_id = getattr(request.state, "trace_id", str(uuid4()))

    logger.warning(
        "FastAPI validation error",
        errors=exc.errors(),
        trace_id=trace_id
    )

    validation_errors = []
    for error in exc.errors():
        field_path = " -> ".join(str(loc) for loc in error["loc"])
        validation_errors.append({
            "field": field_path,
            "message": error["msg"],
            "type": error["type"]
        })

    error_response = create_validation_error_response(
        validation_errors={"errors": validation_errors},
        request_id=trace_id
    )

    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content=error_response.dict()
    )


async def http_exception_handler(request: Request, exc: HTTPException):
    """Handle FastAPI HTTP exceptions"""
    trace_id = getattr(request.state, "trace_id", str(uuid4()))

    logger.warning(
        "HTTP exception",
        status_code=exc.status_code,
        detail=exc.detail,
        trace_id=trace_id
    )

    error_response = create_error_response(
        error_code=f"HTTP_{exc.status_code}",
        error_message=str(exc.detail),
        trace_id=trace_id
    )

    return JSONResponse(
        status_code=exc.status_code,
        content=error_response.dict(),
        headers=getattr(exc, "headers", {})
    )


# Health check for error handling
async def test_error_handling():
    """Test function to verify error handling is working"""
    raise Exception("Test error for error handling verification")
