"""
Dependency injection utilities for FastAPI.

This module provides dependency injection functions and utilities
for managing request context, authentication, and service dependencies.
"""

import uuid
from typing import Optional, Dict, Any, AsyncGenerator
from contextlib import asynccontextmanager

from fastapi import Request, HTTPException, Header, Depends
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
import structlog

from src.config.settings import get_settings
from src.utils.logger import get_logger, bind_context

# Initialize logger
logger = get_logger(__name__)

# HTTP Bearer token scheme for authentication
security = HTTPBearer(auto_error=False)


class HealthChecker:
    """Service health monitoring and dependency checking."""

    def __init__(self):
        self.is_healthy = True
        self.last_check = None
        self.dependencies_status: Dict[str, Dict[str, Any]] = {}
        self.consecutive_failures = 0

    async def check_health(self) -> Dict[str, Any]:
        """
        Perform comprehensive health check.

        Returns:
            Health status dictionary with dependencies and metrics
        """
        from datetime import datetime
        import asyncio

        self.last_check = datetime.utcnow()
        overall_healthy = True
        dependencies = {}

        # In Phase 1, we only have basic service health
        # This will be expanded in later phases with actual dependency checks
        try:
            # Placeholder for future dependency checks
            dependencies["service"] = {
                "status": "healthy",
                "latency_ms": 0,
                "last_check": self.last_check.isoformat(),
                "consecutive_failures": self.consecutive_failures
            }

            self.consecutive_failures = 0

        except Exception as e:
            logger.error("Health check failed", error=str(e))
            self.consecutive_failures += 1
            overall_healthy = False

            dependencies["service"] = {
                "status": "unhealthy",
                "error": str(e),
                "last_check": self.last_check.isoformat(),
                "consecutive_failures": self.consecutive_failures
            }

        self.is_healthy = overall_healthy
        self.dependencies_status = dependencies

        return {
            "status": "healthy" if overall_healthy else "unhealthy",
            "timestamp": self.last_check.isoformat(),
            "dependencies": dependencies,
            "consecutive_failures": self.consecutive_failures
        }

    async def is_service_healthy(self) -> bool:
        """
        Quick health check for service availability.

        Returns:
            Boolean indicating if service is healthy
        """
        try:
            health_status = await self.check_health()
            return health_status["status"] == "healthy"
        except Exception:
            return False


# Global health checker instance
health_checker = HealthChecker()


async def get_health_checker() -> HealthChecker:
    """
    FastAPI dependency for health checker.

    Returns:
        HealthChecker instance
    """
    return health_checker


def get_request_id(request: Request) -> str:
    """
    Extract or generate request ID for tracking.

    Args:
        request: FastAPI request object

    Returns:
        Request ID string
    """
    # Try to get from header first
    request_id = request.headers.get("x-request-id")

    if not request_id:
        # Generate new UUID if not provided
        request_id = str(uuid.uuid4())

    # Bind to logging context
    bind_context(request_id=request_id)

    return request_id


def get_correlation_id(
        request: Request,
        x_correlation_id: Optional[str] = Header(None, alias="X-Correlation-ID")
) -> str:
    """
    Extract or generate correlation ID for request tracing.

    Args:
        request: FastAPI request object
        x_correlation_id: Correlation ID from header

    Returns:
        Correlation ID string
    """
    # Use header value if provided, otherwise generate new
    correlation_id = x_correlation_id or str(uuid.uuid4())

    # Bind to logging context
    bind_context(correlation_id=correlation_id)

    return correlation_id


def get_tenant_id(
        x_tenant_id: Optional[str] = Header(None, alias="X-Tenant-ID")
) -> Optional[str]:
    """
    Extract tenant ID from request headers.

    Args:
        x_tenant_id: Tenant ID from header

    Returns:
        Tenant ID string or None

    Raises:
        HTTPException: If tenant ID is required but not provided
    """
    if x_tenant_id:
        # Validate tenant ID format (basic UUID validation)
        try:
            uuid.UUID(x_tenant_id)
        except ValueError:
            raise HTTPException(
                status_code=400,
                detail="Invalid tenant ID format"
            )

        # Bind to logging context
        bind_context(tenant_id=x_tenant_id)

    return x_tenant_id


def require_tenant_id(
        x_tenant_id: Optional[str] = Header(None, alias="X-Tenant-ID")
) -> str:
    """
    Extract and require tenant ID from request headers.

    Args:
        x_tenant_id: Tenant ID from header

    Returns:
        Tenant ID string

    Raises:
        HTTPException: If tenant ID is not provided or invalid
    """
    if not x_tenant_id:
        raise HTTPException(
            status_code=400,
            detail="Tenant ID is required in X-Tenant-ID header"
        )

    # Validate and return tenant ID
    return get_tenant_id(x_tenant_id)


def get_user_agent(request: Request) -> Optional[str]:
    """
    Extract User-Agent from request headers.

    Args:
        request: FastAPI request object

    Returns:
        User-Agent string or None
    """
    return request.headers.get("user-agent")


def get_client_ip(request: Request) -> str:
    """
    Extract client IP address considering proxy headers.

    Args:
        request: FastAPI request object

    Returns:
        Client IP address string
    """
    # Check for common proxy headers
    forwarded_ips = request.headers.get("x-forwarded-for")
    if forwarded_ips:
        # Take the first IP from the chain
        return forwarded_ips.split(",")[0].strip()

    real_ip = request.headers.get("x-real-ip")
    if real_ip:
        return real_ip

    # Fallback to direct client IP
    return request.client.host if request.client else "unknown"


async def get_auth_token(
        credentials: Optional[HTTPAuthorizationCredentials] = Depends(security)
) -> Optional[str]:
    """
    Extract authentication token from Authorization header.

    Args:
        credentials: HTTP Bearer credentials

    Returns:
        JWT token string or None
    """
    if credentials and credentials.scheme.lower() == "bearer":
        return credentials.credentials
    return None


def require_auth_token(
        credentials: Optional[HTTPAuthorizationCredentials] = Depends(security)
) -> str:
    """
    Extract and require authentication token from Authorization header.

    Args:
        credentials: HTTP Bearer credentials

    Returns:
        JWT token string

    Raises:
        HTTPException: If token is not provided
    """
    if not credentials or credentials.scheme.lower() != "bearer":
        raise HTTPException(
            status_code=401,
            detail="Bearer token required"
        )

    if not credentials.credentials:
        raise HTTPException(
            status_code=401,
            detail="Invalid or missing token"
        )

    return credentials.credentials


def get_api_key(
        x_api_key: Optional[str] = Header(None, alias="X-API-Key")
) -> Optional[str]:
    """
    Extract API key from request headers.

    Args:
        x_api_key: API key from header

    Returns:
        API key string or None
    """
    return x_api_key


def require_api_key(
        x_api_key: Optional[str] = Header(None, alias="X-API-Key")
) -> str:
    """
    Extract and require API key from request headers.

    Args:
        x_api_key: API key from header

    Returns:
        API key string

    Raises:
        HTTPException: If API key is not provided
    """
    if not x_api_key:
        raise HTTPException(
            status_code=401,
            detail="API key required in X-API-Key header"
        )

    return x_api_key


@asynccontextmanager
async def request_context(
        request: Request,
        tenant_id: Optional[str] = None,
        user_id: Optional[str] = None
) -> AsyncGenerator[Dict[str, Any], None]:
    """
    Context manager for request-scoped data and logging context.

    Args:
        request: FastAPI request object
        tenant_id: Optional tenant ID
        user_id: Optional user ID

    Yields:
        Request context dictionary
    """
    import time

    # Generate request tracking IDs
    request_id = get_request_id(request)
    correlation_id = get_correlation_id(request)

    # Build context data
    context = {
        "request_id": request_id,
        "correlation_id": correlation_id,
        "method": request.method,
        "path": request.url.path,
        "client_ip": get_client_ip(request),
        "user_agent": get_user_agent(request),
        "started_at": time.time(),
    }

    if tenant_id:
        context["tenant_id"] = tenant_id

    if user_id:
        context["user_id"] = user_id

    # Bind logging context
    bind_context(**context)

    logger.info(
        "Request started",
        method=request.method,
        path=request.url.path,
        query_params=dict(request.query_params) if request.query_params else None
    )

    try:
        yield context
    finally:
        # Log request completion
        elapsed_time = (time.time() - context["started_at"]) * 1000
        logger.info(
            "Request completed",
            method=request.method,
            path=request.url.path,
            duration_ms=round(elapsed_time, 2)
        )


def get_pagination_params(
        page: int = 1,
        page_size: int = 20,
        sort_by: Optional[str] = None,
        sort_order: str = "desc"
) -> Dict[str, Any]:
    """
    Extract and validate pagination parameters.

    Args:
        page: Page number (1-based)
        page_size: Number of items per page
        sort_by: Field to sort by
        sort_order: Sort order (asc/desc)

    Returns:
        Pagination parameters dictionary

    Raises:
        HTTPException: If pagination parameters are invalid
    """
    from src.config.constants import MAX_PAGE_SIZE, MIN_PAGE_SIZE

    # Validate page number
    if page < 1:
        raise HTTPException(
            status_code=400,
            detail="Page number must be greater than 0"
        )

    # Validate page size
    if page_size < MIN_PAGE_SIZE or page_size > MAX_PAGE_SIZE:
        raise HTTPException(
            status_code=400,
            detail=f"Page size must be between {MIN_PAGE_SIZE} and {MAX_PAGE_SIZE}"
        )

    # Validate sort order
    if sort_order.lower() not in ["asc", "desc"]:
        raise HTTPException(
            status_code=400,
            detail="Sort order must be 'asc' or 'desc'"
        )

    return {
        "page": page,
        "page_size": page_size,
        "offset": (page - 1) * page_size,
        "sort_by": sort_by,
        "sort_order": sort_order.lower()
    }


class RateLimitChecker:
    """Rate limiting functionality for API endpoints."""

    def __init__(self):
        self.enabled = True
        self.default_limit = 1000  # requests per minute

    async def check_rate_limit(
            self,
            identifier: str,
            limit: Optional[int] = None,
            window: int = 60
    ) -> Dict[str, Any]:
        """
        Check rate limit for identifier.

        Args:
            identifier: Unique identifier (tenant_id, api_key, etc.)
            limit: Rate limit (requests per window)
            window: Time window in seconds

        Returns:
            Rate limit status dictionary
        """
        # Placeholder implementation for Phase 1
        # This will be implemented with Redis in later phases

        limit = limit or self.default_limit

        return {
            "allowed": True,
            "limit": limit,
            "remaining": limit - 1,
            "reset_time": window,
            "identifier": identifier
        }


# Global rate limiter instance
rate_limiter = RateLimitChecker()


async def get_rate_limiter() -> RateLimitChecker:
    """
    FastAPI dependency for rate limiter.

    Returns:
        RateLimitChecker instance
    """
    return rate_limiter


def validate_content_type(
        request: Request,
        allowed_types: list[str] = None
) -> str:
    """
    Validate request content type.

    Args:
        request: FastAPI request object
        allowed_types: List of allowed content types

    Returns:
        Content type string

    Raises:
        HTTPException: If content type is not allowed
    """
    if allowed_types is None:
        allowed_types = ["application/json"]

    content_type = request.headers.get("content-type", "").split(";")[0].strip()

    if content_type not in allowed_types:
        raise HTTPException(
            status_code=415,
            detail=f"Content-Type must be one of: {', '.join(allowed_types)}"
        )

    return content_type


# Export commonly used dependencies
__all__ = [
    "HealthChecker",
    "get_health_checker",
    "get_request_id",
    "get_correlation_id",
    "get_tenant_id",
    "require_tenant_id",
    "get_user_agent",
    "get_client_ip",
    "get_auth_token",
    "require_auth_token",
    "get_api_key",
    "require_api_key",
    "request_context",
    "get_pagination_params",
    "RateLimitChecker",
    "get_rate_limiter",
    "validate_content_type",
]