"""
API Middleware Package
Provides middleware components for the Chat Service API.
"""

from .auth_middleware import (
    AuthContext,
    APIKeyAuthContext,
    get_auth_context,
    get_optional_auth_context,
    get_api_key_auth_context,
    get_auth_context_flexible,
    verify_jwt_token,
    validate_tenant_access,
    require_permissions,
    require_role,
    require_scopes
)

from .rate_limit_middleware import (
    RateLimitMiddleware,
    check_rate_limit,
    check_api_key_rate_limit,
    check_webhook_rate_limit,
    check_bulk_operation_rate_limit,
    get_client_ip,
    AdaptiveRateLimiter,
    RATE_LIMITS
)

from .tenant_middleware import (
    TenantContext,
    get_tenant_context,
    validate_tenant_feature,
    require_feature,
    require_plan,
    check_quota_usage,
    require_quota,
    validate_cross_tenant_access,
    DataResidencyValidator,
    require_data_residency_compliance,
    TenantIsolationMiddleware
)

from .error_handler import (
    ErrorHandlingMiddleware,
    validation_exception_handler,
    http_exception_handler,
    test_error_handling
)

from .logging_middleware import (
    LoggingMiddleware,
    PerformanceLoggingMiddleware,
    AuditLoggingMiddleware,
    configure_structured_logging
)

__all__ = [
    # Authentication Middleware
    "AuthContext",
    "APIKeyAuthContext",
    "get_auth_context",
    "get_optional_auth_context",
    "get_api_key_auth_context",
    "get_auth_context_flexible",
    "verify_jwt_token",
    "validate_tenant_access",
    "require_permissions",
    "require_role",
    "require_scopes",

    # Rate Limiting Middleware
    "RateLimitMiddleware",
    "check_rate_limit",
    "check_api_key_rate_limit",
    "check_webhook_rate_limit",
    "check_bulk_operation_rate_limit",
    "get_client_ip",
    "AdaptiveRateLimiter",
    "RATE_LIMITS",

    # Tenant Middleware
    "TenantContext",
    "get_tenant_context",
    "validate_tenant_feature",
    "require_feature",
    "require_plan",
    "check_quota_usage",
    "require_quota",
    "validate_cross_tenant_access",
    "DataResidencyValidator",
    "require_data_residency_compliance",
    "TenantIsolationMiddleware",

    # Error Handling Middleware
    "ErrorHandlingMiddleware",
    "validation_exception_handler",
    "http_exception_handler",
    "test_error_handling",

    # Logging Middleware
    "LoggingMiddleware",
    "PerformanceLoggingMiddleware",
    "AuditLoggingMiddleware",
    "configure_structured_logging"
]

# Middleware application order (important for proper functionality)
MIDDLEWARE_ORDER = [
    "LoggingMiddleware",  # First: Log all requests
    "ErrorHandlingMiddleware",  # Second: Handle all errors
    "TenantIsolationMiddleware",  # Third: Validate tenant access
    "RateLimitMiddleware",  # Fourth: Check rate limits
    "PerformanceLoggingMiddleware",  # Fifth: Monitor performance
    "AuditLoggingMiddleware"  # Last: Audit sensitive operations
]


def create_middleware_stack(
        app,
        include_logging: bool = True,
        include_error_handling: bool = True,
        include_tenant_isolation: bool = True,
        include_rate_limiting: bool = True,
        include_performance_logging: bool = False,
        include_audit_logging: bool = True,
        logging_config: dict = None,
        rate_limit_config: dict = None
):
    """
    Create a complete middleware stack for the application

    Args:
        app: FastAPI application instance
        include_logging: Whether to include request/response logging
        include_error_handling: Whether to include error handling
        include_tenant_isolation: Whether to include tenant isolation
        include_rate_limiting: Whether to include rate limiting
        include_performance_logging: Whether to include performance logging
        include_audit_logging: Whether to include audit logging
        logging_config: Configuration for logging middleware
        rate_limit_config: Configuration for rate limiting middleware

    Returns:
        List of middleware instances in correct order
    """
    middleware_stack = []

    # Apply middleware in reverse order (FastAPI applies them in LIFO order)
    if include_audit_logging:
        middleware_stack.append(AuditLoggingMiddleware(app))

    if include_performance_logging:
        middleware_stack.append(PerformanceLoggingMiddleware(app))

    if include_rate_limiting:
        config = rate_limit_config or {}
        middleware_stack.append(RateLimitMiddleware(app, **config))

    if include_tenant_isolation:
        middleware_stack.append(TenantIsolationMiddleware(app))

    if include_error_handling:
        middleware_stack.append(ErrorHandlingMiddleware(app))

    if include_logging:
        config = logging_config or {}
        middleware_stack.append(LoggingMiddleware(app, **config))

    return middleware_stack


def setup_exception_handlers(app):
    """
    Setup exception handlers for the FastAPI application

    Args:
        app: FastAPI application instance
    """
    from fastapi.exceptions import RequestValidationError
    from fastapi import HTTPException

    app.add_exception_handler(RequestValidationError, validation_exception_handler)
    app.add_exception_handler(HTTPException, http_exception_handler)


# Default middleware configuration
DEFAULT_MIDDLEWARE_CONFIG = {
    "logging": {
        "exclude_paths": {"/health", "/metrics", "/docs", "/openapi.json"},
        "include_request_body": False,
        "include_response_body": False,
        "sanitize_headers": True,
        "max_body_size": 1024
    },
    "rate_limiting": {
        "default_tier": "standard"
    },
    "performance": {
        "slow_request_threshold": 1.0,
        "enable_detailed_timing": False
    },
    "audit": {
        "audit_paths": {
            "/api/v2/chat/message",
            "/api/v2/conversations",
            "/api/v2/integrations",
            "/api/v2/config"
        },
        "audit_methods": {"POST", "PUT", "PATCH", "DELETE"}
    }
}