"""
API Response Package
Provides standardized response models and utilities for the Chat Service API.
"""

from .api_response import (
    APIResponse,
    PaginationMeta,
    PaginatedResponse,
    create_success_response,
    create_error_response,
    create_paginated_response,
    create_validation_error_response,
    create_not_found_response,
    create_unauthorized_response,
    create_forbidden_response
)

from .error_responses import (
    ErrorDetail,
    ErrorResponse,
    ValidationErrorResponse,
    AuthenticationErrorResponse,
    AuthorizationErrorResponse,
    NotFoundErrorResponse,
    RateLimitErrorResponse,
    ConflictErrorResponse,
    InternalErrorResponse,
    ServiceUnavailableErrorResponse,
    TimeoutErrorResponse,
    BadRequestErrorResponse,
    PayloadTooLargeErrorResponse,
    UnsupportedMediaTypeErrorResponse,
    TenantErrorResponse,
    QuotaExceededErrorResponse,
    MaintenanceErrorResponse,
    create_validation_error,
    create_authentication_error,
    create_authorization_error,
    create_not_found_error,
    create_rate_limit_error,
    create_internal_error
)

__all__ = [
    # API Response
    "APIResponse",
    "PaginationMeta",
    "PaginatedResponse",
    "create_success_response",
    "create_error_response",
    "create_paginated_response",
    "create_validation_error_response",
    "create_not_found_response",
    "create_unauthorized_response",
    "create_forbidden_response",

    # Error Responses
    "ErrorDetail",
    "ErrorResponse",
    "ValidationErrorResponse",
    "AuthenticationErrorResponse",
    "AuthorizationErrorResponse",
    "NotFoundErrorResponse",
    "RateLimitErrorResponse",
    "ConflictErrorResponse",
    "InternalErrorResponse",
    "ServiceUnavailableErrorResponse",
    "TimeoutErrorResponse",
    "BadRequestErrorResponse",
    "PayloadTooLargeErrorResponse",
    "UnsupportedMediaTypeErrorResponse",
    "TenantErrorResponse",
    "QuotaExceededErrorResponse",
    "MaintenanceErrorResponse",
    "create_validation_error",
    "create_authentication_error",
    "create_authorization_error",
    "create_not_found_error",
    "create_rate_limit_error",
    "create_internal_error"
]