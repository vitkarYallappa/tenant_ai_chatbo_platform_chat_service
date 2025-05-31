"""
Error Response Models
Provides structured error response models for different error types.
"""

from datetime import datetime
from typing import Optional, Dict, Any, List
from pydantic import BaseModel, Field


class ErrorDetail(BaseModel):
    """Individual error detail"""

    field: Optional[str] = Field(None, description="Field that caused the error")
    message: str = Field(..., description="Error message")
    code: Optional[str] = Field(None, description="Specific error code")
    value: Optional[Any] = Field(None, description="Invalid value that caused the error")


class ErrorResponse(BaseModel):
    """Base error response structure"""

    code: str = Field(..., description="Error code identifier")
    message: str = Field(..., description="Human-readable error message")
    timestamp: datetime = Field(default_factory=datetime.utcnow, description="Error timestamp")
    trace_id: Optional[str] = Field(None, description="Trace ID for debugging")
    details: Optional[Dict[str, Any]] = Field(None, description="Additional error details")

    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }


class ValidationErrorResponse(ErrorResponse):
    """Validation error response"""

    code: str = Field(default="VALIDATION_ERROR", description="Validation error code")
    errors: List[ErrorDetail] = Field(default_factory=list, description="List of validation errors")

    def add_error(
            self,
            message: str,
            field: Optional[str] = None,
            code: Optional[str] = None,
            value: Optional[Any] = None
    ):
        """Add a validation error to the response"""
        self.errors.append(ErrorDetail(
            field=field,
            message=message,
            code=code,
            value=value
        ))


class AuthenticationErrorResponse(ErrorResponse):
    """Authentication error response"""

    code: str = Field(default="AUTHENTICATION_FAILED", description="Authentication error code")
    auth_type: Optional[str] = Field(None, description="Expected authentication type")
    realm: Optional[str] = Field(None, description="Authentication realm")


class AuthorizationErrorResponse(ErrorResponse):
    """Authorization error response"""

    code: str = Field(default="AUTHORIZATION_FAILED", description="Authorization error code")
    required_permissions: Optional[List[str]] = Field(None, description="Required permissions")
    user_permissions: Optional[List[str]] = Field(None, description="User's current permissions")


class NotFoundErrorResponse(ErrorResponse):
    """Resource not found error response"""

    code: str = Field(default="RESOURCE_NOT_FOUND", description="Not found error code")
    resource_type: str = Field(..., description="Type of resource that was not found")
    resource_id: Optional[str] = Field(None, description="ID of the resource that was not found")
    search_criteria: Optional[Dict[str, Any]] = Field(None, description="Search criteria used")


class RateLimitErrorResponse(ErrorResponse):
    """Rate limit exceeded error response"""

    code: str = Field(default="RATE_LIMIT_EXCEEDED", description="Rate limit error code")
    limit: int = Field(..., description="Rate limit threshold")
    window_seconds: int = Field(..., description="Rate limit window in seconds")
    retry_after_seconds: int = Field(..., description="Seconds to wait before retrying")
    current_usage: Optional[int] = Field(None, description="Current usage count")


class ConflictErrorResponse(ErrorResponse):
    """Resource conflict error response"""

    code: str = Field(default="RESOURCE_CONFLICT", description="Conflict error code")
    conflicting_resource: Optional[str] = Field(None, description="Conflicting resource identifier")
    conflict_reason: Optional[str] = Field(None, description="Reason for the conflict")


class InternalErrorResponse(ErrorResponse):
    """Internal server error response"""

    code: str = Field(default="INTERNAL_SERVER_ERROR", description="Internal error code")
    error_id: Optional[str] = Field(None, description="Internal error identifier for tracking")
    support_contact: Optional[str] = Field(None, description="Support contact information")


class ServiceUnavailableErrorResponse(ErrorResponse):
    """Service unavailable error response"""

    code: str = Field(default="SERVICE_UNAVAILABLE", description="Service unavailable error code")
    service_name: Optional[str] = Field(None, description="Name of the unavailable service")
    estimated_recovery_time: Optional[datetime] = Field(None, description="Estimated recovery time")
    retry_after_seconds: Optional[int] = Field(None, description="Suggested retry delay")


class TimeoutErrorResponse(ErrorResponse):
    """Request timeout error response"""

    code: str = Field(default="REQUEST_TIMEOUT", description="Timeout error code")
    timeout_seconds: int = Field(..., description="Timeout threshold in seconds")
    operation: Optional[str] = Field(None, description="Operation that timed out")


class BadRequestErrorResponse(ErrorResponse):
    """Bad request error response"""

    code: str = Field(default="BAD_REQUEST", description="Bad request error code")
    invalid_parameters: Optional[List[str]] = Field(None, description="List of invalid parameters")
    suggestion: Optional[str] = Field(None, description="Suggestion for fixing the request")


class PayloadTooLargeErrorResponse(ErrorResponse):
    """Payload too large error response"""

    code: str = Field(default="PAYLOAD_TOO_LARGE", description="Payload size error code")
    max_size_bytes: int = Field(..., description="Maximum allowed payload size in bytes")
    actual_size_bytes: Optional[int] = Field(None, description="Actual payload size in bytes")


class UnsupportedMediaTypeErrorResponse(ErrorResponse):
    """Unsupported media type error response"""

    code: str = Field(default="UNSUPPORTED_MEDIA_TYPE", description="Media type error code")
    provided_type: Optional[str] = Field(None, description="Media type that was provided")
    supported_types: List[str] = Field(default_factory=list, description="List of supported media types")


class TenantErrorResponse(ErrorResponse):
    """Tenant-related error response"""

    code: str = Field(default="TENANT_ERROR", description="Tenant error code")
    tenant_id: Optional[str] = Field(None, description="Tenant ID related to the error")
    tenant_status: Optional[str] = Field(None, description="Current tenant status")


class QuotaExceededErrorResponse(ErrorResponse):
    """Quota exceeded error response"""

    code: str = Field(default="QUOTA_EXCEEDED", description="Quota exceeded error code")
    quota_type: str = Field(..., description="Type of quota that was exceeded")
    current_usage: int = Field(..., description="Current usage amount")
    quota_limit: int = Field(..., description="Quota limit")
    reset_time: Optional[datetime] = Field(None, description="When the quota resets")


class MaintenanceErrorResponse(ErrorResponse):
    """Maintenance mode error response"""

    code: str = Field(default="MAINTENANCE_MODE", description="Maintenance error code")
    maintenance_message: Optional[str] = Field(None, description="Maintenance notification message")
    expected_duration: Optional[int] = Field(None, description="Expected maintenance duration in seconds")
    maintenance_window: Optional[Dict[str, datetime]] = Field(None, description="Maintenance time window")


# Factory functions for creating common error responses

def create_validation_error(
        errors: List[ErrorDetail],
        message: str = "Request validation failed",
        trace_id: Optional[str] = None
) -> ValidationErrorResponse:
    """Create a validation error response"""
    return ValidationErrorResponse(
        message=message,
        errors=errors,
        trace_id=trace_id
    )


def create_authentication_error(
        message: str = "Authentication required",
        auth_type: Optional[str] = None,
        realm: Optional[str] = None,
        trace_id: Optional[str] = None
) -> AuthenticationErrorResponse:
    """Create an authentication error response"""
    return AuthenticationErrorResponse(
        message=message,
        auth_type=auth_type,
        realm=realm,
        trace_id=trace_id
    )


def create_authorization_error(
        message: str = "Access denied",
        required_permissions: Optional[List[str]] = None,
        user_permissions: Optional[List[str]] = None,
        trace_id: Optional[str] = None
) -> AuthorizationErrorResponse:
    """Create an authorization error response"""
    return AuthorizationErrorResponse(
        message=message,
        required_permissions=required_permissions,
        user_permissions=user_permissions,
        trace_id=trace_id
    )


def create_not_found_error(
        resource_type: str,
        resource_id: Optional[str] = None,
        search_criteria: Optional[Dict[str, Any]] = None,
        trace_id: Optional[str] = None
) -> NotFoundErrorResponse:
    """Create a not found error response"""
    message = f"{resource_type.title()} not found"
    if resource_id:
        message += f" with ID: {resource_id}"

    return NotFoundErrorResponse(
        message=message,
        resource_type=resource_type,
        resource_id=resource_id,
        search_criteria=search_criteria,
        trace_id=trace_id
    )


def create_rate_limit_error(
        limit: int,
        window_seconds: int,
        retry_after_seconds: int,
        current_usage: Optional[int] = None,
        trace_id: Optional[str] = None
) -> RateLimitErrorResponse:
    """Create a rate limit error response"""
    return RateLimitErrorResponse(
        message=f"Rate limit exceeded. {limit} requests per {window_seconds} seconds",
        limit=limit,
        window_seconds=window_seconds,
        retry_after_seconds=retry_after_seconds,
        current_usage=current_usage,
        trace_id=trace_id
    )


def create_internal_error(
        message: str = "An internal server error occurred",
        error_id: Optional[str] = None,
        support_contact: Optional[str] = None,
        trace_id: Optional[str] = None
) -> InternalErrorResponse:
    """Create an internal server error response"""
    return InternalErrorResponse(
        message=message,
        error_id=error_id,
        support_contact=support_contact,
        trace_id=trace_id
    )