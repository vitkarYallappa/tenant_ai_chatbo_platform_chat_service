"""
Standard API Response Format
Provides consistent response structure across all endpoints.
"""

from datetime import datetime
from typing import TypeVar, Generic, Optional, Dict, Any, Union
from uuid import uuid4
from pydantic import BaseModel, Field

T = TypeVar('T')


class APIResponse(BaseModel, Generic[T]):
    """Standard API response wrapper"""

    status: str = Field(
        default="success",
        description="Response status",
        regex=r"^(success|error)$"
    )
    data: Optional[T] = Field(
        default=None,
        description="Response data payload"
    )
    meta: Dict[str, Any] = Field(
        default_factory=dict,
        description="Response metadata"
    )
    error: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Error information if status is error"
    )

    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }


class PaginationMeta(BaseModel):
    """Pagination metadata for list responses"""

    page: int = Field(..., ge=1, description="Current page number")
    page_size: int = Field(..., ge=1, le=100, description="Items per page")
    total_items: int = Field(..., ge=0, description="Total number of items")
    total_pages: int = Field(..., ge=0, description="Total number of pages")
    has_next: bool = Field(..., description="Whether there is a next page")
    has_previous: bool = Field(..., description="Whether there is a previous page")


class PaginatedResponse(BaseModel, Generic[T]):
    """Paginated response wrapper"""

    items: list[T] = Field(..., description="List of items")
    pagination: PaginationMeta = Field(..., description="Pagination metadata")


def create_success_response(
        data: Optional[T] = None,
        message: Optional[str] = None,
        request_id: Optional[str] = None,
        processing_time_ms: Optional[int] = None,
        **meta_kwargs
) -> APIResponse[T]:
    """
    Create a successful API response

    Args:
        data: Response data payload
        message: Optional success message
        request_id: Optional request ID for tracing
        processing_time_ms: Optional processing time in milliseconds
        **meta_kwargs: Additional metadata fields

    Returns:
        APIResponse with success status
    """
    meta = {
        "timestamp": datetime.utcnow().isoformat(),
        "api_version": "v2",
        **meta_kwargs
    }

    if message:
        meta["message"] = message
    if request_id:
        meta["request_id"] = request_id
    if processing_time_ms is not None:
        meta["processing_time_ms"] = processing_time_ms

    return APIResponse(
        status="success",
        data=data,
        meta=meta
    )


def create_error_response(
        error_code: str,
        error_message: str,
        error_details: Optional[Dict[str, Any]] = None,
        request_id: Optional[str] = None,
        trace_id: Optional[str] = None,
        **meta_kwargs
) -> APIResponse[None]:
    """
    Create an error API response

    Args:
        error_code: Error code identifier
        error_message: Human-readable error message
        error_details: Optional additional error details
        request_id: Optional request ID for tracing
        trace_id: Optional trace ID for debugging
        **meta_kwargs: Additional metadata fields

    Returns:
        APIResponse with error status
    """
    error_data = {
        "code": error_code,
        "message": error_message,
        "timestamp": datetime.utcnow().isoformat()
    }

    if error_details:
        error_data["details"] = error_details
    if trace_id:
        error_data["trace_id"] = trace_id

    meta = {
        "timestamp": datetime.utcnow().isoformat(),
        "api_version": "v2",
        **meta_kwargs
    }

    if request_id:
        meta["request_id"] = request_id

    return APIResponse(
        status="error",
        error=error_data,
        meta=meta
    )


def create_paginated_response(
        items: list[T],
        page: int,
        page_size: int,
        total_items: int,
        message: Optional[str] = None,
        **meta_kwargs
) -> APIResponse[PaginatedResponse[T]]:
    """
    Create a paginated API response

    Args:
        items: List of items for current page
        page: Current page number
        page_size: Items per page
        total_items: Total number of items across all pages
        message: Optional success message
        **meta_kwargs: Additional metadata fields

    Returns:
        APIResponse with paginated data
    """
    total_pages = (total_items + page_size - 1) // page_size
    has_next = page < total_pages
    has_previous = page > 1

    pagination_meta = PaginationMeta(
        page=page,
        page_size=page_size,
        total_items=total_items,
        total_pages=total_pages,
        has_next=has_next,
        has_previous=has_previous
    )

    paginated_data = PaginatedResponse(
        items=items,
        pagination=pagination_meta
    )

    return create_success_response(
        data=paginated_data,
        message=message,
        **meta_kwargs
    )


def create_validation_error_response(
        validation_errors: Union[str, Dict[str, Any], list],
        request_id: Optional[str] = None
) -> APIResponse[None]:
    """
    Create a validation error response

    Args:
        validation_errors: Validation error details
        request_id: Optional request ID for tracing

    Returns:
        APIResponse with validation error
    """
    if isinstance(validation_errors, str):
        error_details = {"message": validation_errors}
    elif isinstance(validation_errors, list):
        error_details = {"errors": validation_errors}
    else:
        error_details = validation_errors

    return create_error_response(
        error_code="VALIDATION_ERROR",
        error_message="Request validation failed",
        error_details=error_details,
        request_id=request_id
    )


def create_not_found_response(
        resource_type: str,
        resource_id: Optional[str] = None,
        request_id: Optional[str] = None
) -> APIResponse[None]:
    """
    Create a not found error response

    Args:
        resource_type: Type of resource that was not found
        resource_id: Optional ID of the resource
        request_id: Optional request ID for tracing

    Returns:
        APIResponse with not found error
    """
    message = f"{resource_type.title()} not found"
    if resource_id:
        message += f" with ID: {resource_id}"

    error_details = {
        "resource_type": resource_type,
        "resource_id": resource_id
    } if resource_id else {
        "resource_type": resource_type
    }

    return create_error_response(
        error_code="RESOURCE_NOT_FOUND",
        error_message=message,
        error_details=error_details,
        request_id=request_id
    )


def create_unauthorized_response(
        reason: Optional[str] = None,
        request_id: Optional[str] = None
) -> APIResponse[None]:
    """
    Create an unauthorized error response

    Args:
        reason: Optional reason for authorization failure
        request_id: Optional request ID for tracing

    Returns:
        APIResponse with unauthorized error
    """
    message = "Authentication required"
    error_details = {}

    if reason:
        error_details["reason"] = reason
        message = f"Authentication failed: {reason}"

    return create_error_response(
        error_code="AUTHENTICATION_REQUIRED",
        error_message=message,
        error_details=error_details,
        request_id=request_id
    )


def create_forbidden_response(
        reason: Optional[str] = None,
        required_permissions: Optional[list[str]] = None,
        request_id: Optional[str] = None
) -> APIResponse[None]:
    """
    Create a forbidden error response

    Args:
        reason: Optional reason for authorization failure
        required_permissions: Optional list of required permissions
        request_id: Optional request ID for tracing

    Returns:
        APIResponse with forbidden error
    """
    message = "Access denied"
    error_details = {}

    if reason:
        error_details["reason"] = reason
        message = f"Access denied: {reason}"

    if required_permissions:
        error_details["required_permissions"] = required_permissions

    return create_error_response(
        error_code="ACCESS_DENIED",
        error_message=message,
        error_details=error_details,
        request_id=request_id
    )