"""
Core exceptions for the chatbot platform business logic layer.

This module defines custom exceptions used throughout the core business logic,
including channel operations, message processing, and content validation.
"""

from typing import Dict, Any, Optional, List
from datetime import datetime


class CoreError(Exception):
    """Base exception for all core business logic errors."""

    def __init__(
            self,
            message: str,
            error_code: Optional[str] = None,
            details: Optional[Dict[str, Any]] = None,
            context: Optional[Dict[str, Any]] = None
    ):
        super().__init__(message)
        self.message = message
        self.error_code = error_code or self.__class__.__name__
        self.details = details or {}
        self.context = context or {}
        self.timestamp = datetime.utcnow()

    def to_dict(self) -> Dict[str, Any]:
        """Convert exception to dictionary for logging/serialization."""
        return {
            "error_type": self.__class__.__name__,
            "error_code": self.error_code,
            "message": self.message,
            "details": self.details,
            "context": self.context,
            "timestamp": self.timestamp.isoformat()
        }


class ChannelError(CoreError):
    """Base exception for channel-related operations."""
    pass


class ChannelConnectionError(ChannelError):
    """Raised when channel fails to connect to external service."""

    def __init__(
            self,
            channel: str,
            message: str,
            status_code: Optional[int] = None,
            response_body: Optional[str] = None
    ):
        super().__init__(
            message=f"Channel {channel} connection failed: {message}",
            error_code="CHANNEL_CONNECTION_ERROR",
            details={
                "channel": channel,
                "status_code": status_code,
                "response_body": response_body
            }
        )
        self.channel = channel
        self.status_code = status_code


class ChannelConfigurationError(ChannelError):
    """Raised when channel configuration is invalid."""

    def __init__(self, channel: str, config_issue: str):
        super().__init__(
            message=f"Channel {channel} configuration error: {config_issue}",
            error_code="CHANNEL_CONFIG_ERROR",
            details={"channel": channel, "config_issue": config_issue}
        )


class ChannelRateLimitError(ChannelError):
    """Raised when channel rate limit is exceeded."""

    def __init__(
            self,
            channel: str,
            limit: int,
            window: str,
            retry_after: Optional[int] = None
    ):
        super().__init__(
            message=f"Channel {channel} rate limit exceeded: {limit} requests per {window}",
            error_code="CHANNEL_RATE_LIMIT",
            details={
                "channel": channel,
                "limit": limit,
                "window": window,
                "retry_after": retry_after
            }
        )
        self.retry_after = retry_after


class ChannelDeliveryError(ChannelError):
    """Raised when message delivery through channel fails."""

    def __init__(
            self,
            channel: str,
            recipient: str,
            delivery_error: str,
            is_permanent: bool = False
    ):
        super().__init__(
            message=f"Message delivery failed via {channel} to {recipient}: {delivery_error}",
            error_code="CHANNEL_DELIVERY_ERROR",
            details={
                "channel": channel,
                "recipient": recipient,
                "delivery_error": delivery_error,
                "is_permanent": is_permanent
            }
        )
        self.is_permanent = is_permanent


class ProcessingError(CoreError):
    """Base exception for message processing operations."""
    pass


class ProcessorNotFoundError(ProcessingError):
    """Raised when no suitable processor is found for content type."""

    def __init__(self, content_type: str, available_processors: List[str]):
        super().__init__(
            message=f"No processor found for content type: {content_type}",
            error_code="PROCESSOR_NOT_FOUND",
            details={
                "content_type": content_type,
                "available_processors": available_processors
            }
        )


class ProcessingTimeoutError(ProcessingError):
    """Raised when message processing exceeds timeout."""

    def __init__(self, processor: str, timeout_seconds: int):
        super().__init__(
            message=f"Processing timeout in {processor} after {timeout_seconds}s",
            error_code="PROCESSING_TIMEOUT",
            details={
                "processor": processor,
                "timeout_seconds": timeout_seconds
            }
        )


class ProcessingValidationError(ProcessingError):
    """Raised when input validation fails during processing."""

    def __init__(
            self,
            processor: str,
            validation_errors: List[str],
            input_data: Optional[Dict[str, Any]] = None
    ):
        super().__init__(
            message=f"Validation failed in {processor}: {'; '.join(validation_errors)}",
            error_code="PROCESSING_VALIDATION_ERROR",
            details={
                "processor": processor,
                "validation_errors": validation_errors,
                "input_data": input_data
            }
        )


class ContentNormalizationError(ProcessingError):
    """Raised when content normalization fails."""

    def __init__(self, normalizer: str, content_type: str, error_details: str):
        super().__init__(
            message=f"Content normalization failed in {normalizer} for {content_type}: {error_details}",
            error_code="CONTENT_NORMALIZATION_ERROR",
            details={
                "normalizer": normalizer,
                "content_type": content_type,
                "error_details": error_details
            }
        )


class PipelineError(CoreError):
    """Base exception for pipeline processing operations."""
    pass


class PipelineStageError(PipelineError):
    """Raised when a pipeline stage fails."""

    def __init__(
            self,
            stage_name: str,
            stage_error: str,
            stage_index: int,
            partial_results: Optional[Dict[str, Any]] = None
    ):
        super().__init__(
            message=f"Pipeline stage '{stage_name}' failed: {stage_error}",
            error_code="PIPELINE_STAGE_ERROR",
            details={
                "stage_name": stage_name,
                "stage_error": stage_error,
                "stage_index": stage_index,
                "partial_results": partial_results
            }
        )
        self.stage_name = stage_name
        self.stage_index = stage_index


class PipelineConfigurationError(PipelineError):
    """Raised when pipeline configuration is invalid."""

    def __init__(self, config_issue: str):
        super().__init__(
            message=f"Pipeline configuration error: {config_issue}",
            error_code="PIPELINE_CONFIG_ERROR",
            details={"config_issue": config_issue}
        )


class SecurityError(CoreError):
    """Base exception for security-related issues."""
    pass


class ContentSecurityError(SecurityError):
    """Raised when content fails security validation."""

    def __init__(
            self,
            security_issue: str,
            content_hash: Optional[str] = None,
            risk_level: str = "medium"
    ):
        super().__init__(
            message=f"Content security violation: {security_issue}",
            error_code="CONTENT_SECURITY_ERROR",
            details={
                "security_issue": security_issue,
                "content_hash": content_hash,
                "risk_level": risk_level
            }
        )


class TenantSecurityError(SecurityError):
    """Raised when tenant security policies are violated."""

    def __init__(self, tenant_id: str, security_violation: str):
        super().__init__(
            message=f"Tenant {tenant_id} security violation: {security_violation}",
            error_code="TENANT_SECURITY_ERROR",
            details={
                "tenant_id": tenant_id,
                "security_violation": security_violation
            }
        )


class ValidationError(CoreError):
    """Raised when input validation fails."""

    def __init__(
            self,
            field: str,
            value: Any,
            validation_rule: str,
            expected_format: Optional[str] = None
    ):
        super().__init__(
            message=f"Validation failed for field '{field}': {validation_rule}",
            error_code="VALIDATION_ERROR",
            details={
                "field": field,
                "value": str(value),
                "validation_rule": validation_rule,
                "expected_format": expected_format
            }
        )


class ExternalServiceError(CoreError):
    """Raised when external service integration fails."""

    def __init__(
            self,
            service: str,
            operation: str,
            error_message: str,
            status_code: Optional[int] = None,
            is_retryable: bool = True
    ):
        super().__init__(
            message=f"External service {service} failed during {operation}: {error_message}",
            error_code="EXTERNAL_SERVICE_ERROR",
            details={
                "service": service,
                "operation": operation,
                "error_message": error_message,
                "status_code": status_code,
                "is_retryable": is_retryable
            }
        )
        self.is_retryable = is_retryable


class ResourceExhaustedError(CoreError):
    """Raised when system resources are exhausted."""

    def __init__(
            self,
            resource_type: str,
            current_usage: str,
            limit: str
    ):
        super().__init__(
            message=f"Resource exhausted: {resource_type} usage {current_usage} exceeds limit {limit}",
            error_code="RESOURCE_EXHAUSTED",
            details={
                "resource_type": resource_type,
                "current_usage": current_usage,
                "limit": limit
            }
        )


class ConfigurationError(CoreError):
    """Raised when system configuration is invalid."""

    def __init__(self, component: str, config_issue: str):
        super().__init__(
            message=f"Configuration error in {component}: {config_issue}",
            error_code="CONFIGURATION_ERROR",
            details={
                "component": component,
                "config_issue": config_issue
            }
        )


# Convenience functions for common error scenarios
def channel_unavailable(channel: str, reason: str) -> ChannelConnectionError:
    """Create a channel unavailable error."""
    return ChannelConnectionError(
        channel=channel,
        message=f"Channel temporarily unavailable: {reason}",
        status_code=503
    )


def invalid_message_format(channel: str, format_issue: str) -> ValidationError:
    """Create an invalid message format error."""
    return ValidationError(
        field="message_content",
        value="<message_content>",
        validation_rule=f"Invalid format for {channel}: {format_issue}",
        expected_format=f"Valid {channel} message format"
    )


def processing_failed(processor: str, reason: str) -> ProcessingError:
    """Create a general processing failure error."""
    return ProcessingError(
        message=f"Processing failed in {processor}: {reason}",
        error_code="PROCESSING_FAILED",
        details={"processor": processor, "reason": reason}
    )


def quota_exceeded(resource: str, usage: int, limit: int) -> ResourceExhaustedError:
    """Create a quota exceeded error."""
    return ResourceExhaustedError(
        resource_type=resource,
        current_usage=str(usage),
        limit=str(limit)
    )