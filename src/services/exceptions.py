"""Service layer exceptions"""


class ServiceError(Exception):
    """Base exception for service layer errors"""

    def __init__(self, message: str, original_error: Exception = None, error_code: str = None):
        super().__init__(message)
        self.original_error = original_error
        self.error_code = error_code or "SERVICE_ERROR"
        self.timestamp = None

        # Import here to avoid circular imports
        from datetime import datetime
        self.timestamp = datetime.utcnow()


class ValidationError(ServiceError):
    """Exception for input validation failures"""

    def __init__(self, message: str, field: str = None, value: str = None):
        super().__init__(message, error_code="VALIDATION_ERROR")
        self.field = field
        self.value = value


class UnauthorizedError(ServiceError):
    """Exception for authorization failures"""

    def __init__(self, message: str, user_id: str = None, resource: str = None):
        super().__init__(message, error_code="UNAUTHORIZED")
        self.user_id = user_id
        self.resource = resource


class NotFoundError(ServiceError):
    """Exception for resource not found errors"""

    def __init__(self, message: str, resource_type: str = None, resource_id: str = None):
        super().__init__(message, error_code="NOT_FOUND")
        self.resource_type = resource_type
        self.resource_id = resource_id


class ConflictError(ServiceError):
    """Exception for resource conflict errors"""

    def __init__(self, message: str, resource_type: str = None, conflict_field: str = None):
        super().__init__(message, error_code="CONFLICT")
        self.resource_type = resource_type
        self.conflict_field = conflict_field


class RateLimitError(ServiceError):
    """Exception for rate limit exceeded errors"""

    def __init__(self, message: str, limit: int = None, reset_time: int = None):
        super().__init__(message, error_code="RATE_LIMIT_EXCEEDED")
        self.limit = limit
        self.reset_time = reset_time


class ExternalServiceError(ServiceError):
    """Exception for external service failures"""

    def __init__(self, message: str, service_name: str = None, status_code: int = None):
        super().__init__(message, error_code="EXTERNAL_SERVICE_ERROR")
        self.service_name = service_name
        self.status_code = status_code


class ConfigurationError(ServiceError):
    """Exception for configuration errors"""

    def __init__(self, message: str, config_key: str = None):
        super().__init__(message, error_code="CONFIGURATION_ERROR")
        self.config_key = config_key


class ProcessingError(ServiceError):
    """Exception for message processing errors"""

    def __init__(self, message: str, stage: str = None, processor: str = None):
        super().__init__(message, error_code="PROCESSING_ERROR")
        self.stage = stage
        self.processor = processor


class DeliveryError(ServiceError):
    """Exception for message delivery errors"""

    def __init__(self, message: str, channel: str = None, recipient: str = None):
        super().__init__(message, error_code="DELIVERY_ERROR")
        self.channel = channel
        self.recipient = recipient


class SessionError(ServiceError):
    """Exception for session management errors"""

    def __init__(self, message: str, session_id: str = None, operation: str = None):
        super().__init__(message, error_code="SESSION_ERROR")
        self.session_id = session_id
        self.operation = operation


class IntegrationError(ServiceError):
    """Exception for integration-related errors"""

    def __init__(self, message: str, integration_name: str = None, endpoint: str = None):
        super().__init__(message, error_code="INTEGRATION_ERROR")
        self.integration_name = integration_name
        self.endpoint = endpoint


class ModelError(ServiceError):
    """Exception for AI model-related errors"""

    def __init__(self, message: str, model_name: str = None, provider: str = None):
        super().__init__(message, error_code="MODEL_ERROR")
        self.model_name = model_name
        self.provider = provider


class QuotaExceededError(ServiceError):
    """Exception for quota/limit exceeded errors"""

    def __init__(self, message: str, quota_type: str = None, current_usage: int = None, limit: int = None):
        super().__init__(message, error_code="QUOTA_EXCEEDED")
        self.quota_type = quota_type
        self.current_usage = current_usage
        self.limit = limit


class CircuitBreakerError(ServiceError):
    """Exception for circuit breaker open state"""

    def __init__(self, message: str, service_name: str = None, failure_count: int = None):
        super().__init__(message, error_code="CIRCUIT_BREAKER_OPEN")
        self.service_name = service_name
        self.failure_count = failure_count


class TimeoutError(ServiceError):
    """Exception for operation timeout"""

    def __init__(self, message: str, operation: str = None, timeout_seconds: int = None):
        super().__init__(message, error_code="TIMEOUT_ERROR")
        self.operation = operation
        self.timeout_seconds = timeout_seconds


# Convenience function to determine if an error is retryable
def is_retryable_error(error: Exception) -> bool:
    """
    Determine if an error should be retried

    Args:
        error: Exception to check

    Returns:
        True if the error is typically transient and retryable
    """
    retryable_errors = (
        TimeoutError,
        ExternalServiceError,
        CircuitBreakerError,
        RateLimitError
    )

    return isinstance(error, retryable_errors)


# Function to extract error details for logging
def extract_error_details(error: Exception) -> dict:
    """
    Extract structured error details for logging

    Args:
        error: Exception to extract details from

    Returns:
        Dictionary with error details
    """
    details = {
        "error_type": type(error).__name__,
        "error_message": str(error),
        "is_retryable": is_retryable_error(error)
    }

    # Add specific attributes for known error types
    if hasattr(error, 'error_code'):
        details["error_code"] = error.error_code

    if hasattr(error, 'timestamp'):
        details["timestamp"] = error.timestamp.isoformat() if error.timestamp else None

    if isinstance(error, ValidationError):
        details.update({
            "field": getattr(error, 'field', None),
            "value": getattr(error, 'value', None)
        })

    elif isinstance(error, UnauthorizedError):
        details.update({
            "user_id": getattr(error, 'user_id', None),
            "resource": getattr(error, 'resource', None)
        })

    elif isinstance(error, NotFoundError):
        details.update({
            "resource_type": getattr(error, 'resource_type', None),
            "resource_id": getattr(error, 'resource_id', None)
        })

    elif isinstance(error, ExternalServiceError):
        details.update({
            "service_name": getattr(error, 'service_name', None),
            "status_code": getattr(error, 'status_code', None)
        })

    elif isinstance(error, RateLimitError):
        details.update({
            "limit": getattr(error, 'limit', None),
            "reset_time": getattr(error, 'reset_time', None)
        })

    elif isinstance(error, DeliveryError):
        details.update({
            "channel": getattr(error, 'channel', None),
            "recipient": getattr(error, 'recipient', None)
        })

    elif isinstance(error, ProcessingError):
        details.update({
            "stage": getattr(error, 'stage', None),
            "processor": getattr(error, 'processor', None)
        })

    return details