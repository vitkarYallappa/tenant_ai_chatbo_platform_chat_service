"""
Base Service Class

Abstract base class for all services providing common patterns and utilities
including dependency injection, logging, error handling, and tenant validation.
"""

from abc import ABC
from typing import Dict, Any, Optional
from datetime import datetime, UTC
import structlog

from src.models.types import TenantId, UserId
from src.services.exceptions import ServiceError, UnauthorizedError, ValidationError


class BaseService(ABC):
    """Abstract base class for all services"""

    def __init__(self):
        self.logger = structlog.get_logger(self.__class__.__name__)
        self.service_name = self.__class__.__name__

    async def validate_tenant_access(
            self,
            tenant_id: TenantId,
            user_context: Optional[Dict[str, Any]] = None
    ) -> bool:
        """
        Validate that user has access to tenant resources

        Args:
            tenant_id: Tenant ID to validate access for
            user_context: User context with permissions

        Returns:
            True if access is allowed

        Raises:
            UnauthorizedError: If access is denied
        """
        try:
            if not tenant_id:
                raise ValidationError("Tenant ID is required")

            if user_context:
                user_tenant_id = user_context.get("tenant_id")
                if user_tenant_id and user_tenant_id != tenant_id:
                    self.logger.warning(
                        "Tenant access denied",
                        requested_tenant=tenant_id,
                        user_tenant=user_tenant_id,
                        user_id=user_context.get("user_id")
                    )
                    raise UnauthorizedError(f"Access denied to tenant {tenant_id}")

            return True

        except (UnauthorizedError, ValidationError):
            raise
        except Exception as e:
            self.logger.error(
                "Tenant access validation failed",
                tenant_id=tenant_id,
                error=str(e)
            )
            raise ServiceError(f"Failed to validate tenant access: {e}")

    def log_operation(
            self,
            operation: str,
            tenant_id: Optional[TenantId] = None,
            user_id: Optional[UserId] = None,
            **kwargs
    ) -> None:
        """Log service operation with standard fields"""
        log_data = {
            "service": self.service_name,
            "operation": operation,
            "timestamp": datetime.now(UTC).isoformat(),
            **kwargs
        }

        if tenant_id:
            log_data["tenant_id"] = tenant_id
        if user_id:
            log_data["user_id"] = user_id

        self.logger.info("Service operation", **log_data)

    def handle_service_error(
            self,
            error: Exception,
            operation: str,
            tenant_id: Optional[TenantId] = None,
            **context
    ) -> ServiceError:
        """
        Handle and wrap service errors with context

        Args:
            error: Original exception
            operation: Operation that failed
            tenant_id: Optional tenant ID
            **context: Additional context

        Returns:
            ServiceError with wrapped exception
        """
        error_context = {
            "service": self.service_name,
            "operation": operation,
            "error_type": type(error).__name__,
            "error_message": str(error),
            **context
        }

        if tenant_id:
            error_context["tenant_id"] = tenant_id

        self.logger.error("Service operation failed", **error_context)

        # Re-raise specific errors
        if isinstance(error, (ValidationError, UnauthorizedError)):
            return error

        # Wrap generic errors
        return ServiceError(
            f"{operation} failed: {str(error)}",
            original_error=error
        )

    async def _validate_required_fields(
            self,
            data: Dict[str, Any],
            required_fields: list
    ) -> None:
        """Validate that required fields are present"""
        missing_fields = [
            field for field in required_fields
            if field not in data or data[field] is None
        ]

        if missing_fields:
            raise ValidationError(
                f"Missing required fields: {', '.join(missing_fields)}"
            )

    def _sanitize_log_data(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Remove sensitive data from log entries"""
        sensitive_fields = [
            "password", "token", "secret", "key", "credential",
            "authorization", "api_key", "private_key"
        ]

        sanitized = {}
        for key, value in data.items():
            if any(sensitive in key.lower() for sensitive in sensitive_fields):
                sanitized[key] = "[REDACTED]"
            elif isinstance(value, dict):
                sanitized[key] = self._sanitize_log_data(value)
            else:
                sanitized[key] = value

        return sanitized

    async def _check_rate_limit(
            self,
            tenant_id: TenantId,
            operation: str,
            identifier: str,
            limit: int,
            window_seconds: int = 60
    ) -> bool:
        """
        Check if operation is within rate limits

        Args:
            tenant_id: Tenant identifier
            operation: Operation type for rate limiting
            identifier: Unique identifier (user_id, api_key, etc.)
            limit: Maximum requests allowed
            window_seconds: Time window in seconds

        Returns:
            True if within limits, False if exceeded
        """
        try:
            # This would typically use Redis for distributed rate limiting
            # For now, we'll implement a simple in-memory check
            # In production, this should use Redis sorted sets or similar

            self.logger.debug(
                "Rate limit check",
                tenant_id=tenant_id,
                operation=operation,
                identifier=identifier,
                limit=limit,
                window_seconds=window_seconds
            )

            # TODO: Implement actual rate limiting logic with Redis
            # For now, always return True (no rate limiting)
            return True

        except Exception as e:
            self.logger.error(
                "Rate limit check failed",
                tenant_id=tenant_id,
                operation=operation,
                error=str(e)
            )
            # On error, allow the request (fail open)
            return True

    async def _get_tenant_configuration(
            self,
            tenant_id: TenantId,
            config_type: str
    ) -> Optional[Dict[str, Any]]:
        """
        Get tenant-specific configuration

        Args:
            tenant_id: Tenant identifier
            config_type: Type of configuration to retrieve

        Returns:
            Configuration dictionary or None if not found
        """
        try:
            # This would typically fetch from a configuration repository
            # For now, return empty dict as default

            self.logger.debug(
                "Fetching tenant configuration",
                tenant_id=tenant_id,
                config_type=config_type
            )

            # TODO: Implement actual configuration fetching
            # Should integrate with tenant configuration repository
            return {}

        except Exception as e:
            self.logger.error(
                "Failed to fetch tenant configuration",
                tenant_id=tenant_id,
                config_type=config_type,
                error=str(e)
            )
            return None

    def _format_error_response(
            self,
            error: Exception,
            request_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Format error for API response

        Args:
            error: Exception to format
            request_id: Optional request identifier

        Returns:
            Formatted error dictionary
        """
        error_response = {
            "error": {
                "type": type(error).__name__,
                "message": str(error),
                "timestamp": datetime.now(UTC).isoformat()
            }
        }

        if request_id:
            error_response["error"]["request_id"] = request_id

        # Add specific error details for known error types
        if hasattr(error, 'error_code'):
            error_response["error"]["code"] = error.error_code

        if hasattr(error, 'field'):
            error_response["error"]["field"] = error.field

        if hasattr(error, 'details'):
            error_response["error"]["details"] = error.details

        return error_response

    async def _execute_with_retry(
            self,
            operation_func,
            max_retries: int = 3,
            backoff_factor: float = 1.0,
            *args,
            **kwargs
    ) -> Any:
        """
        Execute operation with retry logic

        Args:
            operation_func: Function to execute
            max_retries: Maximum number of retry attempts
            backoff_factor: Backoff multiplier for delay
            *args: Arguments for operation function
            **kwargs: Keyword arguments for operation function

        Returns:
            Operation result

        Raises:
            Last exception if all retries fail
        """
        import asyncio

        last_exception = None

        for attempt in range(max_retries + 1):
            try:
                if asyncio.iscoroutinefunction(operation_func):
                    return await operation_func(*args, **kwargs)
                else:
                    return operation_func(*args, **kwargs)
            except Exception as e:
                last_exception = e

                if attempt < max_retries:
                    delay = backoff_factor * (2 ** attempt)

                    self.logger.warning(
                        "Operation failed, retrying",
                        attempt=attempt + 1,
                        max_retries=max_retries,
                        delay_seconds=delay,
                        error=str(e)
                    )

                    await asyncio.sleep(delay)
                else:
                    self.logger.error(
                        "Operation failed after all retries",
                        attempts=max_retries + 1,
                        final_error=str(e)
                    )

        # Re-raise the last exception if all retries failed
        raise last_exception