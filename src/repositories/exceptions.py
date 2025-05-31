"""
Repository-specific exceptions
==============================

Defines a comprehensive hierarchy of exceptions for repository operations,
providing specific error types for different failure scenarios.

Exception Hierarchy:
    RepositoryError (base)
    ├── EntityNotFoundError
    ├── DuplicateEntityError
    ├── ValidationError
    ├── ConnectionError
    ├── TransactionError
    ├── ConcurrencyError
    ├── QueryError
    └── ConfigurationError

Benefits:
- Specific error handling at service layer
- Better error messages for debugging
- Standardized error codes for API responses
- Support for error context and metadata
"""

from typing import Optional, Dict, Any, List
from datetime import datetime
import traceback


class RepositoryError(Exception):
    """
    Base exception for all repository operations

    Provides common functionality for error context, metadata,
    and integration with monitoring systems.
    """

    def __init__(
            self,
            message: str,
            original_error: Optional[Exception] = None,
            error_code: Optional[str] = None,
            context: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize repository error

        Args:
            message: Human-readable error message
            original_error: Original exception that caused this error
            error_code: Machine-readable error code
            context: Additional context about the error
        """
        super().__init__(message)
        self.message = message
        self.original_error = original_error
        self.error_code = error_code or self.__class__.__name__.upper()
        self.context = context or {}
        self.timestamp = datetime.utcnow()
        self.traceback = traceback.format_exc() if original_error else None

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert error to dictionary for serialization

        Returns:
            Dictionary representation of the error
        """
        return {
            "error_type": self.__class__.__name__,
            "error_code": self.error_code,
            "message": self.message,
            "timestamp": self.timestamp.isoformat(),
            "context": self.context,
            "original_error": str(self.original_error) if self.original_error else None
        }

    def add_context(self, key: str, value: Any) -> "RepositoryError":
        """
        Add context information to the error

        Args:
            key: Context key
            value: Context value

        Returns:
            Self for method chaining
        """
        self.context[key] = value
        return self

    def __str__(self) -> str:
        """String representation with context"""
        base_msg = self.message
        if self.context:
            context_str = ", ".join(f"{k}={v}" for k, v in self.context.items())
            base_msg += f" (Context: {context_str})"
        return base_msg


class EntityNotFoundError(RepositoryError):
    """
    Exception raised when a requested entity is not found

    Used for GET operations that return empty results
    when an entity is expected to exist.
    """

    def __init__(
            self,
            entity_type: str,
            entity_id: str,
            additional_filters: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize entity not found error

        Args:
            entity_type: Type of entity that was not found
            entity_id: ID of the entity that was not found
            additional_filters: Additional search criteria used
        """
        self.entity_type = entity_type
        self.entity_id = entity_id
        self.additional_filters = additional_filters or {}

        message = f"{entity_type} with ID '{entity_id}' not found"
        if additional_filters:
            filter_str = ", ".join(f"{k}={v}" for k, v in additional_filters.items())
            message += f" (filters: {filter_str})"

        context = {
            "entity_type": entity_type,
            "entity_id": entity_id,
            "additional_filters": additional_filters
        }

        super().__init__(
            message=message,
            error_code="ENTITY_NOT_FOUND",
            context=context
        )


class DuplicateEntityError(RepositoryError):
    """
    Exception raised when trying to create an entity that already exists

    Used for CREATE operations that violate uniqueness constraints.
    """

    def __init__(
            self,
            entity_type: str,
            conflicting_fields: Dict[str, Any],
            existing_entity_id: Optional[str] = None
    ):
        """
        Initialize duplicate entity error

        Args:
            entity_type: Type of entity that has a duplicate
            conflicting_fields: Fields that caused the conflict
            existing_entity_id: ID of the existing entity if available
        """
        self.entity_type = entity_type
        self.conflicting_fields = conflicting_fields
        self.existing_entity_id = existing_entity_id

        field_str = ", ".join(f"{k}={v}" for k, v in conflicting_fields.items())
        message = f"{entity_type} with {field_str} already exists"

        if existing_entity_id:
            message += f" (existing ID: {existing_entity_id})"

        context = {
            "entity_type": entity_type,
            "conflicting_fields": conflicting_fields,
            "existing_entity_id": existing_entity_id
        }

        super().__init__(
            message=message,
            error_code="DUPLICATE_ENTITY",
            context=context
        )


class ValidationError(RepositoryError):
    """
    Exception raised when entity validation fails

    Used when entity data doesn't meet repository requirements
    before attempting database operations.
    """

    def __init__(
            self,
            message: str,
            field: Optional[str] = None,
            validation_errors: Optional[List[Dict[str, Any]]] = None
    ):
        """
        Initialize validation error

        Args:
            message: Validation error message
            field: Specific field that failed validation
            validation_errors: List of detailed validation errors
        """
        self.field = field
        self.validation_errors = validation_errors or []

        context = {
            "field": field,
            "validation_errors": validation_errors
        }

        super().__init__(
            message=message,
            error_code="VALIDATION_ERROR",
            context=context
        )


class ConnectionError(RepositoryError):
    """
    Exception raised when database connection fails

    Used for network, authentication, or configuration issues
    that prevent database access.
    """

    def __init__(
            self,
            database_type: str,
            connection_string: Optional[str] = None,
            original_error: Optional[Exception] = None
    ):
        """
        Initialize connection error

        Args:
            database_type: Type of database (MongoDB, Redis, PostgreSQL)
            connection_string: Sanitized connection string (no passwords)
            original_error: Original connection exception
        """
        self.database_type = database_type
        self.connection_string = connection_string

        message = f"Failed to connect to {database_type}"
        if connection_string:
            # Sanitize connection string to remove sensitive data
            sanitized = self._sanitize_connection_string(connection_string)
            message += f" at {sanitized}"

        context = {
            "database_type": database_type,
            "connection_string": self._sanitize_connection_string(connection_string) if connection_string else None
        }

        super().__init__(
            message=message,
            original_error=original_error,
            error_code="CONNECTION_ERROR",
            context=context
        )

    def _sanitize_connection_string(self, connection_string: str) -> str:
        """Remove sensitive information from connection string"""
        # Remove passwords and API keys from connection strings
        import re
        # Remove password from MongoDB/PostgreSQL URLs
        sanitized = re.sub(r'://([^:]+):([^@]+)@', r'://\1:***@', connection_string)
        # Remove API keys and tokens
        sanitized = re.sub(r'(password|token|key|secret)=([^&\s]+)', r'\1=***', sanitized)
        return sanitized


class TransactionError(RepositoryError):
    """
    Exception raised when database transaction fails

    Used for commit failures, rollback issues, or transaction timeout.
    """

    def __init__(
            self,
            message: str,
            operation: Optional[str] = None,
            transaction_id: Optional[str] = None,
            original_error: Optional[Exception] = None
    ):
        """
        Initialize transaction error

        Args:
            message: Transaction error message
            operation: Operation that was being performed
            transaction_id: Transaction identifier if available
            original_error: Original transaction exception
        """
        self.operation = operation
        self.transaction_id = transaction_id

        context = {
            "operation": operation,
            "transaction_id": transaction_id
        }

        super().__init__(
            message=message,
            original_error=original_error,
            error_code="TRANSACTION_ERROR",
            context=context
        )


class ConcurrencyError(RepositoryError):
    """
    Exception raised when concurrent access conflicts occur

    Used for optimistic locking failures, version conflicts,
    or when multiple operations conflict.
    """

    def __init__(
            self,
            entity_type: str,
            entity_id: str,
            expected_version: Optional[str] = None,
            actual_version: Optional[str] = None
    ):
        """
        Initialize concurrency error

        Args:
            entity_type: Type of entity with conflict
            entity_id: ID of the conflicting entity
            expected_version: Expected version for optimistic locking
            actual_version: Actual version found in database
        """
        self.entity_type = entity_type
        self.entity_id = entity_id
        self.expected_version = expected_version
        self.actual_version = actual_version

        message = f"Concurrency conflict for {entity_type} {entity_id}"
        if expected_version and actual_version:
            message += f" (expected version: {expected_version}, actual: {actual_version})"

        context = {
            "entity_type": entity_type,
            "entity_id": entity_id,
            "expected_version": expected_version,
            "actual_version": actual_version
        }

        super().__init__(
            message=message,
            error_code="CONCURRENCY_ERROR",
            context=context
        )


class QueryError(RepositoryError):
    """
    Exception raised when database query execution fails

    Used for malformed queries, syntax errors, or query timeouts.
    """

    def __init__(
            self,
            message: str,
            query: Optional[str] = None,
            query_type: Optional[str] = None,
            original_error: Optional[Exception] = None
    ):
        """
        Initialize query error

        Args:
            message: Query error message
            query: The query that failed (sanitized)
            query_type: Type of query (find, update, delete, aggregate)
            original_error: Original database exception
        """
        self.query = query
        self.query_type = query_type

        context = {
            "query": query,
            "query_type": query_type
        }

        super().__init__(
            message=message,
            original_error=original_error,
            error_code="QUERY_ERROR",
            context=context
        )


class ConfigurationError(RepositoryError):
    """
    Exception raised when repository configuration is invalid

    Used for missing settings, invalid parameters, or
    configuration mismatches.
    """

    def __init__(
            self,
            message: str,
            config_key: Optional[str] = None,
            config_value: Optional[Any] = None
    ):
        """
        Initialize configuration error

        Args:
            message: Configuration error message
            config_key: Configuration key that is invalid
            config_value: Configuration value that is invalid
        """
        self.config_key = config_key
        self.config_value = config_value

        context = {
            "config_key": config_key,
            "config_value": str(config_value) if config_value is not None else None
        }

        super().__init__(
            message=message,
            error_code="CONFIGURATION_ERROR",
            context=context
        )


class TimeoutError(RepositoryError):
    """
    Exception raised when repository operations timeout

    Used for operations that exceed configured timeout limits.
    """

    def __init__(
            self,
            operation: str,
            timeout_seconds: float,
            entity_type: Optional[str] = None
    ):
        """
        Initialize timeout error

        Args:
            operation: Operation that timed out
            timeout_seconds: Timeout duration in seconds
            entity_type: Entity type involved in operation
        """
        self.operation = operation
        self.timeout_seconds = timeout_seconds
        self.entity_type = entity_type

        message = f"Operation '{operation}' timed out after {timeout_seconds} seconds"
        if entity_type:
            message += f" for {entity_type}"

        context = {
            "operation": operation,
            "timeout_seconds": timeout_seconds,
            "entity_type": entity_type
        }

        super().__init__(
            message=message,
            error_code="TIMEOUT_ERROR",
            context=context
        )


class PermissionError(RepositoryError):
    """
    Exception raised when repository access is denied

    Used for insufficient database permissions or tenant isolation violations.
    """

    def __init__(
            self,
            operation: str,
            resource: str,
            tenant_id: Optional[str] = None,
            user_id: Optional[str] = None
    ):
        """
        Initialize permission error

        Args:
            operation: Operation that was denied
            resource: Resource that access was denied to
            tenant_id: Tenant ID if relevant
            user_id: User ID if relevant
        """
        self.operation = operation
        self.resource = resource
        self.tenant_id = tenant_id
        self.user_id = user_id

        message = f"Permission denied for {operation} on {resource}"

        context = {
            "operation": operation,
            "resource": resource,
            "tenant_id": tenant_id,
            "user_id": user_id
        }

        super().__init__(
            message=message,
            error_code="PERMISSION_ERROR",
            context=context
        )


# Exception factory for creating appropriate exceptions
class RepositoryExceptionFactory:
    """Factory for creating repository exceptions from database errors"""

    @staticmethod
    def from_database_error(
            database_error: Exception,
            operation: str,
            entity_type: Optional[str] = None,
            entity_id: Optional[str] = None
    ) -> RepositoryError:
        """
        Create appropriate repository exception from database error

        Args:
            database_error: Original database exception
            operation: Operation that failed
            entity_type: Entity type involved
            entity_id: Entity ID involved

        Returns:
            Appropriate RepositoryError subclass
        """
        error_message = str(database_error)
        error_type = type(database_error).__name__

        # MongoDB specific errors
        if "DuplicateKeyError" in error_type:
            return DuplicateEntityError(
                entity_type=entity_type or "Entity",
                conflicting_fields={"id": entity_id} if entity_id else {}
            )

        if "ConnectionFailure" in error_type or "ServerSelectionTimeoutError" in error_type:
            return ConnectionError(
                database_type="MongoDB",
                original_error=database_error
            )

        if "CursorNotFound" in error_type or "QueryExecutionTimeoutError" in error_type:
            return TimeoutError(
                operation=operation,
                timeout_seconds=30.0,  # Default timeout
                entity_type=entity_type
            )

        # Redis specific errors
        if "ConnectionError" in error_type and "redis" in error_message.lower():
            return ConnectionError(
                database_type="Redis",
                original_error=database_error
            )

        if "TimeoutError" in error_type:
            return TimeoutError(
                operation=operation,
                timeout_seconds=5.0,  # Redis default timeout
                entity_type=entity_type
            )

        # Generic query error for other database exceptions
        return QueryError(
            message=f"Database operation failed: {error_message}",
            query_type=operation,
            original_error=database_error
        )


# Convenience functions for common error scenarios
def entity_not_found(entity_type: str, entity_id: str) -> EntityNotFoundError:
    """Create entity not found error"""
    return EntityNotFoundError(entity_type, entity_id)


def duplicate_entity(entity_type: str, **conflicting_fields) -> DuplicateEntityError:
    """Create duplicate entity error"""
    return DuplicateEntityError(entity_type, conflicting_fields)


def validation_failed(message: str, field: Optional[str] = None) -> ValidationError:
    """Create validation error"""
    return ValidationError(message, field)


def connection_failed(database_type: str, original_error: Exception) -> ConnectionError:
    """Create connection error"""
    return ConnectionError(database_type, original_error=original_error)