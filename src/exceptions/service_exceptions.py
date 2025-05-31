"""
Service-specific exceptions for Chat Service.

This module provides custom exception classes for service-layer errors
including business logic failures, data processing issues, and
service integration problems.
"""

from typing import Any, Dict, List, Optional, Union
from datetime import datetime, timezone

from src.exceptions.base_exceptions import ChatServiceException
from src.config.constants import ErrorCategory


class ServiceException(ChatServiceException):
    """Base exception for all service-layer errors."""

    def __init__(
            self,
            message: str,
            service_name: Optional[str] = None,
            operation: Optional[str] = None,
            **kwargs
    ):
        details = kwargs.pop("details", {})
        if service_name:
            details["service_name"] = service_name
        if operation:
            details["operation"] = operation

        super().__init__(
            message=message,
            error_code="SERVICE_ERROR",
            status_code=500,
            category=ErrorCategory.INTERNAL,
            user_message="A service error occurred. Please try again later.",
            details=details,
            **kwargs
        )


class ConversationServiceException(ServiceException):
    """Exception for conversation service errors."""

    def __init__(
            self,
            message: str,
            conversation_id: Optional[str] = None,
            **kwargs
    ):
        details = kwargs.pop("details", {})
        if conversation_id:
            details["conversation_id"] = conversation_id

        super().__init__(
            message=message,
            service_name="conversation_service",
            details=details,
            **kwargs
        )


class ConversationNotFoundError(ConversationServiceException):
    """Exception for conversation not found errors."""

    def __init__(
            self,
            conversation_id: str,
            tenant_id: Optional[str] = None,
            **kwargs
    ):
        details = kwargs.pop("details", {})
        details["conversation_id"] = conversation_id
        if tenant_id:
            details["tenant_id"] = tenant_id

        message = f"Conversation not found: {conversation_id}"
        user_message = "The requested conversation could not be found"

        super().__init__(
            message=message,
            conversation_id=conversation_id,
            error_code="CONVERSATION_NOT_FOUND",
            status_code=404,
            category=ErrorCategory.NOT_FOUND,
            user_message=user_message,
            details=details,
            **kwargs
        )


class ConversationStateError(ConversationServiceException):
    """Exception for conversation state management errors."""

    def __init__(
            self,
            conversation_id: str,
            current_state: str,
            attempted_action: str,
            **kwargs
    ):
        details = kwargs.pop("details", {})
        details.update({
            "conversation_id": conversation_id,
            "current_state": current_state,
            "attempted_action": attempted_action
        })

        message = f"Invalid state transition for conversation {conversation_id}: {current_state} -> {attempted_action}"
        user_message = "This action is not allowed in the current conversation state"

        super().__init__(
            message=message,
            conversation_id=conversation_id,
            error_code="CONVERSATION_STATE_ERROR",
            status_code=409,
            category=ErrorCategory.VALIDATION,
            user_message=user_message,
            details=details,
            **kwargs
        )


class MessageServiceException(ServiceException):
    """Exception for message service errors."""

    def __init__(
            self,
            message: str,
            message_id: Optional[str] = None,
            **kwargs
    ):
        details = kwargs.pop("details", {})
        if message_id:
            details["message_id"] = message_id

        super().__init__(
            message=message,
            service_name="message_service",
            details=details,
            **kwargs
        )


class MessageProcessingError(MessageServiceException):
    """Exception for message processing failures."""

    def __init__(
            self,
            message_id: str,
            processing_stage: str,
            reason: str,
            **kwargs
    ):
        details = kwargs.pop("details", {})
        details.update({
            "message_id": message_id,
            "processing_stage": processing_stage,
            "failure_reason": reason
        })

        message = f"Message processing failed at stage '{processing_stage}': {reason}"
        user_message = "Failed to process your message. Please try again."

        super().__init__(
            message=message,
            message_id=message_id,
            error_code="MESSAGE_PROCESSING_ERROR",
            status_code=422,
            category=ErrorCategory.INTERNAL,
            user_message=user_message,
            retryable=True,
            details=details,
            **kwargs
        )


class MessageNotFoundError(MessageServiceException):
    """Exception for message not found errors."""

    def __init__(
            self,
            message_id: str,
            conversation_id: Optional[str] = None,
            **kwargs
    ):
        details = kwargs.pop("details", {})
        details["message_id"] = message_id
        if conversation_id:
            details["conversation_id"] = conversation_id

        message = f"Message not found: {message_id}"
        user_message = "The requested message could not be found"

        super().__init__(
            message=message,
            message_id=message_id,
            error_code="MESSAGE_NOT_FOUND",
            status_code=404,
            category=ErrorCategory.NOT_FOUND,
            user_message=user_message,
            details=details,
            **kwargs
        )


class SessionServiceException(ServiceException):
    """Exception for session service errors."""

    def __init__(
            self,
            message: str,
            session_id: Optional[str] = None,
            **kwargs
    ):
        details = kwargs.pop("details", {})
        if session_id:
            details["session_id"] = session_id

        super().__init__(
            message=message,
            service_name="session_service",
            details=details,
            **kwargs
        )


class SessionExpiredError(SessionServiceException):
    """Exception for expired session errors."""

    def __init__(
            self,
            session_id: str,
            expired_at: Optional[datetime] = None,
            **kwargs
    ):
        details = kwargs.pop("details", {})
        details["session_id"] = session_id
        if expired_at:
            details["expired_at"] = expired_at.isoformat()

        message = f"Session expired: {session_id}"
        user_message = "Your session has expired. Please start a new conversation."

        super().__init__(
            message=message,
            session_id=session_id,
            error_code="SESSION_EXPIRED",
            status_code=401,
            category=ErrorCategory.AUTHENTICATION,
            user_message=user_message,
            details=details,
            **kwargs
        )


class SessionLimitExceededError(SessionServiceException):
    """Exception for session limit exceeded errors."""

    def __init__(
            self,
            user_id: str,
            current_sessions: int,
            max_sessions: int,
            **kwargs
    ):
        details = kwargs.pop("details", {})
        details.update({
            "user_id": user_id,
            "current_sessions": current_sessions,
            "max_sessions": max_sessions
        })

        message = f"Session limit exceeded for user {user_id}: {current_sessions}/{max_sessions}"
        user_message = f"You have reached the maximum number of active sessions ({max_sessions})"

        super().__init__(
            message=message,
            error_code="SESSION_LIMIT_EXCEEDED",
            status_code=429,
            category=ErrorCategory.RATE_LIMIT,
            user_message=user_message,
            details=details,
            **kwargs
        )


class FlowServiceException(ServiceException):
    """Exception for conversation flow service errors."""

    def __init__(
            self,
            message: str,
            flow_id: Optional[str] = None,
            **kwargs
    ):
        details = kwargs.pop("details", {})
        if flow_id:
            details["flow_id"] = flow_id

        super().__init__(
            message=message,
            service_name="flow_service",
            details=details,
            **kwargs
        )


class FlowExecutionError(FlowServiceException):
    """Exception for flow execution errors."""

    def __init__(
            self,
            flow_id: str,
            current_state: str,
            execution_error: str,
            **kwargs
    ):
        details = kwargs.pop("details", {})
        details.update({
            "flow_id": flow_id,
            "current_state": current_state,
            "execution_error": execution_error
        })

        message = f"Flow execution failed: {flow_id} at state {current_state} - {execution_error}"
        user_message = "There was an issue processing your request. Please try again."

        super().__init__(
            message=message,
            flow_id=flow_id,
            error_code="FLOW_EXECUTION_ERROR",
            status_code=422,
            category=ErrorCategory.INTERNAL,
            user_message=user_message,
            retryable=True,
            details=details,
            **kwargs
        )


class FlowNotFoundError(FlowServiceException):
    """Exception for flow not found errors."""

    def __init__(
            self,
            flow_id: str,
            tenant_id: Optional[str] = None,
            **kwargs
    ):
        details = kwargs.pop("details", {})
        details["flow_id"] = flow_id
        if tenant_id:
            details["tenant_id"] = tenant_id

        message = f"Conversation flow not found: {flow_id}"
        user_message = "The conversation flow is not available"

        super().__init__(
            message=message,
            flow_id=flow_id,
            error_code="FLOW_NOT_FOUND",
            status_code=404,
            category=ErrorCategory.NOT_FOUND,
            user_message=user_message,
            details=details,
            **kwargs
        )


class IntegrationServiceException(ServiceException):
    """Exception for integration service errors."""

    def __init__(
            self,
            message: str,
            integration_id: Optional[str] = None,
            **kwargs
    ):
        details = kwargs.pop("details", {})
        if integration_id:
            details["integration_id"] = integration_id

        super().__init__(
            message=message,
            service_name="integration_service",
            details=details,
            **kwargs
        )


class IntegrationExecutionError(IntegrationServiceException):
    """Exception for integration execution errors."""

    def __init__(
            self,
            integration_id: str,
            operation: str,
            error_details: str,
            retry_count: int = 0,
            **kwargs
    ):
        details = kwargs.pop("details", {})
        details.update({
            "integration_id": integration_id,
            "operation": operation,
            "error_details": error_details,
            "retry_count": retry_count
        })

        message = f"Integration execution failed: {integration_id}.{operation} - {error_details}"
        user_message = "External service integration failed. Please try again later."

        super().__init__(
            message=message,
            integration_id=integration_id,
            operation=operation,
            error_code="INTEGRATION_EXECUTION_ERROR",
            status_code=502,
            category=ErrorCategory.EXTERNAL,
            user_message=user_message,
            retryable=True,
            details=details,
            **kwargs
        )


class IntegrationTimeoutError(IntegrationServiceException):
    """Exception for integration timeout errors."""

    def __init__(
            self,
            integration_id: str,
            operation: str,
            timeout_seconds: float,
            **kwargs
    ):
        details = kwargs.pop("details", {})
        details.update({
            "integration_id": integration_id,
            "operation": operation,
            "timeout_seconds": timeout_seconds
        })

        message = f"Integration timeout: {integration_id}.{operation} after {timeout_seconds}s"
        user_message = "External service request timed out. Please try again."

        super().__init__(
            message=message,
            integration_id=integration_id,
            operation=operation,
            error_code="INTEGRATION_TIMEOUT",
            status_code=504,
            category=ErrorCategory.TIMEOUT,
            user_message=user_message,
            retryable=True,
            details=details,
            **kwargs
        )


class DataServiceException(ServiceException):
    """Exception for data service errors."""

    def __init__(
            self,
            message: str,
            operation: Optional[str] = None,
            **kwargs
    ):
        super().__init__(
            message=message,
            service_name="data_service",
            operation=operation,
            **kwargs
        )


class DataStorageError(DataServiceException):
    """Exception for data storage errors."""

    def __init__(
            self,
            operation: str,
            entity_type: str,
            entity_id: Optional[str] = None,
            reason: Optional[str] = None,
            **kwargs
    ):
        details = kwargs.pop("details", {})
        details.update({
            "operation": operation,
            "entity_type": entity_type
        })

        if entity_id:
            details["entity_id"] = entity_id
        if reason:
            details["reason"] = reason

        message = f"Data storage error: {operation} {entity_type}"
        if reason:
            message += f" - {reason}"

        user_message = "Failed to save data. Please try again."

        super().__init__(
            message=message,
            operation=operation,
            error_code="DATA_STORAGE_ERROR",
            status_code=500,
            category=ErrorCategory.INTERNAL,
            user_message=user_message,
            retryable=True,
            details=details,
            **kwargs
        )


class DataRetrievalError(DataServiceException):
    """Exception for data retrieval errors."""

    def __init__(
            self,
            entity_type: str,
            entity_id: Optional[str] = None,
            query_params: Optional[Dict[str, Any]] = None,
            **kwargs
    ):
        details = kwargs.pop("details", {})
        details["entity_type"] = entity_type

        if entity_id:
            details["entity_id"] = entity_id
        if query_params:
            details["query_params"] = query_params

        message = f"Data retrieval error: {entity_type}"
        user_message = "Failed to retrieve data. Please try again."

        super().__init__(
            message=message,
            operation="retrieve",
            error_code="DATA_RETRIEVAL_ERROR",
            status_code=500,
            category=ErrorCategory.INTERNAL,
            user_message=user_message,
            retryable=True,
            details=details,
            **kwargs
        )


class BusinessLogicError(ServiceException):
    """Exception for business logic violations."""

    def __init__(
            self,
            rule: str,
            message: str,
            affected_entities: Optional[List[str]] = None,
            **kwargs
    ):
        details = kwargs.pop("details", {})
        details.update({
            "business_rule": rule,
            "violation_message": message
        })

        if affected_entities:
            details["affected_entities"] = affected_entities

        error_message = f"Business logic violation: {rule} - {message}"
        user_message = f"Business rule violation: {message}"

        super().__init__(
            message=error_message,
            error_code="BUSINESS_LOGIC_ERROR",
            status_code=422,
            category=ErrorCategory.VALIDATION,
            user_message=user_message,
            details=details,
            **kwargs
        )


class ConcurrencyError(ServiceException):
    """Exception for concurrency control errors."""

    def __init__(
            self,
            resource_type: str,
            resource_id: str,
            operation: str,
            **kwargs
    ):
        details = kwargs.pop("details", {})
        details.update({
            "resource_type": resource_type,
            "resource_id": resource_id,
            "operation": operation
        })

        message = f"Concurrency error: {operation} on {resource_type} {resource_id}"
        user_message = "The resource is being modified by another user. Please try again."

        super().__init__(
            message=message,
            error_code="CONCURRENCY_ERROR",
            status_code=409,
            category=ErrorCategory.VALIDATION,
            user_message=user_message,
            retryable=True,
            details=details,
            **kwargs
        )


class ResourceQuotaExceededError(ServiceException):
    """Exception for resource quota exceeded errors."""

    def __init__(
            self,
            resource_type: str,
            current_usage: int,
            quota_limit: int,
            tenant_id: Optional[str] = None,
            **kwargs
    ):
        details = kwargs.pop("details", {})
        details.update({
            "resource_type": resource_type,
            "current_usage": current_usage,
            "quota_limit": quota_limit
        })

        if tenant_id:
            details["tenant_id"] = tenant_id

        message = f"Resource quota exceeded: {resource_type} {current_usage}/{quota_limit}"
        user_message = f"You have reached the limit for {resource_type} ({quota_limit})"

        super().__init__(
            message=message,
            error_code="RESOURCE_QUOTA_EXCEEDED",
            status_code=429,
            category=ErrorCategory.RATE_LIMIT,
            user_message=user_message,
            details=details,
            **kwargs
        )


class ServiceUnavailableError(ServiceException):
    """Exception for service unavailability."""

    def __init__(
            self,
            service_name: str,
            reason: str = "Service temporarily unavailable",
            estimated_recovery: Optional[datetime] = None,
            **kwargs
    ):
        details = kwargs.pop("details", {})
        details.update({
            "service_name": service_name,
            "reason": reason
        })

        if estimated_recovery:
            details["estimated_recovery"] = estimated_recovery.isoformat()

        message = f"Service unavailable: {service_name} - {reason}"
        user_message = f"The {service_name} is temporarily unavailable. Please try again later."

        super().__init__(
            message=message,
            service_name=service_name,
            error_code="SERVICE_UNAVAILABLE",
            status_code=503,
            category=ErrorCategory.NETWORK,
            user_message=user_message,
            retryable=True,
            details=details,
            **kwargs
        )


# Convenience functions for raising service-specific errors
def raise_conversation_not_found(conversation_id: str, tenant_id: str = None):
    """Raise ConversationNotFoundError."""
    raise ConversationNotFoundError(
        conversation_id=conversation_id,
        tenant_id=tenant_id
    )


def raise_message_processing_error(
        message_id: str,
        stage: str,
        reason: str
):
    """Raise MessageProcessingError."""
    raise MessageProcessingError(
        message_id=message_id,
        processing_stage=stage,
        reason=reason
    )


def raise_session_expired_error(session_id: str, expired_at: datetime = None):
    """Raise SessionExpiredError."""
    raise SessionExpiredError(
        session_id=session_id,
        expired_at=expired_at
    )


def raise_flow_execution_error(flow_id: str, state: str, error: str):
    """Raise FlowExecutionError."""
    raise FlowExecutionError(
        flow_id=flow_id,
        current_state=state,
        execution_error=error
    )


def raise_integration_timeout_error(
        integration_id: str,
        operation: str,
        timeout_seconds: float
):
    """Raise IntegrationTimeoutError."""
    raise IntegrationTimeoutError(
        integration_id=integration_id,
        operation=operation,
        timeout_seconds=timeout_seconds
    )


def raise_business_logic_error(rule: str, message: str, affected_entities: List[str] = None):
    """Raise BusinessLogicError."""
    raise BusinessLogicError(
        rule=rule,
        message=message,
        affected_entities=affected_entities
    )


def raise_resource_quota_exceeded(
        resource_type: str,
        current_usage: int,
        quota_limit: int,
        tenant_id: str = None
):
    """Raise ResourceQuotaExceededError."""
    raise ResourceQuotaExceededError(
        resource_type=resource_type,
        current_usage=current_usage,
        quota_limit=quota_limit,
        tenant_id=tenant_id
    )


# Export all service exception classes and utilities
__all__ = [
    # Base service exceptions
    'ServiceException',

    # Conversation service exceptions
    'ConversationServiceException',
    'ConversationNotFoundError',
    'ConversationStateError',

    # Message service exceptions
    'MessageServiceException',
    'MessageProcessingError',
    'MessageNotFoundError',

    # Session service exceptions
    'SessionServiceException',
    'SessionExpiredError',
    'SessionLimitExceededError',

    # Flow service exceptions
    'FlowServiceException',
    'FlowExecutionError',
    'FlowNotFoundError',

    # Integration service exceptions
    'IntegrationServiceException',
    'IntegrationExecutionError',
    'IntegrationTimeoutError',

    # Data service exceptions
    'DataServiceException',
    'DataStorageError',
    'DataRetrievalError',

    # Business logic exceptions
    'BusinessLogicError',
    'ConcurrencyError',
    'ResourceQuotaExceededError',
    'ServiceUnavailableError',

    # Convenience functions
    'raise_conversation_not_found',
    'raise_message_processing_error',
    'raise_session_expired_error',
    'raise_flow_execution_error',
    'raise_integration_timeout_error',
    'raise_business_logic_error',
    'raise_resource_quota_exceeded',
]