"""
Services Package

This package contains the service layer implementation for the Chat Service.
Services orchestrate business logic, coordinate between repositories and core components,
and provide the main interface for API layer operations.

Service Architecture:
- BaseService: Abstract base with common patterns and utilities
- MessageService: Core message processing orchestration
- ConversationService: Conversation lifecycle management
- ChannelService: Channel abstraction and routing
- SessionService: User session and state management
- DeliveryService: Message delivery tracking and retries
- AuditService: Compliance and audit logging

All services follow dependency injection patterns and provide comprehensive
error handling, logging, and performance monitoring.
"""

from .base_service import BaseService
from .exceptions import (
    ServiceError,
    ValidationError,
    UnauthorizedError,
    NotFoundError,
    ConflictError,
    RateLimitError,
    ExternalServiceError,
    ConfigurationError,
    ProcessingError,
    DeliveryError
)
from .message_service import MessageService
from .session_service import SessionService

# Import other services when they're implemented
try:
    from .conversation_service import ConversationService
except ImportError:
    ConversationService = None

try:
    from .channel_service import ChannelService
except ImportError:
    ChannelService = None

try:
    from .delivery_service import DeliveryService
except ImportError:
    DeliveryService = None

try:
    from .audit_service import AuditService
except ImportError:
    AuditService = None

try:
    from .service_container import ServiceContainer
except ImportError:
    ServiceContainer = None

__all__ = [
    # Base classes
    "BaseService",

    # Exceptions
    "ServiceError",
    "ValidationError",
    "UnauthorizedError",
    "NotFoundError",
    "ConflictError",
    "RateLimitError",
    "ExternalServiceError",
    "ConfigurationError",
    "ProcessingError",
    "DeliveryError",

    # Core services
    "MessageService",
    "SessionService",
    "ConversationService",
    "ChannelService",
    "DeliveryService",
    "AuditService",
    "ServiceContainer",
]

# Service registry for dependency injection
SERVICES = {
    "message_service": MessageService,
    "session_service": SessionService,
    "conversation_service": ConversationService,
    "channel_service": ChannelService,
    "delivery_service": DeliveryService,
    "audit_service": AuditService,
}


def get_available_services():
    """Return list of available service names"""
    return [name for name, cls in SERVICES.items() if cls is not None]


def get_service_class(service_name: str):
    """Get service class by name"""
    service_class = SERVICES.get(service_name)
    if service_class is None:
        raise ImportError(f"Service '{service_name}' is not available")
    return service_class