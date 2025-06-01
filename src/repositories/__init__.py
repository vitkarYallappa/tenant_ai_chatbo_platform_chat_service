"""
Repository Layer Package
========================

This package provides the data access layer for the chat service, implementing
the repository pattern for clean separation between business logic and data persistence.

Supported Databases:
- MongoDB: Conversation and message document storage
- Redis: Session management, caching, and rate limiting
- PostgreSQL: Configuration and metadata (future phase)

Repository Pattern Benefits:
- Clean separation of concerns
- Testability through dependency injection
- Consistent error handling
- Database agnostic business logic
"""

from .base_repository import (
    BaseRepository,
    Pagination,
    PaginatedResult
)

from .exceptions import (
    RepositoryError,
    EntityNotFoundError,
    DuplicateEntityError,
    ValidationError,
    ConnectionError,
    TransactionError
)

from .conversation_repository import (
    ConversationRepository,
    get_conversation_repository
)

from .message_repository import (
    MessageRepository,
    get_message_repository
)

from .session_repository import (
    SessionRepository,
    get_session_repository
)

from .rate_limit_repository import (
    RateLimitRepository,
    get_rate_limit_repository
)

from .cache_repository import (
    CacheRepository,
    get_cache_repository
)

from .tenant_repository import (
    TenantRepository
)

# Repository factory for dependency injection
from typing import Dict, Type, Any

from .webhook_repository import (
    WebhookRepository
)

_repository_registry: Dict[str, Type[Any]] = {
    'conversation': ConversationRepository,
    'message': MessageRepository,
    'session': SessionRepository,
    'rate_limit': RateLimitRepository,
    'cache': CacheRepository,
    'tenant': TenantRepository,
    'webhook': WebhookRepository
}


def get_repository(repository_type: str) -> Type[Any]:
    """
    Get repository class by type

    Args:
        repository_type: Type of repository ('conversation', 'message', etc.)

    Returns:
        Repository class

    Raises:
        ValueError: If repository type is not found
    """
    if repository_type not in _repository_registry:
        raise ValueError(f"Unknown repository type: {repository_type}")

    return _repository_registry[repository_type]


def register_repository(repository_type: str, repository_class: Type[Any]) -> None:
    """
    Register a custom repository type

    Args:
        repository_type: Unique identifier for the repository
        repository_class: Repository class implementing BaseRepository
    """
    _repository_registry[repository_type] = repository_class


__all__ = [
    # Base classes
    'BaseRepository',
    'Pagination',
    'PaginatedResult',

    # Exceptions
    'RepositoryError',
    'EntityNotFoundError',
    'DuplicateEntityError',
    'ValidationError',
    'ConnectionError',
    'TransactionError',

    # Repository implementations
    'ConversationRepository',
    'MessageRepository',
    'SessionRepository',
    'RateLimitRepository',
    'CacheRepository',
    'TenantRepository',
    'WebhookRepository',



    # Dependency injection helpers
    'get_conversation_repository',
    'get_message_repository',
    'get_session_repository',
    'get_rate_limit_repository',
    'get_cache_repository',

    # Factory functions
    'get_repository',
    'register_repository',
]

# Version information
__version__ = "1.0.0"
__author__ = "Chat Service Team"
__description__ = "Repository layer for multi-tenant chat service"
