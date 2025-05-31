"""
Dependency injection for services and repositories

Provides FastAPI dependency providers for services, repositories, and other components.
Centralizes dependency management and configuration.
"""

from functools import lru_cache
from typing import Annotated, Optional
from fastapi import Depends, HTTPException, status

# Database dependencies
try:
    from src.database.mongodb import get_mongodb
except ImportError:
    get_mongodb = None

try:
    from src.database.redis_client import get_redis
except ImportError:
    get_redis = None

# Repository dependencies
try:
    from src.repositories.conversation_repository import ConversationRepository
except ImportError:
    ConversationRepository = None

try:
    from src.repositories.message_repository import MessageRepository
except ImportError:
    MessageRepository = None

try:
    from src.repositories.session_repository import SessionRepository
except ImportError:
    SessionRepository = None

try:
    from src.repositories.rate_limit_repository import RateLimitRepository
except ImportError:
    RateLimitRepository = None

# Core component dependencies
try:
    from src.core.channels.channel_factory import ChannelFactory
except ImportError:
    ChannelFactory = None

try:
    from src.core.processors.processor_factory import ProcessorFactory
except ImportError:
    ProcessorFactory = None

# Service dependencies
from src.services.message_service import MessageService
from src.services.session_service import SessionService
from src.services.conversation_service import ConversationService
from src.services.channel_service import ChannelService
from src.services.delivery_service import DeliveryService
from src.services.audit_service import AuditService
from src.services.service_container import get_service_container_sync


# =============================================================================
# Repository Dependency Providers
# =============================================================================

async def get_conversation_repository() -> Optional['ConversationRepository']:
    """Get conversation repository instance"""
    if not ConversationRepository or not get_mongodb:
        return None

    try:
        database = await get_mongodb()
        return ConversationRepository(database)
    except Exception as e:
        print(f"Failed to create ConversationRepository: {e}")
        return None


async def get_message_repository() -> Optional['MessageRepository']:
    """Get message repository instance"""
    if not MessageRepository or not get_mongodb:
        return None

    try:
        database = await get_mongodb()
        return MessageRepository(database)
    except Exception as e:
        print(f"Failed to create MessageRepository: {e}")
        return None


async def get_session_repository() -> Optional['SessionRepository']:
    """Get session repository instance"""
    if not SessionRepository or not get_redis:
        return None

    try:
        redis_client = await get_redis()
        return SessionRepository(redis_client)
    except Exception as e:
        print(f"Failed to create SessionRepository: {e}")
        return None


async def get_rate_limit_repository() -> Optional['RateLimitRepository']:
    """Get rate limit repository instance"""
    if not RateLimitRepository or not get_redis:
        return None

    try:
        redis_client = await get_redis()
        return RateLimitRepository(redis_client)
    except Exception as e:
        print(f"Failed to create RateLimitRepository: {e}")
        return None


# =============================================================================
# Core Component Dependency Providers
# =============================================================================

@lru_cache()
def get_channel_factory() -> Optional['ChannelFactory']:
    """Get channel factory instance (cached)"""
    if not ChannelFactory:
        return None

    try:
        return ChannelFactory()
    except Exception as e:
        print(f"Failed to create ChannelFactory: {e}")
        return None


@lru_cache()
def get_processor_factory() -> Optional['ProcessorFactory']:
    """Get processor factory instance (cached)"""
    if not ProcessorFactory:
        return None

    try:
        return ProcessorFactory()
    except Exception as e:
        print(f"Failed to create ProcessorFactory: {e}")
        return None


# =============================================================================
# Service Dependency Providers
# =============================================================================

async def get_message_service(
        conversation_repo: Annotated[Optional['ConversationRepository'], Depends(get_conversation_repository)] = None,
        message_repo: Annotated[Optional['MessageRepository'], Depends(get_message_repository)] = None,
        session_repo: Annotated[Optional['SessionRepository'], Depends(get_session_repository)] = None,
        channel_factory: Annotated[Optional['ChannelFactory'], Depends(get_channel_factory)] = None,
        processor_factory: Annotated[Optional['ProcessorFactory'], Depends(get_processor_factory)] = None
) -> MessageService:
    """Get message service instance with all dependencies"""

    # Check if we can create the service with available dependencies
    if conversation_repo and message_repo and session_repo and channel_factory and processor_factory:
        return MessageService(
            conversation_repo=conversation_repo,
            message_repo=message_repo,
            session_repo=session_repo,
            channel_factory=channel_factory,
            processor_factory=processor_factory
        )

    # Try to get from service container
    try:
        container = get_service_container_sync()
        # This would be async in real implementation
        # For now, create with available dependencies
        return MessageService(
            conversation_repo=conversation_repo,
            message_repo=message_repo,
            session_repo=session_repo,
            channel_factory=channel_factory or get_channel_factory(),
            processor_factory=processor_factory or get_processor_factory()
        )
    except Exception as e:
        print(f"Failed to get MessageService: {e}")
        # Create with minimal dependencies for development
        return MessageService(
            conversation_repo=conversation_repo,
            message_repo=message_repo,
            session_repo=session_repo,
            channel_factory=get_channel_factory(),
            processor_factory=get_processor_factory()
        )


async def get_session_service(
        session_repo: Annotated[Optional['SessionRepository'], Depends(get_session_repository)] = None
) -> SessionService:
    """Get session service instance"""

    # Create with available repository
    return SessionService(session_repo)


async def get_conversation_service(
        conversation_repo: Annotated[Optional['ConversationRepository'], Depends(get_conversation_repository)] = None,
        message_repo: Annotated[Optional['MessageRepository'], Depends(get_message_repository)] = None
) -> ConversationService:
    """Get conversation service instance"""

    # Create with available repositories
    return ConversationService(conversation_repo, message_repo)


async def get_channel_service(
        channel_factory: Annotated[Optional['ChannelFactory'], Depends(get_channel_factory)] = None
) -> ChannelService:
    """Get channel service instance"""

    # Create with available factory
    return ChannelService(channel_factory or get_channel_factory())


async def get_delivery_service(
        message_repo: Annotated[Optional['MessageRepository'], Depends(get_message_repository)] = None,
        channel_factory: Annotated[Optional['ChannelFactory'], Depends(get_channel_factory)] = None
) -> DeliveryService:
    """Get delivery service instance"""

    # Create with available dependencies
    return DeliveryService(
        message_repo=message_repo,
        channel_factory=channel_factory or get_channel_factory()
    )


async def get_audit_service() -> AuditService:
    """Get audit service instance"""

    # Audit service has minimal dependencies
    return AuditService()


# =============================================================================
# Health Check Dependencies
# =============================================================================

async def get_health_checkers():
    """Get all health check dependencies"""
    health_checkers = {}

    if get_mongodb:
        try:
            health_checkers["mongodb"] = await get_mongodb()
        except Exception as e:
            print(f"MongoDB health check failed: {e}")

    if get_redis:
        try:
            health_checkers["redis"] = await get_redis()
        except Exception as e:
            print(f"Redis health check failed: {e}")

    return health_checkers


# =============================================================================
# Service Container Dependencies
# =============================================================================

async def get_service_container():
    """Get service container instance"""
    return get_service_container_sync()


# =============================================================================
# Authentication and Authorization Dependencies
# =============================================================================

async def get_current_user(
        # This would typically validate JWT token from headers
        # For now, return a mock user context
) -> dict:
    """Get current authenticated user context"""
    # In production, this would:
    # 1. Extract JWT from Authorization header
    # 2. Validate token with Security Hub
    # 3. Return user context with permissions

    # For development, return mock user
    return {
        "user_id": "dev_user",
        "tenant_id": "dev_tenant",
        "role": "admin",
        "permissions": ["*"]
    }


async def get_tenant_id(
        # This would typically extract from headers or user context
) -> str:
    """Get current tenant ID"""
    # In production, this would extract from:
    # 1. X-Tenant-ID header
    # 2. Subdomain
    # 3. User context

    # For development, return default tenant
    return "dev_tenant"


async def get_api_key_context(
        # This would validate API key from headers
) -> Optional[dict]:
    """Get API key context if present"""
    # In production, this would:
    # 1. Extract API key from headers
    # 2. Validate with Security Hub
    # 3. Return API key context with permissions

    return None


# =============================================================================
# Request Context Dependencies
# =============================================================================

async def get_request_context(
        # request: Request  # Would be FastAPI Request object
) -> dict:
    """Get request context for audit logging"""
    # In production, this would extract from FastAPI Request:
    # - IP address
    # - User agent
    # - Request ID
    # - Timestamp

    return {
        "ip_address": "127.0.0.1",
        "user_agent": "development",
        "request_id": "dev_request",
        "timestamp": "2025-01-01T00:00:00Z"
    }


# =============================================================================
# Configuration Dependencies
# =============================================================================

@lru_cache()
def get_app_config() -> dict:
    """Get application configuration"""
    # In production, this would load from:
    # - Environment variables
    # - Configuration files
    # - External configuration services

    return {
        "debug": True,
        "environment": "development",
        "services": {
            "message_service": {"enabled": True},
            "session_service": {"enabled": True},
            "conversation_service": {"enabled": True},
            "channel_service": {"enabled": True},
            "delivery_service": {"enabled": True},
            "audit_service": {"enabled": True}
        }
    }


# =============================================================================
# Error Handling Dependencies
# =============================================================================

def handle_service_unavailable(service_name: str):
    """Handle case when a service is unavailable"""
    raise HTTPException(
        status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
        detail=f"Service {service_name} is currently unavailable"
    )


def handle_dependency_error(dependency_name: str, error: Exception):
    """Handle dependency injection errors"""
    print(f"Dependency injection error for {dependency_name}: {error}")
    # In production, this might:
    # - Log to monitoring system
    # - Trigger alerts
    # - Return fallback dependencies


# =============================================================================
# Service Factory Functions
# =============================================================================

async def create_message_service_with_fallbacks() -> MessageService:
    """Create message service with fallback dependencies"""
    try:
        # Try to get all dependencies
        conversation_repo = await get_conversation_repository()
        message_repo = await get_message_repository()
        session_repo = await get_session_repository()
        channel_factory = get_channel_factory()
        processor_factory = get_processor_factory()

        return MessageService(
            conversation_repo=conversation_repo,
            message_repo=message_repo,
            session_repo=session_repo,
            channel_factory=channel_factory,
            processor_factory=processor_factory
        )
    except Exception as e:
        handle_dependency_error("MessageService", e)
        raise


async def create_session_service_with_fallbacks() -> SessionService:
    """Create session service with fallback dependencies"""
    try:
        session_repo = await get_session_repository()
        return SessionService(session_repo)
    except Exception as e:
        handle_dependency_error("SessionService", e)
        raise


# =============================================================================
# Development and Testing Dependencies
# =============================================================================

def get_mock_dependencies() -> dict:
    """Get mock dependencies for testing"""
    return {
        "conversation_repo": None,
        "message_repo": None,
        "session_repo": None,
        "channel_factory": get_channel_factory(),
        "processor_factory": get_processor_factory()
    }


async def override_dependencies_for_testing():
    """Override dependencies for testing environment"""
    # This would be used in test setup to provide mock implementations
    pass


# =============================================================================
# Utility Functions
# =============================================================================

def get_available_services() -> list:
    """Get list of available service types"""
    return [
        "message_service",
        "session_service",
        "conversation_service",
        "channel_service",
        "delivery_service",
        "audit_service"
    ]


async def validate_service_dependencies(service_name: str) -> bool:
    """Validate that all dependencies for a service are available"""
    try:
        if service_name == "message_service":
            await get_message_service()
        elif service_name == "session_service":
            await get_session_service()
        elif service_name == "conversation_service":
            await get_conversation_service()
        elif service_name == "channel_service":
            await get_channel_service()
        elif service_name == "delivery_service":
            await get_delivery_service()
        elif service_name == "audit_service":
            await get_audit_service()
        else:
            return False

        return True
    except Exception:
        return False