"""
Dependency injection for services and repositories

Provides FastAPI dependency providers for services, repositories, and other components.
Centralizes dependency management and configuration.
"""

from functools import lru_cache
from typing import Annotated, Optional
from fastapi import Depends, HTTPException, status
from sqlalchemy.ext.asyncio import async_sessionmaker, AsyncSession

# Database dependencies
try:
    from src.database.mongodb import get_mongodb
except ImportError:
    get_mongodb = None

try:
    from src.database.redis_client import get_redis
except ImportError:
    get_redis = None

try:
    from src.database.postgresql import get_postgresql_engine
except ImportError:
    get_postgresql_engine = None

# MongoDB Repository dependencies
try:
    from src.repositories.conversation_repository import ConversationRepository
except ImportError:
    ConversationRepository = None

try:
    from src.repositories.message_repository import MessageRepository
except ImportError:
    MessageRepository = None

# Redis Repository dependencies
try:
    from src.repositories.session_repository import SessionRepository
except ImportError:
    SessionRepository = None

try:
    from src.repositories.rate_limit_repository import RateLimitRepository
except ImportError:
    RateLimitRepository = None

try:
    from src.repositories.cache_repository import CacheRepository
except ImportError:
    CacheRepository = None

# PostgreSQL Repository dependencies
try:
    from src.repositories.tenant_repository import TenantRepository
except ImportError:
    TenantRepository = None

try:
    from src.repositories.webhook_repository import WebhookRepository
except ImportError:
    WebhookRepository = None

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

# Global session factory cache for PostgreSQL
_session_factory: Optional[async_sessionmaker[AsyncSession]] = None


# =============================================================================
# PostgreSQL Session Factory Management
# =============================================================================

async def get_session_factory() -> Optional[async_sessionmaker[AsyncSession]]:
    """
    Get SQLAlchemy session factory with error handling

    Returns:
        Session factory or None if setup fails
    """
    global _session_factory

    if not get_postgresql_engine:
        print("PostgreSQL engine not available - database module not imported")
        return None

    if _session_factory is None:
        try:
            engine = await get_postgresql_engine()
            _session_factory = async_sessionmaker(
                engine,
                class_=AsyncSession,
                expire_on_commit=False
            )
        except Exception as e:
            print(f"Failed to create session factory: {e}")
            return None

    return _session_factory


# =============================================================================
# MongoDB Repository Dependency Providers
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


# =============================================================================
# Redis Repository Dependency Providers
# =============================================================================

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


async def get_cache_repository() -> Optional['CacheRepository']:
    """Get cache repository instance"""
    if not CacheRepository or not get_redis:
        return None

    try:
        redis_client = await get_redis()
        return CacheRepository(redis_client)
    except Exception as e:
        print(f"Failed to create CacheRepository: {e}")
        return None


# =============================================================================
# PostgreSQL Repository Dependency Providers
# =============================================================================

async def get_tenant_repository() -> Optional['TenantRepository']:
    """Get tenant repository instance"""
    if not TenantRepository or not get_postgresql_engine:
        return None

    try:
        session_factory = await get_session_factory()
        if not session_factory:
            return None
        return TenantRepository(session_factory)
    except Exception as e:
        print(f"Failed to create TenantRepository: {e}")
        return None


async def get_webhook_repository() -> Optional['WebhookRepository']:
    """Get webhook repository instance"""
    if not WebhookRepository or not get_postgresql_engine:
        return None

    try:
        session_factory = await get_session_factory()
        if not session_factory:
            return None
        return WebhookRepository(session_factory)
    except Exception as e:
        print(f"Failed to create WebhookRepository: {e}")
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
# Tenant and Webhook Services (New)
# =============================================================================

async def get_tenant_service(
        tenant_repo: Annotated[Optional['TenantRepository'], Depends(get_tenant_repository)] = None,
        audit_service: Annotated[AuditService, Depends(get_audit_service)] = None
):
    """Get tenant service instance"""
    try:
        from src.services.tenant_service import TenantService
        return TenantService(
            tenant_repo=tenant_repo,
            audit_service=audit_service
        )
    except ImportError:
        print("TenantService not available - service module not imported")
        return None
    except Exception as e:
        print(f"Failed to create TenantService: {e}")
        return None


async def get_webhook_service(
        webhook_repo: Annotated[Optional['WebhookRepository'], Depends(get_webhook_repository)] = None,
        tenant_repo: Annotated[Optional['TenantRepository'], Depends(get_tenant_repository)] = None,
        audit_service: Annotated[AuditService, Depends(get_audit_service)] = None
):
    """Get webhook service instance"""
    try:
        from src.services.webhook_service import WebhookService
        return WebhookService(
            webhook_repo=webhook_repo,
            tenant_repo=tenant_repo,
            audit_service=audit_service
        )
    except ImportError:
        print("WebhookService not available - service module not imported")
        return None
    except Exception as e:
        print(f"Failed to create WebhookService: {e}")
        return None


# =============================================================================
# Health Check Dependencies
# =============================================================================

async def get_health_checkers():
    """Get all health check dependencies"""
    health_checkers = {}

    # MongoDB health check
    if get_mongodb:
        try:
            health_checkers["mongodb"] = await get_mongodb()
        except Exception as e:
            print(f"MongoDB health check failed: {e}")

    # Redis health check
    if get_redis:
        try:
            health_checkers["redis"] = await get_redis()
        except Exception as e:
            print(f"Redis health check failed: {e}")

    # PostgreSQL health check
    if get_postgresql_engine:
        try:
            health_checkers["postgresql"] = await get_postgresql_engine()
        except Exception as e:
            print(f"PostgreSQL health check failed: {e}")

    return health_checkers


async def check_repository_health() -> dict:
    """
    Check health of all repository dependencies

    Returns:
        Dictionary with health status of each repository type
    """
    health_status = {
        "postgresql": {
            "available": is_postgresql_available(),
            "repositories": {
                "tenant": TenantRepository is not None,
                "webhook": WebhookRepository is not None
            }
        },
        "mongodb": {
            "available": is_mongodb_available(),
            "repositories": {
                "conversation": ConversationRepository is not None,
                "message": MessageRepository is not None
            }
        },
        "redis": {
            "available": is_redis_available(),
            "repositories": {
                "session": SessionRepository is not None,
                "rate_limit": RateLimitRepository is not None,
                "cache": CacheRepository is not None
            }
        }
    }

    # Test actual connections
    try:
        tenant_repo = await get_tenant_repository()
        health_status["postgresql"]["connection"] = tenant_repo is not None
    except Exception as e:
        health_status["postgresql"]["connection"] = False
        health_status["postgresql"]["error"] = str(e)

    try:
        conversation_repo = await get_conversation_repository()
        health_status["mongodb"]["connection"] = conversation_repo is not None
    except Exception as e:
        health_status["mongodb"]["connection"] = False
        health_status["mongodb"]["error"] = str(e)

    try:
        session_repo = await get_session_repository()
        health_status["redis"]["connection"] = session_repo is not None
    except Exception as e:
        health_status["redis"]["connection"] = False
        health_status["redis"]["error"] = str(e)

    return health_status


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
            "audit_service": {"enabled": True},
            "tenant_service": {"enabled": True},
            "webhook_service": {"enabled": True}
        },
        "databases": {
            "postgresql": {"enabled": get_postgresql_engine is not None},
            "mongodb": {"enabled": get_mongodb is not None},
            "redis": {"enabled": get_redis is not None}
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
# Repository Requirement Functions
# =============================================================================

def require_tenant_repository(repo: Optional['TenantRepository']) -> 'TenantRepository':
    """
    Require tenant repository or raise HTTPException

    Args:
        repo: Injected repository (may be None)

    Returns:
        TenantRepository instance

    Raises:
        HTTPException: If repository is not available
    """
    if repo is None:
        raise HTTPException(
            status_code=503,
            detail="Tenant repository not available - check PostgreSQL database connection"
        )
    return repo


def require_webhook_repository(repo: Optional['WebhookRepository']) -> 'WebhookRepository':
    """
    Require webhook repository or raise HTTPException

    Args:
        repo: Injected repository (may be None)

    Returns:
        WebhookRepository instance

    Raises:
        HTTPException: If repository is not available
    """
    if repo is None:
        raise HTTPException(
            status_code=503,
            detail="Webhook repository not available - check PostgreSQL database connection"
        )
    return repo


def require_conversation_repository(repo: Optional['ConversationRepository']) -> 'ConversationRepository':
    """
    Require conversation repository or raise HTTPException

    Args:
        repo: Injected repository (may be None)

    Returns:
        ConversationRepository instance

    Raises:
        HTTPException: If repository is not available
    """
    if repo is None:
        raise HTTPException(
            status_code=503,
            detail="Conversation repository not available - check MongoDB database connection"
        )
    return repo


def require_message_repository(repo: Optional['MessageRepository']) -> 'MessageRepository':
    """
    Require message repository or raise HTTPException

    Args:
        repo: Injected repository (may be None)

    Returns:
        MessageRepository instance

    Raises:
        HTTPException: If repository is not available
    """
    if repo is None:
        raise HTTPException(
            status_code=503,
            detail="Message repository not available - check MongoDB database connection"
        )
    return repo


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


async def create_tenant_service_with_fallbacks():
    """Create tenant service with fallback dependencies"""
    try:
        tenant_repo = await get_tenant_repository()
        audit_service = await get_audit_service()
        return await get_tenant_service(tenant_repo, audit_service)
    except Exception as e:
        handle_dependency_error("TenantService", e)
        raise


async def create_webhook_service_with_fallbacks():
    """Create webhook service with fallback dependencies"""
    try:
        webhook_repo = await get_webhook_repository()
        tenant_repo = await get_tenant_repository()
        audit_service = await get_audit_service()
        return await get_webhook_service(webhook_repo, tenant_repo, audit_service)
    except Exception as e:
        handle_dependency_error("WebhookService", e)
        raise


# =============================================================================
# Availability Check Functions
# =============================================================================

def is_postgresql_available() -> bool:
    """Check if PostgreSQL repositories are available"""
    return TenantRepository is not None and WebhookRepository is not None and get_postgresql_engine is not None


def is_mongodb_available() -> bool:
    """Check if MongoDB repositories are available"""
    return ConversationRepository is not None and MessageRepository is not None and get_mongodb is not None


def is_redis_available() -> bool:
    """Check if Redis repositories are available"""
    return (SessionRepository is not None and
            RateLimitRepository is not None and
            get_redis is not None)


# =============================================================================
# Development and Testing Dependencies
# =============================================================================

def get_mock_dependencies() -> dict:
    """Get mock dependencies for testing"""
    return {
        "conversation_repo": None,
        "message_repo": None,
        "session_repo": None,
        "tenant_repo": None,
        "webhook_repo": None,
        "cache_repo": None,
        "channel_factory": get_channel_factory(),
        "processor_factory": get_processor_factory()
    }


async def override_dependencies_for_testing():
    """Override dependencies for testing environment"""
    # This would be used in test setup to provide mock implementations
    pass


# =============================================================================
# Initialization Functions
# =============================================================================

async def initialize_repositories():
    """Initialize all repositories during startup"""
    print("Initializing repositories...")

    # Initialize PostgreSQL repositories
    if is_postgresql_available():
        try:
            tenant_repo = await get_tenant_repository()
            webhook_repo = await get_webhook_repository()
            if tenant_repo and webhook_repo:
                print("✅ PostgreSQL repositories initialized successfully")
            else:
                print("❌ Failed to initialize PostgreSQL repositories")
        except Exception as e:
            print(f"❌ PostgreSQL repository initialization error: {e}")
    else:
        print("⚠️  PostgreSQL repositories not available")

    # Initialize MongoDB repositories
    if is_mongodb_available():
        try:
            conversation_repo = await get_conversation_repository()
            message_repo = await get_message_repository()
            if conversation_repo and message_repo:
                print("✅ MongoDB repositories initialized successfully")
            else:
                print("❌ Failed to initialize MongoDB repositories")
        except Exception as e:
            print(f"❌ MongoDB repository initialization error: {e}")
    else:
        print("⚠️  MongoDB repositories not available")

    # Initialize Redis repositories
    if is_redis_available():
        try:
            session_repo = await get_session_repository()
            rate_limit_repo = await get_rate_limit_repository()
            cache_repo = await get_cache_repository()
            if session_repo and rate_limit_repo and cache_repo:
                print("✅ Redis repositories initialized successfully")
            else:
                print("❌ Failed to initialize Redis repositories")
        except Exception as e:
            print(f"❌ Redis repository initialization error: {e}")
    else:
        print("⚠️  Redis repositories not available")


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
        "audit_service",
        "tenant_service",
        "webhook_service"
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
        elif service_name == "tenant_service":
            await get_tenant_service()
        elif service_name == "webhook_service":
            await get_webhook_service()
        else:
            return False

        return True
    except Exception:
        return False


# =============================================================================
# Type Annotations for FastAPI Dependencies
# =============================================================================

# MongoDB Repository Dependencies
ConversationRepositoryDep = Annotated[Optional[ConversationRepository], Depends(get_conversation_repository)]
MessageRepositoryDep = Annotated[Optional[MessageRepository], Depends(get_message_repository)]

# Redis Repository Dependencies
SessionRepositoryDep = Annotated[Optional[SessionRepository], Depends(get_session_repository)]
RateLimitRepositoryDep = Annotated[Optional[RateLimitRepository], Depends(get_rate_limit_repository)]
CacheRepositoryDep = Annotated[Optional[CacheRepository], Depends(get_cache_repository)]

# PostgreSQL Repository Dependencies
TenantRepositoryDep = Annotated[Optional[TenantRepository], Depends(get_tenant_repository)]
WebhookRepositoryDep = Annotated[Optional[WebhookRepository], Depends(get_webhook_repository)]

# Service Dependencies
MessageServiceDep = Annotated[MessageService, Depends(get_message_service)]
SessionServiceDep = Annotated[SessionService, Depends(get_session_service)]
ConversationServiceDep = Annotated[ConversationService, Depends(get_conversation_service)]
ChannelServiceDep = Annotated[ChannelService, Depends(get_channel_service)]
DeliveryServiceDep = Annotated[DeliveryService, Depends(get_delivery_service)]
AuditServiceDep = Annotated[AuditService, Depends(get_audit_service)]

# Authentication Dependencies
CurrentUserDep = Annotated[dict, Depends(get_current_user)]
TenantIdDep = Annotated[str, Depends(get_tenant_id)]
RequestContextDep = Annotated[dict, Depends(get_request_context)]