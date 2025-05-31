# src/database/__init__.py
"""
Database package for the Chat Service.
Provides connection managers and health monitoring for all databases.
"""

from src.database.mongodb import (
    mongodb_manager,
    MongoDBManager,
    get_mongodb,
    get_mongodb_collection,
    get_conversations_collection,
    get_messages_collection,
    get_sessions_collection
)

from src.database.redis_client import (
    redis_manager,
    RedisManager,
    get_redis,
    get_session_cache,
    set_session_cache,
    get_rate_limit_count,
    increment_rate_limit
)

from src.database.health_checks import (
    health_checker,
    health_scheduler,
    DatabaseHealthChecker,
    HealthCheckScheduler,
    HealthCheckResult,
    check_all_databases,
    check_mongodb,
    check_redis,
    get_detailed_health_status,
    get_health_history,
    get_uptime_stats
)

# Version information
__version__ = "1.0.0"


# Database connection lifecycle functions
async def connect_all_databases() -> None:
    """Connect to all databases."""
    try:
        # Connect to MongoDB
        await mongodb_manager.connect()

        # Connect to Redis
        await redis_manager.connect()

        # Start health monitoring
        await health_scheduler.start_monitoring()

        print("All databases connected successfully")

    except Exception as e:
        print(f"Database connection failed: {e}")
        raise


async def disconnect_all_databases() -> None:
    """Disconnect from all databases."""
    try:
        # Stop health monitoring
        await health_scheduler.stop_monitoring()

        # Disconnect from Redis
        await redis_manager.disconnect()

        # Disconnect from MongoDB
        await mongodb_manager.disconnect()

        print("All databases disconnected successfully")

    except Exception as e:
        print(f"Database disconnection error: {e}")


async def initialize_databases() -> None:
    """Initialize databases with indexes and settings."""
    try:
        # Connect first
        await connect_all_databases()

        # Create MongoDB indexes
        await mongodb_manager.create_indexes()

        # Ensure collection settings
        await mongodb_manager.ensure_collection_settings()

        print("Databases initialized successfully")

    except Exception as e:
        print(f"Database initialization failed: {e}")
        raise


# Export all database components
__all__ = [
    # MongoDB
    "mongodb_manager",
    "MongoDBManager",
    "get_mongodb",
    "get_mongodb_collection",
    "get_conversations_collection",
    "get_messages_collection",
    "get_sessions_collection",

    # Redis
    "redis_manager",
    "RedisManager",
    "get_redis",
    "get_session_cache",
    "set_session_cache",
    "get_rate_limit_count",
    "increment_rate_limit",

    # Health checks
    "health_checker",
    "health_scheduler",
    "DatabaseHealthChecker",
    "HealthCheckScheduler",
    "HealthCheckResult",
    "check_all_databases",
    "check_mongodb",
    "check_redis",
    "get_detailed_health_status",
    "get_health_history",
    "get_uptime_stats",

    # Lifecycle functions
    "connect_all_databases",
    "disconnect_all_databases",
    "initialize_databases"
]