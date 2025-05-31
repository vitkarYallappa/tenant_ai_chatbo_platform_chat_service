"""
Database Initialization and Management
=====================================

Centralized database initialization, health monitoring, and management
for all database systems used by the chat service.

Features:
- MongoDB and Redis initialization
- Health monitoring for all databases
- Graceful startup and shutdown
- Database migration support
- Performance monitoring
"""

import asyncio
from typing import Dict, Any, Optional, List
import structlog
from dataclasses import dataclass

from .mongodb import (
    MongoDBConfig,
    initialize_mongodb,
    close_mongodb,
    mongodb_health_check,
    setup_mongodb_indexes,
    get_mongodb
)
from .redis_client import (
    RedisConfig,
    initialize_redis,
    close_redis,
    redis_health_check,
    get_redis
)

logger = structlog.get_logger(__name__)


@dataclass
class DatabaseConfig:
    """Combined database configuration"""
    mongodb: MongoDBConfig
    redis: RedisConfig

    @classmethod
    def from_env(cls) -> "DatabaseConfig":
        """Create configuration from environment variables"""
        return cls(
            mongodb=MongoDBConfig(),
            redis=RedisConfig()
        )


class DatabaseManager:
    """
    Centralized database manager for all database systems
    """

    def __init__(self, config: DatabaseConfig):
        """
        Initialize database manager

        Args:
            config: Database configuration
        """
        self.config = config
        self._initialized = False
        self._health_monitor_task: Optional[asyncio.Task] = None
        self._health_status: Dict[str, Any] = {}

    async def initialize(self) -> None:
        """
        Initialize all database connections

        Raises:
            Exception: If any database initialization fails
        """
        if self._initialized:
            logger.warning("Database manager already initialized")
            return

        logger.info("Initializing database connections")

        try:
            # Initialize MongoDB
            logger.info("Initializing MongoDB connection")
            await initialize_mongodb(self.config.mongodb)

            # Setup MongoDB indexes
            logger.info("Setting up MongoDB indexes")
            await setup_mongodb_indexes()

            # Initialize Redis
            logger.info("Initializing Redis connection")
            await initialize_redis(self.config.redis)

            # Start health monitoring
            await self._start_health_monitoring()

            self._initialized = True
            logger.info("All database connections initialized successfully")

        except Exception as e:
            logger.error("Failed to initialize databases", error=str(e))
            await self.shutdown()
            raise

    async def shutdown(self) -> None:
        """
        Gracefully shutdown all database connections
        """
        if not self._initialized:
            return

        logger.info("Shutting down database connections")

        # Stop health monitoring
        if self._health_monitor_task:
            self._health_monitor_task.cancel()
            try:
                await self._health_monitor_task
            except asyncio.CancelledError:
                pass
            self._health_monitor_task = None

        # Close connections
        try:
            await close_mongodb()
            logger.info("MongoDB connection closed")
        except Exception as e:
            logger.error("Error closing MongoDB", error=str(e))

        try:
            await close_redis()
            logger.info("Redis connection closed")
        except Exception as e:
            logger.error("Error closing Redis", error=str(e))

        self._initialized = False
        logger.info("Database shutdown completed")

    async def health_check(self) -> Dict[str, Any]:
        """
        Perform comprehensive health check on all databases

        Returns:
            Health status for all databases
        """
        health_status = {
            "overall_status": "healthy",
            "databases": {},
            "timestamp": asyncio.get_event_loop().time()
        }

        # Check MongoDB
        try:
            mongo_health = await mongodb_health_check()
            health_status["databases"]["mongodb"] = mongo_health

            if not mongo_health.get("healthy", False):
                health_status["overall_status"] = "degraded"

        except Exception as e:
            health_status["databases"]["mongodb"] = {
                "healthy": False,
                "error": str(e)
            }
            health_status["overall_status"] = "unhealthy"

        # Check Redis
        try:
            redis_health = await redis_health_check()
            health_status["databases"]["redis"] = redis_health

            if not redis_health.get("healthy", False):
                if health_status["overall_status"] == "healthy":
                    health_status["overall_status"] = "degraded"

        except Exception as e:
            health_status["databases"]["redis"] = {
                "healthy": False,
                "error": str(e)
            }
            health_status["overall_status"] = "unhealthy"

        # Store latest health status
        self._health_status = health_status

        return health_status

    async def get_database_info(self) -> Dict[str, Any]:
        """
        Get detailed information about all databases

        Returns:
            Database information dictionary
        """
        info = {
            "initialized": self._initialized,
            "config": {
                "mongodb": {
                    "host": self.config.mongodb.host,
                    "port": self.config.mongodb.port,
                    "database": self.config.mongodb.database_name,
                    "max_pool_size": self.config.mongodb.max_pool_size
                },
                "redis": {
                    "host": self.config.redis.host,
                    "port": self.config.redis.port,
                    "db": self.config.redis.db,
                    "max_connections": self.config.redis.max_connections
                }
            }
        }

        if self._initialized:
            health_status = await self.health_check()
            info["health"] = health_status

        return info

    async def _start_health_monitoring(self) -> None:
        """Start background health monitoring"""
        if self._health_monitor_task is None:
            self._health_monitor_task = asyncio.create_task(self._health_monitor())

    async def _health_monitor(self) -> None:
        """Background health monitoring coroutine"""
        while True:
            try:
                await asyncio.sleep(60)  # Check every minute
                await self.health_check()

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error("Health monitoring error", error=str(e))

    def get_latest_health_status(self) -> Dict[str, Any]:
        """Get the latest cached health status"""
        return self._health_status.copy() if self._health_status else {}

    @property
    def is_healthy(self) -> bool:
        """Check if all databases are healthy"""
        if not self._health_status:
            return False

        return self._health_status.get("overall_status") == "healthy"

    @property
    def is_initialized(self) -> bool:
        """Check if database manager is initialized"""
        return self._initialized


# Global database manager instance
_db_manager: Optional[DatabaseManager] = None


async def initialize_databases(config: Optional[DatabaseConfig] = None) -> DatabaseManager:
    """
    Initialize all databases

    Args:
        config: Database configuration (uses default if None)

    Returns:
        Database manager instance
    """
    global _db_manager

    if config is None:
        config = DatabaseConfig.from_env()

    if _db_manager is None:
        _db_manager = DatabaseManager(config)

    await _db_manager.initialize()
    return _db_manager


async def shutdown_databases() -> None:
    """Shutdown all database connections"""
    global _db_manager

    if _db_manager:
        await _db_manager.shutdown()
        _db_manager = None


async def database_health_check() -> Dict[str, Any]:
    """
    Get health status for all databases

    Returns:
        Combined health status
    """
    global _db_manager

    if _db_manager:
        return await _db_manager.health_check()
    else:
        return {
            "overall_status": "unhealthy",
            "error": "Database manager not initialized",
            "databases": {}
        }


def get_database_manager() -> Optional[DatabaseManager]:
    """Get the current database manager instance"""
    return _db_manager


async def ensure_databases_initialized() -> DatabaseManager:
    """
    Ensure databases are initialized, initializing if necessary

    Returns:
        Database manager instance
    """
    global _db_manager

    if _db_manager is None or not _db_manager.is_initialized:
        return await initialize_databases()

    return _db_manager


# Convenience functions for direct database access
async def get_mongo_db():
    """Get MongoDB database instance"""
    await ensure_databases_initialized()
    return await get_mongodb()


async def get_redis_client():
    """Get Redis client instance"""
    await ensure_databases_initialized()
    return await get_redis()


# Database migration support
class DatabaseMigrationManager:
    """
    Database migration manager for schema updates and data migrations
    """

    def __init__(self, db_manager: DatabaseManager):
        """
        Initialize migration manager

        Args:
            db_manager: Database manager instance
        """
        self.db_manager = db_manager
        self.logger = structlog.get_logger("DatabaseMigration")

    async def check_migration_status(self) -> Dict[str, Any]:
        """
        Check current migration status

        Returns:
            Migration status information
        """
        try:
            mongodb = await get_mongodb()

            # Check if migrations collection exists
            collections = await mongodb.list_collection_names()

            migration_status = {
                "migrations_collection_exists": "migrations" in collections,
                "applied_migrations": [],
                "pending_migrations": []
            }

            if migration_status["migrations_collection_exists"]:
                # Get applied migrations
                migrations_collection = mongodb.migrations
                applied = await migrations_collection.find({}).to_list(length=None)
                migration_status["applied_migrations"] = [
                    {
                        "migration_id": m.get("migration_id"),
                        "applied_at": m.get("applied_at"),
                        "version": m.get("version")
                    }
                    for m in applied
                ]

            return migration_status

        except Exception as e:
            self.logger.error("Failed to check migration status", error=str(e))
            return {"error": str(e)}

    async def apply_migrations(self, dry_run: bool = True) -> Dict[str, Any]:
        """
        Apply pending database migrations

        Args:
            dry_run: If True, only check what would be migrated

        Returns:
            Migration results
        """
        self.logger.info("Starting database migrations", dry_run=dry_run)

        try:
            results = {
                "dry_run": dry_run,
                "migrations_applied": [],
                "errors": []
            }

            # For now, we'll just ensure indexes are created
            if not dry_run:
                await setup_mongodb_indexes()
                results["migrations_applied"].append({
                    "migration_id": "create_indexes_v1",
                    "description": "Create MongoDB indexes",
                    "applied_at": asyncio.get_event_loop().time()
                })
            else:
                results["migrations_applied"].append({
                    "migration_id": "create_indexes_v1",
                    "description": "Create MongoDB indexes",
                    "would_apply": True
                })

            self.logger.info("Database migrations completed", results=results)
            return results

        except Exception as e:
            self.logger.error("Migration failed", error=str(e))
            return {
                "dry_run": dry_run,
                "error": str(e),
                "migrations_applied": [],
                "errors": [str(e)]
            }


async def run_database_migrations(dry_run: bool = True) -> Dict[str, Any]:
    """
    Run database migrations

    Args:
        dry_run: If True, only check what would be migrated

    Returns:
        Migration results
    """
    db_manager = await ensure_databases_initialized()
    migration_manager = DatabaseMigrationManager(db_manager)
    return await migration_manager.apply_migrations(dry_run)


# Export main components
__all__ = [
    # Configuration
    'DatabaseConfig',
    'MongoDBConfig',
    'RedisConfig',

    # Manager classes
    'DatabaseManager',
    'DatabaseMigrationManager',

    # Initialization functions
    'initialize_databases',
    'shutdown_databases',
    'ensure_databases_initialized',

    # Health and monitoring
    'database_health_check',
    'get_database_manager',

    # Direct access functions
    'get_mongo_db',
    'get_redis_client',

    # Migration functions
    'run_database_migrations',
]