"""
MongoDB Connection Management
============================

Centralized MongoDB connection management with connection pooling,
health monitoring, and configuration management.

Features:
- Connection pooling with optimal settings
- Health monitoring and reconnection logic
- Index management and optimization
- Multi-tenant database isolation options
- Performance monitoring
- Graceful connection handling
"""

import asyncio
from typing import Optional, Dict, Any, List
from motor.motor_asyncio import AsyncIOMotorClient, AsyncIOMotorDatabase
from pymongo import MongoClient, IndexModel, ASCENDING, DESCENDING, TEXT
from pymongo.errors import ConnectionFailure, ServerSelectionTimeoutError
import structlog
from dataclasses import dataclass, field
import os
from urllib.parse import quote_plus

logger = structlog.get_logger(__name__)


@dataclass
class MongoDBConfig:
    """MongoDB configuration with secure defaults"""
    # Connection settings
    host: str = "localhost"
    port: int = 27017
    database_name: str = "chatbot_platform"
    username: Optional[str] = None
    password: Optional[str] = None
    auth_source: str = "admin"

    # Connection pool settings
    max_pool_size: int = 100
    min_pool_size: int = 10
    max_idle_time_ms: int = 30000

    # Timeout settings
    connect_timeout_ms: int = 10000
    socket_timeout_ms: int = 5000
    server_selection_timeout_ms: int = 5000

    # Reliability settings
    retry_writes: bool = True
    retry_reads: bool = True
    read_preference: str = "primary"

    # TLS/SSL settings
    tls: bool = False
    tls_cert_reqs: str = "CERT_REQUIRED"
    tls_ca_file: Optional[str] = None
    tls_cert_file: Optional[str] = None
    tls_key_file: Optional[str] = None

    # Advanced settings
    replica_set: Optional[str] = None
    read_concern_level: str = "local"
    write_concern_w: int = 1
    write_concern_j: bool = True

    # Environment-based overrides
    def __post_init__(self):
        """Override with environment variables if available"""
        self.host = os.getenv("MONGODB_HOST", self.host)
        self.port = int(os.getenv("MONGODB_PORT", str(self.port)))
        self.database_name = os.getenv("MONGODB_DATABASE", self.database_name)
        self.username = os.getenv("MONGODB_USERNAME", self.username)
        self.password = os.getenv("MONGODB_PASSWORD", self.password)
        self.replica_set = os.getenv("MONGODB_REPLICA_SET", self.replica_set)

        # TLS settings from environment
        if os.getenv("MONGODB_TLS", "").lower() in ("true", "1", "yes"):
            self.tls = True
        self.tls_ca_file = os.getenv("MONGODB_TLS_CA_FILE", self.tls_ca_file)
        self.tls_cert_file = os.getenv("MONGODB_TLS_CERT_FILE", self.tls_cert_file)
        self.tls_key_file = os.getenv("MONGODB_TLS_KEY_FILE", self.tls_key_file)

    def get_connection_string(self) -> str:
        """
        Build MongoDB connection string

        Returns:
            MongoDB URI string
        """
        # Build base URI
        if self.username and self.password:
            # URL encode username and password to handle special characters
            encoded_username = quote_plus(self.username)
            encoded_password = quote_plus(self.password)
            auth_part = f"{encoded_username}:{encoded_password}@"
        else:
            auth_part = ""

        # Build host part
        if self.replica_set:
            # For replica sets, you might have multiple hosts
            host_part = f"{self.host}:{self.port}"
        else:
            host_part = f"{self.host}:{self.port}"

        # Build URI
        uri = f"mongodb://{auth_part}{host_part}/{self.database_name}"

        # Add query parameters
        params = []

        if self.auth_source:
            params.append(f"authSource={self.auth_source}")

        if self.replica_set:
            params.append(f"replicaSet={self.replica_set}")

        params.extend([
            f"maxPoolSize={self.max_pool_size}",
            f"minPoolSize={self.min_pool_size}",
            f"maxIdleTimeMS={self.max_idle_time_ms}",
            f"connectTimeoutMS={self.connect_timeout_ms}",
            f"socketTimeoutMS={self.socket_timeout_ms}",
            f"serverSelectionTimeoutMS={self.server_selection_timeout_ms}",
            f"retryWrites={str(self.retry_writes).lower()}",
            f"retryReads={str(self.retry_reads).lower()}",
            f"readPreference={self.read_preference}"
        ])

        if self.tls:
            params.append("tls=true")
            params.append(f"tlsCertificateKeyFile={self.tls_cert_file}")
            if self.tls_ca_file:
                params.append(f"tlsCAFile={self.tls_ca_file}")

        if params:
            uri += "?" + "&".join(params)

        return uri

    def get_client_options(self) -> Dict[str, Any]:
        """
        Get client options for AsyncIOMotorClient

        Returns:
            Dictionary of client options
        """
        options = {
            'maxPoolSize': self.max_pool_size,
            'minPoolSize': self.min_pool_size,
            'maxIdleTimeMS': self.max_idle_time_ms,
            'connectTimeoutMS': self.connect_timeout_ms,
            'socketTimeoutMS': self.socket_timeout_ms,
            'serverSelectionTimeoutMS': self.server_selection_timeout_ms,
            'retryWrites': self.retry_writes,
            'retryReads': self.retry_reads
        }

        if self.tls:
            options.update({
                'tls': True,
                'tlsCertificateKeyFile': self.tls_cert_file,
                'tlsCAFile': self.tls_ca_file
            })

        return options


class MongoDBConnectionManager:
    """
    MongoDB connection manager with health monitoring and reconnection logic
    """

    def __init__(self, config: MongoDBConfig):
        """
        Initialize connection manager

        Args:
            config: MongoDB configuration
        """
        self.config = config
        self.client: Optional[AsyncIOMotorClient] = None
        self.database: Optional[AsyncIOMotorDatabase] = None
        self._connection_lock = asyncio.Lock()
        self._health_check_task: Optional[asyncio.Task] = None
        self._is_healthy = False

    async def connect(self) -> AsyncIOMotorDatabase:
        """
        Establish connection to MongoDB

        Returns:
            MongoDB database instance

        Raises:
            ConnectionFailure: If connection cannot be established
        """
        async with self._connection_lock:
            if self.client is None:
                try:
                    logger.info("Connecting to MongoDB", host=self.config.host, database=self.config.database_name)

                    # Create client with connection string
                    connection_string = self.config.get_connection_string()
                    self.client = AsyncIOMotorClient(connection_string)

                    # Get database
                    self.database = self.client[self.config.database_name]

                    # Test connection
                    await self._test_connection()

                    # Start health monitoring
                    self._start_health_monitoring()

                    self._is_healthy = True
                    logger.info("Successfully connected to MongoDB")

                except Exception as e:
                    logger.error("Failed to connect to MongoDB", error=str(e))
                    await self.disconnect()
                    raise ConnectionFailure(f"Failed to connect to MongoDB: {e}")

            return self.database

    async def disconnect(self) -> None:
        """Disconnect from MongoDB"""
        async with self._connection_lock:
            if self._health_check_task:
                self._health_check_task.cancel()
                try:
                    await self._health_check_task
                except asyncio.CancelledError:
                    pass
                self._health_check_task = None

            if self.client:
                self.client.close()
                self.client = None
                self.database = None
                self._is_healthy = False

                logger.info("Disconnected from MongoDB")

    async def get_database(self) -> AsyncIOMotorDatabase:
        """
        Get database instance, connecting if necessary

        Returns:
            MongoDB database instance
        """
        if self.database is None or not self._is_healthy:
            await self.connect()

        return self.database

    async def _test_connection(self) -> None:
        """Test MongoDB connection"""
        if self.client:
            # Ping the database
            await self.client.admin.command('ping')

            # Test database access
            await self.database.command('ping')

    def _start_health_monitoring(self) -> None:
        """Start background health monitoring task"""
        if self._health_check_task is None:
            self._health_check_task = asyncio.create_task(self._health_monitor())

    async def _health_monitor(self) -> None:
        """Background health monitoring coroutine"""
        while True:
            try:
                await asyncio.sleep(30)  # Check every 30 seconds
                await self._test_connection()

                if not self._is_healthy:
                    self._is_healthy = True
                    logger.info("MongoDB connection restored")

            except Exception as e:
                if self._is_healthy:
                    self._is_healthy = False
                    logger.error("MongoDB health check failed", error=str(e))

                # Try to reconnect after a delay
                await asyncio.sleep(5)
                try:
                    await self.connect()
                except Exception as reconnect_error:
                    logger.error("Failed to reconnect to MongoDB", error=str(reconnect_error))

    async def health_check(self) -> Dict[str, Any]:
        """
        Perform comprehensive health check

        Returns:
            Health status information
        """
        health_info = {
            "connected": False,
            "healthy": self._is_healthy,
            "database": self.config.database_name,
            "host": self.config.host,
            "port": self.config.port
        }

        try:
            if self.client and self.database:
                # Test connection
                start_time = asyncio.get_event_loop().time()
                await self._test_connection()
                response_time = (asyncio.get_event_loop().time() - start_time) * 1000

                # Get server status
                server_status = await self.database.command('serverStatus')

                health_info.update({
                    "connected": True,
                    "response_time_ms": round(response_time, 2),
                    "server_version": server_status.get('version', 'unknown'),
                    "uptime_seconds": server_status.get('uptime', 0),
                    "connections": server_status.get('connections', {}),
                    "memory": server_status.get('mem', {}),
                    "operations": server_status.get('opcounters', {})
                })

        except Exception as e:
            health_info.update({
                "error": str(e),
                "healthy": False
            })

        return health_info


# Index Management
class MongoIndexManager:
    """MongoDB index management utilities"""

    def __init__(self, database: AsyncIOMotorDatabase):
        """
        Initialize index manager

        Args:
            database: MongoDB database instance
        """
        self.database = database
        self.logger = structlog.get_logger("MongoIndexManager")

    async def create_conversation_indexes(self) -> None:
        """Create indexes for conversations collection"""
        collection = self.database.conversations

        indexes = [
            # Tenant and conversation lookup
            IndexModel([("tenant_id", ASCENDING), ("conversation_id", ASCENDING)], unique=True),
            IndexModel([("conversation_id", ASCENDING)], unique=True),

            # User and tenant queries
            IndexModel([("tenant_id", ASCENDING), ("user_id", ASCENDING), ("started_at", DESCENDING)]),
            IndexModel([("tenant_id", ASCENDING), ("status", ASCENDING), ("last_activity_at", DESCENDING)]),

            # Channel and status filtering
            IndexModel([("tenant_id", ASCENDING), ("channel", ASCENDING), ("started_at", DESCENDING)]),
            IndexModel([("tenant_id", ASCENDING), ("status", ASCENDING)]),

            # Business context queries
            IndexModel([("tenant_id", ASCENDING), ("business_context.category", ASCENDING)]),
            IndexModel([("tenant_id", ASCENDING), ("business_context.outcome", ASCENDING)]),

            # Flow and state management
            IndexModel([("flow_id", ASCENDING), ("current_state", ASCENDING)]),

            # Date-based queries
            IndexModel([("started_at", DESCENDING)]),
            IndexModel([("last_activity_at", DESCENDING)]),

            # Cleanup and retention
            IndexModel([("compliance.data_retention_until", ASCENDING)]),

            # Text search
            IndexModel([("$**", TEXT)], name="text_search_index")
        ]

        try:
            await collection.create_indexes(indexes)
            self.logger.info("Conversation indexes created successfully")
        except Exception as e:
            self.logger.error("Failed to create conversation indexes", error=str(e))
            raise

    async def create_message_indexes(self) -> None:
        """Create indexes for messages collection"""
        collection = self.database.messages

        indexes = [
            # Message lookup and conversation queries
            IndexModel([("conversation_id", ASCENDING), ("sequence_number", ASCENDING)]),
            IndexModel([("message_id", ASCENDING)], unique=True),
            IndexModel([("tenant_id", ASCENDING), ("timestamp", DESCENDING)]),

            # Direction and type filtering
            IndexModel([("tenant_id", ASCENDING), ("direction", ASCENDING), ("timestamp", DESCENDING)]),
            IndexModel([("content.type", ASCENDING), ("timestamp", DESCENDING)]),

            # AI analysis queries
            IndexModel([("tenant_id", ASCENDING), ("ai_analysis.intent.detected_intent", ASCENDING)]),
            IndexModel([("ai_analysis.sentiment.label", ASCENDING), ("timestamp", DESCENDING)]),

            # Moderation and cleanup
            IndexModel([("moderation.flagged", ASCENDING), ("timestamp", DESCENDING)]),
            IndexModel([("privacy.auto_delete_at", ASCENDING)]),

            # Performance optimization
            IndexModel([("conversation_id", ASCENDING), ("direction", ASCENDING), ("sequence_number", ASCENDING)]),

            # Text search
            IndexModel([("content.text", TEXT)], name="message_text_search")
        ]

        try:
            await collection.create_indexes(indexes)
            self.logger.info("Message indexes created successfully")
        except Exception as e:
            self.logger.error("Failed to create message indexes", error=str(e))
            raise

    async def create_all_indexes(self) -> None:
        """Create all required indexes"""
        await self.create_conversation_indexes()
        await self.create_message_indexes()
        self.logger.info("All MongoDB indexes created successfully")

    async def list_collection_indexes(self, collection_name: str) -> List[Dict[str, Any]]:
        """
        List indexes for a collection

        Args:
            collection_name: Name of the collection

        Returns:
            List of index information
        """
        try:
            collection = self.database[collection_name]
            indexes = await collection.list_indexes().to_list(length=None)
            return indexes
        except Exception as e:
            self.logger.error("Failed to list indexes", collection=collection_name, error=str(e))
            return []

    async def drop_index(self, collection_name: str, index_name: str) -> bool:
        """
        Drop an index

        Args:
            collection_name: Name of the collection
            index_name: Name of the index

        Returns:
            True if successful
        """
        try:
            collection = self.database[collection_name]
            await collection.drop_index(index_name)
            self.logger.info("Index dropped successfully", collection=collection_name, index=index_name)
            return True
        except Exception as e:
            self.logger.error("Failed to drop index", collection=collection_name, index=index_name, error=str(e))
            return False


# Global connection manager instance
_connection_manager: Optional[MongoDBConnectionManager] = None


async def initialize_mongodb(config: Optional[MongoDBConfig] = None) -> AsyncIOMotorDatabase:
    """
    Initialize MongoDB connection

    Args:
        config: MongoDB configuration (uses default if None)

    Returns:
        MongoDB database instance
    """
    global _connection_manager

    if config is None:
        config = MongoDBConfig()

    if _connection_manager is None:
        _connection_manager = MongoDBConnectionManager(config)

    return await _connection_manager.connect()


async def get_mongodb() -> AsyncIOMotorDatabase:
    """
    Get MongoDB database instance

    Returns:
        MongoDB database instance

    Raises:
        RuntimeError: If MongoDB is not initialized
    """
    global _connection_manager

    if _connection_manager is None:
        # Auto-initialize with default config
        return await initialize_mongodb()

    return await _connection_manager.get_database()


async def close_mongodb() -> None:
    """Close MongoDB connection"""
    global _connection_manager

    if _connection_manager:
        await _connection_manager.disconnect()
        _connection_manager = None


async def mongodb_health_check() -> Dict[str, Any]:
    """
    Get MongoDB health status

    Returns:
        Health check results
    """
    global _connection_manager

    if _connection_manager:
        return await _connection_manager.health_check()
    else:
        return {
            "connected": False,
            "healthy": False,
            "error": "MongoDB not initialized"
        }


async def setup_mongodb_indexes() -> None:
    """Setup all required MongoDB indexes"""
    database = await get_mongodb()
    index_manager = MongoIndexManager(database)
    await index_manager.create_all_indexes()