"""
Redis Connection Management
==========================

Centralized Redis connection management with connection pooling,
health monitoring, and configuration management.

Features:
- Connection pooling with optimal settings
- Health monitoring and reconnection logic
- Cluster support
- Sentinel support for high availability
- Performance monitoring
- Graceful connection handling
"""

import asyncio
from typing import Optional, Dict, Any, List, Union
from redis.asyncio import Redis, ConnectionPool, Sentinel
from redis.exceptions import (
    ConnectionError as RedisConnectionError,
    TimeoutError as RedisTimeoutError,
    ResponseError, RedisError
)
import structlog
from dataclasses import dataclass
import os
from urllib.parse import urlparse

logger = structlog.get_logger(__name__)


@dataclass
class RedisConfig:
    """Redis configuration with secure defaults"""
    # Connection settings
    host: str = "localhost"
    port: int = 6379
    db: int = 0
    username: Optional[str] = None
    password: Optional[str] = None

    # Connection pool settings
    max_connections: int = 50
    retry_on_timeout: bool = True
    health_check_interval: int = 30

    # Timeout settings
    socket_connect_timeout: float = 5.0
    socket_timeout: float = 5.0
    socket_keepalive: bool = True
    socket_keepalive_options: Optional[Dict[str, int]] = None

    # TLS/SSL settings
    ssl: bool = False
    ssl_certfile: Optional[str] = None
    ssl_keyfile: Optional[str] = None
    ssl_ca_certs: Optional[str] = None
    ssl_cert_reqs: str = "required"
    ssl_check_hostname: bool = True

    # Cluster settings
    cluster_mode: bool = False
    cluster_nodes: Optional[List[Dict[str, Union[str, int]]]] = None

    # Sentinel settings
    sentinel_mode: bool = False
    sentinel_hosts: Optional[List[Dict[str, Union[str, int]]]] = None
    sentinel_service_name: Optional[str] = None
    sentinel_socket_timeout: float = 0.1

    # Advanced settings
    decode_responses: bool = True
    encoding: str = "utf-8"
    encoding_errors: str = "strict"
    connection_class: Optional[type] = None

    def __post_init__(self):
        """Override with environment variables if available"""
        # Check for Redis URL first
        redis_url = os.getenv("REDIS_URL")
        if redis_url:
            self._parse_redis_url(redis_url)
        else:
            # Individual settings
            self.host = os.getenv("REDIS_HOST", self.host)
            self.port = int(os.getenv("REDIS_PORT", str(self.port)))
            self.db = int(os.getenv("REDIS_DB", str(self.db)))
            self.username = os.getenv("REDIS_USERNAME", self.username)
            self.password = os.getenv("REDIS_PASSWORD", self.password)

        # Pool settings
        self.max_connections = int(os.getenv("REDIS_MAX_CONNECTIONS", str(self.max_connections)))

        # SSL settings
        if os.getenv("REDIS_SSL", "").lower() in ("true", "1", "yes"):
            self.ssl = True
        self.ssl_certfile = os.getenv("REDIS_SSL_CERTFILE", self.ssl_certfile)
        self.ssl_keyfile = os.getenv("REDIS_SSL_KEYFILE", self.ssl_keyfile)
        self.ssl_ca_certs = os.getenv("REDIS_SSL_CA_CERTS", self.ssl_ca_certs)

        # Cluster mode
        if os.getenv("REDIS_CLUSTER", "").lower() in ("true", "1", "yes"):
            self.cluster_mode = True

        # Sentinel mode
        if os.getenv("REDIS_SENTINEL", "").lower() in ("true", "1", "yes"):
            self.sentinel_mode = True
            self.sentinel_service_name = os.getenv("REDIS_SENTINEL_SERVICE", self.sentinel_service_name)

        # Set default socket keepalive options
        if self.socket_keepalive_options is None:
            self.socket_keepalive_options = {
                1: 1,  # TCP_KEEPIDLE
                2: 3,  # TCP_KEEPINTVL  
                3: 5  # TCP_KEEPCNT
            }

    def _parse_redis_url(self, url: str) -> None:
        """Parse Redis URL and update configuration"""
        try:
            parsed = urlparse(url)

            self.host = parsed.hostname or self.host
            self.port = parsed.port or self.port

            if parsed.username:
                self.username = parsed.username
            if parsed.password:
                self.password = parsed.password

            # Extract database number from path
            if parsed.path and len(parsed.path) > 1:
                try:
                    self.db = int(parsed.path[1:])
                except ValueError:
                    pass

            # Check for SSL
            if parsed.scheme == "rediss":
                self.ssl = True

        except Exception as e:
            logger.warning("Failed to parse Redis URL", url=url, error=str(e))

    def get_connection_kwargs(self) -> Dict[str, Any]:
        """
        Get connection parameters for Redis client

        Returns:
            Dictionary of connection parameters
        """
        kwargs = {
            'host': self.host,
            'port': self.port,
            'db': self.db,
            'socket_connect_timeout': self.socket_connect_timeout,
            'socket_timeout': self.socket_timeout,
            'socket_keepalive': self.socket_keepalive,
            'socket_keepalive_options': self.socket_keepalive_options,
            'decode_responses': self.decode_responses,
            'encoding': self.encoding,
            'encoding_errors': self.encoding_errors,
            'retry_on_timeout': self.retry_on_timeout,
            'health_check_interval': self.health_check_interval
        }

        # Authentication
        if self.username:
            kwargs['username'] = self.username
        if self.password:
            kwargs['password'] = self.password

        # SSL/TLS
        if self.ssl:
            kwargs.update({
                'ssl': True,
                'ssl_certfile': self.ssl_certfile,
                'ssl_keyfile': self.ssl_keyfile,
                'ssl_ca_certs': self.ssl_ca_certs,
                'ssl_cert_reqs': self.ssl_cert_reqs,
                'ssl_check_hostname': self.ssl_check_hostname
            })

        return kwargs

    def get_pool_kwargs(self) -> Dict[str, Any]:
        """
        Get connection pool parameters

        Returns:
            Dictionary of pool parameters
        """
        pool_kwargs = self.get_connection_kwargs()
        pool_kwargs['max_connections'] = self.max_connections

        if self.connection_class:
            pool_kwargs['connection_class'] = self.connection_class

        return pool_kwargs


class RedisConnectionManager:
    """
    Redis connection manager with health monitoring and reconnection logic
    """

    def __init__(self, config: RedisConfig):
        """
        Initialize connection manager

        Args:
            config: Redis configuration
        """
        self.config = config
        self.client: Optional[Redis] = None
        self.pool: Optional[ConnectionPool] = None
        self.sentinel: Optional[Sentinel] = None
        self._connection_lock = asyncio.Lock()
        self._health_check_task: Optional[asyncio.Task] = None
        self._is_healthy = False

    async def connect(self) -> Redis:
        """
        Establish connection to Redis

        Returns:
            Redis client instance

        Raises:
            ConnectionError: If connection cannot be established
        """
        async with self._connection_lock:
            if self.client is None:
                try:
                    logger.info(
                        "Connecting to Redis",
                        host=self.config.host,
                        port=self.config.port,
                        cluster_mode=self.config.cluster_mode,
                        sentinel_mode=self.config.sentinel_mode
                    )

                    if self.config.cluster_mode:
                        await self._connect_cluster()
                    elif self.config.sentinel_mode:
                        await self._connect_sentinel()
                    else:
                        await self._connect_single()

                    # Test connection
                    await self._test_connection()

                    # Start health monitoring
                    self._start_health_monitoring()

                    self._is_healthy = True
                    logger.info("Successfully connected to Redis")

                except Exception as e:
                    logger.error("Failed to connect to Redis", error=str(e))
                    await self.disconnect()
                    raise RedisConnectionError(f"Failed to connect to Redis: {e}")

            return self.client

    async def _connect_single(self) -> None:
        """Connect to single Redis instance"""
        pool_kwargs = self.config.get_pool_kwargs()
        self.pool = ConnectionPool(**pool_kwargs)
        self.client = Redis(connection_pool=self.pool)

    async def _connect_cluster(self) -> None:
        """Connect to Redis cluster"""
        try:
            from redis.asyncio.cluster import RedisCluster

            if self.config.cluster_nodes:
                startup_nodes = self.config.cluster_nodes
            else:
                startup_nodes = [{"host": self.config.host, "port": self.config.port}]

            cluster_kwargs = {
                'startup_nodes': startup_nodes,
                'decode_responses': self.config.decode_responses,
                'socket_timeout': self.config.socket_timeout,
                'socket_connect_timeout': self.config.socket_connect_timeout,
                'retry_on_timeout': self.config.retry_on_timeout,
                'health_check_interval': self.config.health_check_interval
            }

            if self.config.password:
                cluster_kwargs['password'] = self.config.password

            self.client = RedisCluster(**cluster_kwargs)

        except ImportError:
            raise RedisConnectionError("Redis cluster support not available. Install redis[cluster]")

    async def _connect_sentinel(self) -> None:
        """Connect using Redis Sentinel"""
        if not self.config.sentinel_hosts or not self.config.sentinel_service_name:
            raise RedisConnectionError("Sentinel hosts and service name are required for sentinel mode")

        sentinel_list = [
            (host['host'], host['port'])
            for host in self.config.sentinel_hosts
        ]

        sentinel_kwargs = {
            'socket_timeout': self.config.sentinel_socket_timeout,
            'socket_connect_timeout': self.config.socket_connect_timeout
        }

        if self.config.password:
            sentinel_kwargs['password'] = self.config.password

        self.sentinel = Sentinel(sentinel_list, **sentinel_kwargs)

        # Get master connection
        master_kwargs = {
            'socket_timeout': self.config.socket_timeout,
            'socket_connect_timeout': self.config.socket_connect_timeout,
            'decode_responses': self.config.decode_responses,
            'retry_on_timeout': self.config.retry_on_timeout,
            'health_check_interval': self.config.health_check_interval
        }

        if self.config.password:
            master_kwargs['password'] = self.config.password

        self.client = self.sentinel.master_for(
            self.config.sentinel_service_name,
            **master_kwargs
        )

    async def disconnect(self) -> None:
        """Disconnect from Redis"""
        async with self._connection_lock:
            if self._health_check_task:
                self._health_check_task.cancel()
                try:
                    await self._health_check_task
                except asyncio.CancelledError:
                    pass
                self._health_check_task = None

            if self.client:
                await self.client.close()
                self.client = None
                self._is_healthy = False

            if self.pool:
                await self.pool.disconnect()
                self.pool = None

            self.sentinel = None

            logger.info("Disconnected from Redis")

    async def get_client(self) -> Redis:
        """
        Get Redis client instance, connecting if necessary

        Returns:
            Redis client instance
        """
        if self.client is None or not self._is_healthy:
            await self.connect()

        return self.client

    async def _test_connection(self) -> None:
        """Test Redis connection"""
        if self.client:
            await self.client.ping()

    def _start_health_monitoring(self) -> None:
        """Start background health monitoring task"""
        if self._health_check_task is None:
            self._health_check_task = asyncio.create_task(self._health_monitor())

    async def _health_monitor(self) -> None:
        """Background health monitoring coroutine"""
        while True:
            try:
                await asyncio.sleep(self.config.health_check_interval)
                await self._test_connection()

                if not self._is_healthy:
                    self._is_healthy = True
                    logger.info("Redis connection restored")

            except Exception as e:
                if self._is_healthy:
                    self._is_healthy = False
                    logger.error("Redis health check failed", error=str(e))

                # Try to reconnect after a delay
                await asyncio.sleep(5)
                try:
                    await self.connect()
                except Exception as reconnect_error:
                    logger.error("Failed to reconnect to Redis", error=str(reconnect_error))

    async def health_check(self) -> Dict[str, Any]:
        """
        Perform comprehensive health check

        Returns:
            Health status information
        """
        health_info = {
            "connected": False,
            "healthy": self._is_healthy,
            "host": self.config.host,
            "port": self.config.port,
            "cluster_mode": self.config.cluster_mode,
            "sentinel_mode": self.config.sentinel_mode
        }

        try:
            if self.client:
                # Test connection with timing
                start_time = asyncio.get_event_loop().time()
                await self._test_connection()
                response_time = (asyncio.get_event_loop().time() - start_time) * 1000

                # Get Redis info
                info = await self.client.info()

                health_info.update({
                    "connected": True,
                    "response_time_ms": round(response_time, 2),
                    "redis_version": info.get('redis_version', 'unknown'),
                    "uptime_in_seconds": info.get('uptime_in_seconds', 0),
                    "connected_clients": info.get('connected_clients', 0),
                    "used_memory": info.get('used_memory', 0),
                    "used_memory_human": info.get('used_memory_human', '0B'),
                    "used_memory_peak": info.get('used_memory_peak', 0),
                    "used_memory_peak_human": info.get('used_memory_peak_human', '0B'),
                    "total_commands_processed": info.get('total_commands_processed', 0),
                    "instantaneous_ops_per_sec": info.get('instantaneous_ops_per_sec', 0),
                    "keyspace_hits": info.get('keyspace_hits', 0),
                    "keyspace_misses": info.get('keyspace_misses', 0)
                })

                # Calculate hit rate
                hits = health_info.get('keyspace_hits', 0)
                misses = health_info.get('keyspace_misses', 0)
                total = hits + misses
                if total > 0:
                    health_info['hit_rate'] = hits / total
                else:
                    health_info['hit_rate'] = 0.0

        except Exception as e:
            health_info.update({
                "error": str(e),
                "healthy": False
            })

        return health_info

    async def flush_db(self, async_flush: bool = False) -> bool:
        """
        Flush current database

        Args:
            async_flush: Whether to flush asynchronously

        Returns:
            True if successful
        """
        try:
            if self.client:
                if async_flush:
                    await self.client.flushdb(asynchronous=True)
                else:
                    await self.client.flushdb()

                logger.info("Redis database flushed", async_flush=async_flush)
                return True
        except Exception as e:
            logger.error("Failed to flush Redis database", error=str(e))
            return False

    async def get_memory_stats(self) -> Dict[str, Any]:
        """
        Get detailed memory statistics

        Returns:
            Memory statistics dictionary
        """
        try:
            if self.client:
                info = await self.client.info('memory')
                return {
                    "used_memory": info.get('used_memory', 0),
                    "used_memory_human": info.get('used_memory_human', '0B'),
                    "used_memory_rss": info.get('used_memory_rss', 0),
                    "used_memory_rss_human": info.get('used_memory_rss_human', '0B'),
                    "used_memory_peak": info.get('used_memory_peak', 0),
                    "used_memory_peak_human": info.get('used_memory_peak_human', '0B'),
                    "used_memory_overhead": info.get('used_memory_overhead', 0),
                    "used_memory_dataset": info.get('used_memory_dataset', 0),
                    "total_system_memory": info.get('total_system_memory', 0),
                    "total_system_memory_human": info.get('total_system_memory_human', '0B'),
                    "maxmemory": info.get('maxmemory', 0),
                    "maxmemory_human": info.get('maxmemory_human', '0B'),
                    "maxmemory_policy": info.get('maxmemory_policy', 'noeviction')
                }
        except Exception as e:
            logger.error("Failed to get memory stats", error=str(e))
            return {"error": str(e)}


# Global connection manager instance
_connection_manager: Optional[RedisConnectionManager] = None


async def initialize_redis(config: Optional[RedisConfig] = None) -> Redis:
    """
    Initialize Redis connection

    Args:
        config: Redis configuration (uses default if None)

    Returns:
        Redis client instance
    """
    global _connection_manager

    if config is None:
        config = RedisConfig()

    if _connection_manager is None:
        _connection_manager = RedisConnectionManager(config)

    return await _connection_manager.connect()


async def get_redis() -> Redis:
    """
    Get Redis client instance

    Returns:
        Redis client instance

    Raises:
        RuntimeError: If Redis is not initialized
    """
    global _connection_manager

    if _connection_manager is None:
        # Auto-initialize with default config
        return await initialize_redis()

    return await _connection_manager.get_client()


async def close_redis() -> None:
    """Close Redis connection"""
    global _connection_manager

    if _connection_manager:
        await _connection_manager.disconnect()
        _connection_manager = None


async def redis_health_check() -> Dict[str, Any]:
    """
    Get Redis health status

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
            "error": "Redis not initialized"
        }


async def flush_redis_db(async_flush: bool = False) -> bool:
    """
    Flush Redis database

    Args:
        async_flush: Whether to flush asynchronously

    Returns:
        True if successful
    """
    global _connection_manager

    if _connection_manager:
        return await _connection_manager.flush_db(async_flush)
    else:
        return False


async def get_redis_memory_stats() -> Dict[str, Any]:
    """
    Get Redis memory statistics

    Returns:
        Memory statistics
    """
    global _connection_manager

    if _connection_manager:
        return await _connection_manager.get_memory_stats()
    else:
        return {"error": "Redis not initialized"}