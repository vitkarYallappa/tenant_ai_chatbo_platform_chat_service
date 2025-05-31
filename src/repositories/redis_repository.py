"""
Redis Base Repository Implementation
===================================

Base repository implementation for Redis operations providing common
functionality for all Redis-based repositories.

Features:
- Connection management and pooling
- Pipeline support for atomic operations
- Lua script execution
- Key pattern management
- TTL management utilities
- Performance monitoring
- Multi-tenant key isolation
- Cluster support
"""

from datetime import datetime, timedelta
from typing import Optional, List, Dict, Any, Union, Set, Callable, Tuple
from redis.asyncio import Redis, ConnectionPool
from redis.exceptions import (
    ConnectionError as RedisConnectionError,
    TimeoutError as RedisTimeoutError,
    ResponseError, DataError, RedisError
)
import json
import pickle
import structlog
import asyncio
from contextlib import asynccontextmanager
from dataclasses import dataclass
from enum import Enum

from .exceptions import (
    RepositoryError, ConnectionError, TimeoutError, ValidationError
)

# Type definitions
TenantId = str
RedisKey = str
RedisValue = Union[str, bytes, int, float]

class RedisDataType(str, Enum):
    """Redis data types"""
    STRING = "string"
    HASH = "hash"
    LIST = "list"
    SET = "set"
    ZSET = "zset"
    STREAM = "stream"

class SerializationMethod(str, Enum):
    """Serialization methods for complex data"""
    JSON = "json"
    PICKLE = "pickle"
    STRING = "string"

@dataclass
class RedisConfig:
    """Redis configuration settings"""
    url: str = "redis://localhost:6379"
    max_connections: int = 50
    retry_on_timeout: bool = True
    health_check_interval: int = 30
    socket_connect_timeout: float = 5.0
    socket_timeout: float = 5.0
    connection_pool_kwargs: Optional[Dict[str, Any]] = None
    decode_responses: bool = True

class RedisMetrics:
    """Redis operation metrics"""
    def __init__(self):
        self.operations = 0
        self.pipeline_operations = 0
        self.lua_script_executions = 0
        self.errors = 0
        self.total_time_ms = 0.0
        self.slow_operations = 0  # Operations > 100ms

    @property
    def avg_operation_time_ms(self) -> float:
        """Calculate average operation time"""
        return self.total_time_ms / self.operations if self.operations > 0 else 0.0

    @property
    def error_rate(self) -> float:
        """Calculate error rate"""
        total = self.operations + self.errors
        return self.errors / total if total > 0 else 0.0

    def to_dict(self) -> Dict[str, Any]:
        """Convert metrics to dictionary"""
        return {
            "operations": self.operations,
            "pipeline_operations": self.pipeline_operations,
            "lua_script_executions": self.lua_script_executions,
            "errors": self.errors,
            "avg_operation_time_ms": self.avg_operation_time_ms,
            "error_rate": self.error_rate,
            "slow_operations": self.slow_operations
        }

class RedisRepository:
    """
    Base repository for Redis operations

    Provides common Redis functionality including:
    - Connection management
    - Key namespace management
    - Serialization utilities
    - Pipeline operations
    - Lua script execution
    - Performance monitoring
    """

    def __init__(
        self,
        redis_client: Redis,
        key_prefix: str = "",
        default_ttl: Optional[int] = None
    ):
        """
        Initialize Redis repository

        Args:
            redis_client: Redis client instance
            key_prefix: Prefix for all keys (for namespacing)
            default_ttl: Default TTL for keys in seconds
        """
        self.redis = redis_client
        self.key_prefix = key_prefix
        self.default_ttl = default_ttl
        self.logger = structlog.get_logger("RedisRepository")
        self._metrics = RedisMetrics()
        self._lua_scripts: Dict[str, str] = {}

    # Key Management

    def build_key(self, tenant_id: TenantId, key_parts: Union[str, List[str]]) -> str:
        """
        Build Redis key with tenant isolation and prefix

        Args:
            tenant_id: Tenant identifier for isolation
            key_parts: Key components (string or list)

        Returns:
            Formatted Redis key
        """
        if isinstance(key_parts, str):
            key_parts = [key_parts]

        parts = []
        if self.key_prefix:
            parts.append(self.key_prefix)

        parts.append(tenant_id)
        parts.extend(str(part) for part in key_parts)

        return ":".join(parts)

    def extract_tenant_from_key(self, key: str) -> Optional[TenantId]:
        """
        Extract tenant ID from Redis key

        Args:
            key: Redis key

        Returns:
            Tenant ID if found, None otherwise
        """
        try:
            parts = key.split(":")
            if self.key_prefix:
                if len(parts) >= 2 and parts[0] == self.key_prefix:
                    return parts[1]
            else:
                if len(parts) >= 1:
                    return parts[0]
            return None
        except Exception:
            return None

    async def scan_keys(
        self,
        pattern: str,
        count: int = 100
    ) -> List[str]:
        """
        Scan for keys matching pattern

        Args:
            pattern: Key pattern (supports wildcards)
            count: Number of keys to return per iteration

        Returns:
            List of matching keys
        """
        try:
            keys = []
            async with self._timed_operation("scan_keys") as timer:
                async for key in self.redis.scan_iter(match=pattern, count=count):
                    keys.append(key if isinstance(key, str) else key.decode())

                self._record_operation(timer.duration_ms)
                return keys

        except Exception as e:
            self._record_error()
            self.logger.error("Failed to scan keys", pattern=pattern, error=str(e))
            raise RepositoryError(f"Failed to scan keys: {e}", original_error=e)

    async def delete_pattern(
        self,
        pattern: str,
        batch_size: int = 100
    ) -> int:
        """
        Delete all keys matching pattern

        Args:
            pattern: Key pattern
            batch_size: Number of keys to delete in each batch

        Returns:
            Number of keys deleted
        """
        try:
            deleted_count = 0

            async with self._timed_operation("delete_pattern") as timer:
                keys_to_delete = []

                async for key in self.redis.scan_iter(match=pattern):
                    keys_to_delete.append(key)

                    if len(keys_to_delete) >= batch_size:
                        if keys_to_delete:
                            deleted = await self.redis.delete(*keys_to_delete)
                            deleted_count += deleted
                            keys_to_delete = []

                # Delete remaining keys
                if keys_to_delete:
                    deleted = await self.redis.delete(*keys_to_delete)
                    deleted_count += deleted

                self._record_operation(timer.duration_ms)

                self.logger.info(
                    "Pattern deletion completed",
                    pattern=pattern,
                    deleted_count=deleted_count
                )

                return deleted_count

        except Exception as e:
            self._record_error()
            self.logger.error("Failed to delete pattern", pattern=pattern, error=str(e))
            raise RepositoryError(f"Failed to delete pattern: {e}", original_error=e)

    # Serialization Utilities

    def serialize_value(
        self,
        value: Any,
        method: SerializationMethod = SerializationMethod.JSON
    ) -> Union[str, bytes]:
        """
        Serialize value for Redis storage

        Args:
            value: Value to serialize
            method: Serialization method

        Returns:
            Serialized value
        """
        try:
            if method == SerializationMethod.JSON:
                return json.dumps(value, default=str)
            elif method == SerializationMethod.PICKLE:
                return pickle.dumps(value)
            else:  # STRING
                return str(value)

        except Exception as e:
            self.logger.warning(
                "Failed to serialize value, using string representation",
                value_type=type(value).__name__,
                method=method,
                error=str(e)
            )
            return str(value)

    def deserialize_value(
        self,
        value: Union[str, bytes],
        method: SerializationMethod = SerializationMethod.JSON
    ) -> Any:
        """
        Deserialize value from Redis

        Args:
            value: Serialized value
            method: Serialization method used

        Returns:
            Deserialized value
        """
        try:
            if value is None:
                return None

            if method == SerializationMethod.JSON:
                if isinstance(value, bytes):
                    value = value.decode('utf-8')
                return json.loads(value)
            elif method == SerializationMethod.PICKLE:
                if isinstance(value, str):
                    value = value.encode('utf-8')
                return pickle.loads(value)
            else:  # STRING
                return str(value)

        except Exception as e:
            self.logger.warning(
                "Failed to deserialize value, returning as string",
                method=method,
                error=str(e)
            )
            return str(value) if value else None

    # Pipeline Operations

    @asynccontextmanager
    async def pipeline(self, transaction: bool = True):
        """
        Context manager for Redis pipeline operations

        Args:
            transaction: Whether to use transaction (MULTI/EXEC)

        Usage:
            async with repository.pipeline() as pipe:
                pipe.set("key1", "value1")
                pipe.set("key2", "value2")
                results = await pipe.execute()
        """
        pipe = None
        start_time = asyncio.get_event_loop().time()

        try:
            pipe = self.redis.pipeline(transaction=transaction)
            yield pipe

        except Exception as e:
            self._record_error()
            self.logger.error("Pipeline operation failed", error=str(e))
            raise RepositoryError(f"Pipeline operation failed: {e}", original_error=e)
        finally:
            if pipe:
                try:
                    await pipe.reset()
                except Exception:
                    pass  # Ignore reset errors

            # Record metrics
            duration_ms = (asyncio.get_event_loop().time() - start_time) * 1000
            self._record_pipeline_operation(duration_ms)

    # Lua Script Support

    def register_lua_script(self, name: str, script: str) -> None:
        """
        Register a Lua script for later execution

        Args:
            name: Script name
            script: Lua script content
        """
        self._lua_scripts[name] = script
        self.logger.debug("Lua script registered", name=name)

    async def execute_lua_script(
        self,
        script_name: str,
        keys: Optional[List[str]] = None,
        args: Optional[List[Any]] = None
    ) -> Any:
        """
        Execute registered Lua script

        Args:
            script_name: Name of registered script
            keys: Redis keys for the script
            args: Script arguments

        Returns:
            Script execution result
        """
        try:
            if script_name not in self._lua_scripts:
                raise ValidationError(f"Lua script '{script_name}' not registered")

            script = self._lua_scripts[script_name]
            keys = keys or []
            args = args or []

            async with self._timed_operation("lua_script") as timer:
                result = await self.redis.eval(script, len(keys), *keys, *args)

                self._record_lua_execution(timer.duration_ms)
                return result

        except Exception as e:
            self._record_error()
            self.logger.error(
                "Lua script execution failed",
                script_name=script_name,
                error=str(e)
            )
            raise RepositoryError(f"Lua script execution failed: {e}", original_error=e)

    # TTL Management

    async def set_with_ttl(
        self,
        key: str,
        value: RedisValue,
        ttl_seconds: Optional[int] = None
    ) -> bool:
        """
        Set key with TTL

        Args:
            key: Redis key
            value: Value to set
            ttl_seconds: TTL in seconds (uses default if None)

        Returns:
            True if successful
        """
        try:
            ttl = ttl_seconds or self.default_ttl

            async with self._timed_operation("set_with_ttl") as timer:
                if ttl:
                    result = await self.redis.setex(key, ttl, value)
                else:
                    result = await self.redis.set(key, value)

                self._record_operation(timer.duration_ms)
                return bool(result)

        except Exception as e:
            self._record_error()
            self.logger.error("Failed to set key with TTL", key=key, error=str(e))
            raise RepositoryError(f"Failed to set key with TTL: {e}", original_error=e)

    async def refresh_ttl(
        self,
        key: str,
        ttl_seconds: Optional[int] = None
    ) -> bool:
        """
        Refresh TTL for existing key

        Args:
            key: Redis key
            ttl_seconds: New TTL in seconds

        Returns:
            True if TTL was set
        """
        try:
            ttl = ttl_seconds or self.default_ttl
            if not ttl:
                return False

            async with self._timed_operation("refresh_ttl") as timer:
                result = await self.redis.expire(key, ttl)

                self._record_operation(timer.duration_ms)
                return bool(result)

        except Exception as e:
            self._record_error()
            self.logger.error("Failed to refresh TTL", key=key, error=str(e))
            raise RepositoryError(f"Failed to refresh TTL: {e}", original_error=e)

    async def get_ttl(self, key: str) -> int:
        """
        Get TTL for key

        Args:
            key: Redis key

        Returns:
            TTL in seconds (-1 if no expiration, -2 if key doesn't exist)
        """
        try:
            async with self._timed_operation("get_ttl") as timer:
                ttl = await self.redis.ttl(key)

                self._record_operation(timer.duration_ms)
                return ttl

        except Exception as e:
            self._record_error()
            self.logger.error("Failed to get TTL", key=key, error=str(e))
            return -2

    # Health and Monitoring

    async def health_check(self) -> Dict[str, Any]:
        """
        Perform health check on Redis connection

        Returns:
            Health status information
        """
        try:
            start_time = asyncio.get_event_loop().time()

            # Test basic operations
            test_key = f"health_check:{datetime.utcnow().timestamp()}"

            # Set and get test
            await self.redis.set(test_key, "health_check", ex=60)
            result = await self.redis.get(test_key)
            await self.redis.delete(test_key)

            duration_ms = (asyncio.get_event_loop().time() - start_time) * 1000

            # Get Redis info
            info = await self.redis.info()

            return {
                "status": "healthy" if result == "health_check" else "degraded",
                "response_time_ms": round(duration_ms, 2),
                "connected_clients": info.get("connected_clients", 0),
                "used_memory": info.get("used_memory", 0),
                "used_memory_human": info.get("used_memory_human", "0B"),
                "redis_version": info.get("redis_version", "unknown"),
                "uptime_in_seconds": info.get("uptime_in_seconds", 0),
                "metrics": self._metrics.to_dict()
            }

        except Exception as e:
            self.logger.error("Health check failed", error=str(e))
            return {
                "status": "unhealthy",
                "error": str(e),
                "metrics": self._metrics.to_dict()
            }

    async def get_memory_usage(self, key: str) -> Optional[int]:
        """
        Get memory usage for a key

        Args:
            key: Redis key

        Returns:
            Memory usage in bytes (None if not available)
        """
        try:
            # This command might not be available in all Redis versions
            usage = await self.redis.memory_usage(key)
            return usage
        except Exception:
            # Command not available or key doesn't exist
            return None

    async def get_key_info(self, key: str) -> Dict[str, Any]:
        """
        Get comprehensive information about a key

        Args:
            key: Redis key

        Returns:
            Key information dictionary
        """
        try:
            info = {}

            # Check if key exists
            exists = await self.redis.exists(key)
            if not exists:
                return {"exists": False}

            info["exists"] = True
            info["type"] = await self.redis.type(key)
            info["ttl"] = await self.redis.ttl(key)

            # Get memory usage if available
            memory_usage = await self.get_memory_usage(key)
            if memory_usage:
                info["memory_bytes"] = memory_usage

            # Type-specific information
            key_type = info["type"]
            if key_type == "string":
                info["length"] = await self.redis.strlen(key)
            elif key_type == "list":
                info["length"] = await self.redis.llen(key)
            elif key_type == "set":
                info["cardinality"] = await self.redis.scard(key)
            elif key_type == "zset":
                info["cardinality"] = await self.redis.zcard(key)
            elif key_type == "hash":
                info["field_count"] = await self.redis.hlen(key)

            return info

        except Exception as e:
            self.logger.error("Failed to get key info", key=key, error=str(e))
            return {"error": str(e)}

    # Metrics and Performance

    def get_metrics(self) -> Dict[str, Any]:
        """Get repository metrics"""
        return self._metrics.to_dict()

    def reset_metrics(self) -> None:
        """Reset all metrics"""
        self._metrics = RedisMetrics()

    # Private Helper Methods

    @asynccontextmanager
    async def _timed_operation(self, operation_name: str):
        """Context manager for timing operations"""
        start_time = asyncio.get_event_loop().time()
        timer = type('Timer', (), {})()

        try:
            self.logger.debug(f"Starting Redis operation: {operation_name}")
            yield timer
        finally:
            duration_ms = (asyncio.get_event_loop().time() - start_time) * 1000
            timer.duration_ms = duration_ms

            self.logger.debug(
                f"Redis operation completed: {operation_name}",
                duration_ms=round(duration_ms, 2)
            )

    def _record_operation(self, duration_ms: float) -> None:
        """Record operation metrics"""
        self._metrics.operations += 1
        self._metrics.total_time_ms += duration_ms

        if duration_ms > 100:  # Slow operation threshold
            self._metrics.slow_operations += 1
            self.logger.warning(
                "Slow Redis operation detected",
                duration_ms=duration_ms
            )

    def _record_pipeline_operation(self, duration_ms: float) -> None:
        """Record pipeline operation metrics"""
        self._metrics.pipeline_operations += 1
        self._record_operation(duration_ms)

    def _record_lua_execution(self, duration_ms: float) -> None:
        """Record Lua script execution metrics"""
        self._metrics.lua_script_executions += 1
        self._record_operation(duration_ms)

    def _record_error(self) -> None:
        """Record error metrics"""
        self._metrics.errors += 1


# Connection Management
class RedisConnectionManager:
    """Redis connection manager with connection pooling"""

    _clients: Dict[str, Redis] = {}

    @classmethod
    async def get_client(cls, config: RedisConfig) -> Redis:
        """
        Get Redis client instance

        Args:
            config: Redis configuration

        Returns:
            Redis client instance
        """
        config_key = f"{config.url}:{config.max_connections}"

        if config_key not in cls._clients:
            try:
                # Create connection pool
                pool_kwargs = config.connection_pool_kwargs or {}
                pool = ConnectionPool.from_url(
                    config.url,
                    max_connections=config.max_connections,
                    retry_on_timeout=config.retry_on_timeout,
                    health_check_interval=config.health_check_interval,
                    socket_connect_timeout=config.socket_connect_timeout,
                    socket_timeout=config.socket_timeout,
                    decode_responses=config.decode_responses,
                    **pool_kwargs
                )

                # Create Redis client
                client = Redis(connection_pool=pool)

                # Test connection
                await client.ping()

                cls._clients[config_key] = client

            except Exception as e:
                raise ConnectionError("Redis", original_error=e)

        return cls._clients[config_key]

    @classmethod
    async def close_all_connections(cls):
        """Close all Redis connections"""
        for client in cls._clients.values():
            await client.close()
        cls._clients.clear()