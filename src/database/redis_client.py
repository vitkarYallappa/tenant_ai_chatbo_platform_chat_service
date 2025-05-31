# src/database/redis_client.py
"""
Redis connection manager and client setup.
Provides Redis client with connection pooling, cluster support, and health monitoring.
"""

import redis.asyncio as redis
from redis.asyncio import Redis, ConnectionPool
from redis.exceptions import ConnectionError as RedisConnectionError, TimeoutError, ResponseError
from typing import Optional, Dict, Any, List, Union
import structlog
import asyncio
import json
from datetime import datetime, timedelta

# Note: This would normally import from settings, but since it's not yet available:
# from src.config.settings import get_settings

logger = structlog.get_logger()


class RedisManager:
    """
    Redis connection manager with connection pooling and health monitoring.
    Supports both standalone Redis and Redis Cluster configurations.
    """

    def __init__(self):
        self.client: Optional[Redis] = None
        self.connection_pool: Optional[ConnectionPool] = None
        # self.settings = get_settings()  # Will be available after config setup

        # Default settings (will be replaced by actual config)
        self.redis_url = "redis://localhost:6379"
        self.max_connections = 50
        self.socket_timeout = 10
        self.socket_connect_timeout = 10
        self.retry_on_timeout = True
        self.decode_responses = True
        self.encoding = "utf-8"
        self.db = 0

    async def connect(self) -> None:
        """
        Establish Redis connection with connection pooling.

        Raises:
            ConnectionError: If connection fails
        """
        try:
            # Create connection pool for better performance
            self.connection_pool = redis.ConnectionPool.from_url(
                self.redis_url,
                max_connections=self.max_connections,
                socket_timeout=self.socket_timeout,
                socket_connect_timeout=self.socket_connect_timeout,
                retry_on_timeout=self.retry_on_timeout,
                decode_responses=self.decode_responses,
                encoding=self.encoding,
                db=self.db
            )

            self.client = redis.Redis(connection_pool=self.connection_pool)

            # Test connection
            await self.client.ping()

            logger.info(
                "Redis connected successfully",
                url=self._safe_url(),
                max_connections=self.max_connections,
                db=self.db
            )

        except RedisConnectionError as e:
            logger.error("Redis connection error", error=str(e))
            raise ConnectionError(f"Redis connection failed: {e}")
        except TimeoutError as e:
            logger.error("Redis connection timeout", error=str(e))
            raise ConnectionError(f"Redis connection timeout: {e}")
        except Exception as e:
            logger.error("Unexpected Redis connection error", error=str(e))
            raise ConnectionError(f"Redis connection error: {e}")

    async def disconnect(self) -> None:
        """Close Redis connection gracefully."""
        if self.client:
            await self.client.close()
            logger.info("Redis connection closed")

        if self.connection_pool:
            await self.connection_pool.disconnect()
            logger.info("Redis connection pool closed")

    def get_client(self) -> Redis:
        """
        Get Redis client instance.

        Returns:
            Redis client instance

        Raises:
            RuntimeError: If Redis not connected
        """
        if not self.client:
            raise RuntimeError("Redis not connected. Call connect() first.")
        return self.client

    async def health_check(self) -> bool:
        """
        Check Redis connection health.

        Returns:
            True if healthy, False otherwise
        """
        try:
            if self.client:
                # Use a short timeout for health checks
                await asyncio.wait_for(self.client.ping(), timeout=1.0)
                return True
            return False
        except Exception as e:
            logger.error("Redis health check failed", error=str(e))
            return False

    async def get_server_info(self) -> Dict[str, Any]:
        """
        Get Redis server information.

        Returns:
            Dictionary with server information
        """
        try:
            if not self.client:
                return {"status": "disconnected"}

            info = await self.client.info()

            return {
                "status": "connected",
                "redis_version": info.get("redis_version"),
                "uptime_in_seconds": info.get("uptime_in_seconds"),
                "connected_clients": info.get("connected_clients"),
                "used_memory": info.get("used_memory"),
                "used_memory_human": info.get("used_memory_human"),
                "total_commands_processed": info.get("total_commands_processed"),
                "keyspace_hits": info.get("keyspace_hits"),
                "keyspace_misses": info.get("keyspace_misses")
            }
        except Exception as e:
            logger.error("Failed to get Redis server info", error=str(e))
            return {"status": "error", "error": str(e)}

    # String operations
    async def set_with_ttl(
            self,
            key: str,
            value: str,
            ttl_seconds: int
    ) -> bool:
        """
        Set key with TTL.

        Args:
            key: Redis key
            value: Value to set
            ttl_seconds: TTL in seconds

        Returns:
            True if successful, False otherwise
        """
        try:
            client = self.get_client()
            result = await client.setex(key, ttl_seconds, value)
            return bool(result)
        except Exception as e:
            logger.error("Redis set operation failed", key=key, error=str(e))
            return False

    async def get_string(self, key: str) -> Optional[str]:
        """
        Get string value from Redis.

        Args:
            key: Redis key

        Returns:
            String value or None if not found
        """
        try:
            client = self.get_client()
            return await client.get(key)
        except Exception as e:
            logger.error("Redis get operation failed", key=key, error=str(e))
            return None

    # Hash operations
    async def get_hash(self, key: str) -> Dict[str, str]:
        """
        Get hash from Redis.

        Args:
            key: Redis key

        Returns:
            Dictionary with hash fields and values
        """
        try:
            client = self.get_client()
            return await client.hgetall(key)
        except Exception as e:
            logger.error("Redis hash get failed", key=key, error=str(e))
            return {}

    async def set_hash(
            self,
            key: str,
            mapping: Dict[str, str],
            ttl_seconds: Optional[int] = None
    ) -> bool:
        """
        Set hash in Redis with optional TTL.

        Args:
            key: Redis key
            mapping: Dictionary with field-value pairs
            ttl_seconds: Optional TTL in seconds

        Returns:
            True if successful, False otherwise
        """
        try:
            client = self.get_client()
            pipe = client.pipeline()
            pipe.hset(key, mapping=mapping)
            if ttl_seconds:
                pipe.expire(key, ttl_seconds)
            await pipe.execute()
            return True
        except Exception as e:
            logger.error("Redis hash set failed", key=key, error=str(e))
            return False

    async def hash_field_exists(self, key: str, field: str) -> bool:
        """Check if hash field exists."""
        try:
            client = self.get_client()
            return await client.hexists(key, field)
        except Exception as e:
            logger.error("Redis hash field check failed", key=key, field=field, error=str(e))
            return False

    async def delete_hash_field(self, key: str, field: str) -> bool:
        """Delete hash field."""
        try:
            client = self.get_client()
            result = await client.hdel(key, field)
            return result > 0
        except Exception as e:
            logger.error("Redis hash field delete failed", key=key, field=field, error=str(e))
            return False

    # Set operations
    async def add_to_set(self, key: str, *values: str) -> int:
        """Add values to set."""
        try:
            client = self.get_client()
            return await client.sadd(key, *values)
        except Exception as e:
            logger.error("Redis set add failed", key=key, error=str(e))
            return 0

    async def remove_from_set(self, key: str, *values: str) -> int:
        """Remove values from set."""
        try:
            client = self.get_client()
            return await client.srem(key, *values)
        except Exception as e:
            logger.error("Redis set remove failed", key=key, error=str(e))
            return 0

    async def is_member_of_set(self, key: str, value: str) -> bool:
        """Check if value is member of set."""
        try:
            client = self.get_client()
            return await client.sismember(key, value)
        except Exception as e:
            logger.error("Redis set membership check failed", key=key, value=value, error=str(e))
            return False

    # Sorted set operations
    async def add_to_sorted_set(
            self,
            key: str,
            mapping: Dict[str, float],
            ttl_seconds: Optional[int] = None
    ) -> int:
        """
        Add items to sorted set.

        Args:
            key: Redis key
            mapping: Dictionary with member-score pairs
            ttl_seconds: Optional TTL in seconds

        Returns:
            Number of items added
        """
        try:
            client = self.get_client()
            pipe = client.pipeline()
            result = pipe.zadd(key, mapping)
            if ttl_seconds:
                pipe.expire(key, ttl_seconds)
            results = await pipe.execute()
            return results[0]
        except Exception as e:
            logger.error("Redis sorted set add failed", key=key, error=str(e))
            return 0

    async def remove_from_sorted_set_by_score(
            self,
            key: str,
            min_score: float,
            max_score: float
    ) -> int:
        """Remove items from sorted set by score range."""
        try:
            client = self.get_client()
            return await client.zremrangebyscore(key, min_score, max_score)
        except Exception as e:
            logger.error("Redis sorted set remove by score failed", key=key, error=str(e))
            return 0

    async def count_in_sorted_set_by_score(
            self,
            key: str,
            min_score: float,
            max_score: float
    ) -> int:
        """Count items in sorted set within score range."""
        try:
            client = self.get_client()
            return await client.zcount(key, min_score, max_score)
        except Exception as e:
            logger.error("Redis sorted set count failed", key=key, error=str(e))
            return 0

    # List operations
    async def push_to_list(self, key: str, *values: str) -> int:
        """Push values to list (left push)."""
        try:
            client = self.get_client()
            return await client.lpush(key, *values)
        except Exception as e:
            logger.error("Redis list push failed", key=key, error=str(e))
            return 0

    async def pop_from_list(self, key: str) -> Optional[str]:
        """Pop value from list (right pop)."""
        try:
            client = self.get_client()
            return await client.rpop(key)
        except Exception as e:
            logger.error("Redis list pop failed", key=key, error=str(e))
            return None

    async def get_list_range(self, key: str, start: int = 0, end: int = -1) -> List[str]:
        """Get range of values from list."""
        try:
            client = self.get_client()
            return await client.lrange(key, start, end)
        except Exception as e:
            logger.error("Redis list range failed", key=key, error=str(e))
            return []

    # Key management
    async def delete_key(self, *keys: str) -> int:
        """Delete one or more keys."""
        try:
            client = self.get_client()
            return await client.delete(*keys)
        except Exception as e:
            logger.error("Redis key delete failed", keys=keys, error=str(e))
            return 0

    async def key_exists(self, key: str) -> bool:
        """Check if key exists."""
        try:
            client = self.get_client()
            return await client.exists(key) > 0
        except Exception as e:
            logger.error("Redis key existence check failed", key=key, error=str(e))
            return False

    async def set_key_ttl(self, key: str, ttl_seconds: int) -> bool:
        """Set TTL for existing key."""
        try:
            client = self.get_client()
            return await client.expire(key, ttl_seconds)
        except Exception as e:
            logger.error("Redis TTL set failed", key=key, error=str(e))
            return False

    async def get_key_ttl(self, key: str) -> int:
        """Get TTL for key (-1 if no TTL, -2 if key doesn't exist)."""
        try:
            client = self.get_client()
            return await client.ttl(key)
        except Exception as e:
            logger.error("Redis TTL get failed", key=key, error=str(e))
            return -2

    # Pattern operations
    async def find_keys(self, pattern: str) -> List[str]:
        """Find keys matching pattern."""
        try:
            client = self.get_client()
            return await client.keys(pattern)
        except Exception as e:
            logger.error("Redis key pattern search failed", pattern=pattern, error=str(e))
            return []

    async def delete_by_pattern(self, pattern: str) -> int:
        """Delete keys matching pattern."""
        try:
            keys = await self.find_keys(pattern)
            if keys:
                return await self.delete_key(*keys)
            return 0
        except Exception as e:
            logger.error("Redis pattern delete failed", pattern=pattern, error=str(e))
            return 0

    # Utility methods
    async def serialize_and_set(
            self,
            key: str,
            data: Dict[str, Any],
            ttl_seconds: Optional[int] = None
    ) -> bool:
        """Serialize dictionary to JSON and store in Redis."""
        try:
            json_data = json.dumps(data)
            if ttl_seconds:
                return await self.set_with_ttl(key, json_data, ttl_seconds)
            else:
                client = self.get_client()
                result = await client.set(key, json_data)
                return bool(result)
        except Exception as e:
            logger.error("Redis serialize and set failed", key=key, error=str(e))
            return False

    async def get_and_deserialize(self, key: str) -> Optional[Dict[str, Any]]:
        """Get value from Redis and deserialize from JSON."""
        try:
            json_data = await self.get_string(key)
            if json_data:
                return json.loads(json_data)
            return None
        except Exception as e:
            logger.error("Redis get and deserialize failed", key=key, error=str(e))
            return None

    def _safe_url(self) -> str:
        """Get Redis URL with password masked for logging."""
        url = self.redis_url
        if '@' in url:
            # Mask password in URL
            parts = url.split('@')
            if len(parts) == 2:
                auth_part = parts[0]
                if '://' in auth_part:
                    scheme_and_auth = auth_part.split('://')
                    if len(scheme_and_auth) == 2 and ':' in scheme_and_auth[1]:
                        user_pass = scheme_and_auth[1].split(':')
                        if len(user_pass) == 2:
                            masked_url = f"{scheme_and_auth[0]}://{user_pass[0]}:***@{parts[1]}"
                            return masked_url
        return url

    async def cleanup_expired_keys(self, patterns: List[str]) -> Dict[str, int]:
        """
        Clean up expired keys matching patterns.

        Args:
            patterns: List of key patterns to check

        Returns:
            Dictionary with cleanup counts per pattern
        """
        cleanup_results = {}

        try:
            for pattern in patterns:
                keys = await self.find_keys(pattern)
                deleted_count = 0

                for key in keys:
                    ttl = await self.get_key_ttl(key)
                    if ttl == -2:  # Key doesn't exist (expired)
                        deleted_count += 1
                    elif ttl == 0:  # Key expired but not cleaned up
                        await self.delete_key(key)
                        deleted_count += 1

                cleanup_results[pattern] = deleted_count

            if sum(cleanup_results.values()) > 0:
                logger.info("Redis cleanup completed", **cleanup_results)

        except Exception as e:
            logger.error("Redis cleanup failed", error=str(e))

        return cleanup_results


# Global Redis manager instance
redis_manager = RedisManager()


async def get_redis() -> Redis:
    """
    Dependency function to get Redis client instance.

    Returns:
        Redis client instance
    """
    return redis_manager.get_client()


# Convenience functions for common operations
async def get_session_cache(tenant_id: str, session_id: str) -> Dict[str, str]:
    """Get session cache data."""
    key = f"session:{tenant_id}:{session_id}"
    return await redis_manager.get_hash(key)


async def set_session_cache(
        tenant_id: str,
        session_id: str,
        data: Dict[str, str],
        ttl_seconds: int = 3600
) -> bool:
    """Set session cache data."""
    key = f"session:{tenant_id}:{session_id}"
    return await redis_manager.set_hash(key, data, ttl_seconds)


async def get_rate_limit_count(tenant_id: str, identifier: str, window: str) -> int:
    """Get current rate limit count."""
    key = f"rate_limit:{tenant_id}:{identifier}:{window}"
    count_str = await redis_manager.get_string(key)
    return int(count_str) if count_str else 0


async def increment_rate_limit(
        tenant_id: str,
        identifier: str,
        window: str,
        ttl_seconds: int
) -> int:
    """Increment rate limit counter."""
    try:
        client = redis_manager.get_client()
        key = f"rate_limit:{tenant_id}:{identifier}:{window}"

        pipe = client.pipeline()
        pipe.incr(key)
        pipe.expire(key, ttl_seconds)
        results = await pipe.execute()

        return results[0]
    except Exception as e:
        logger.error("Rate limit increment failed", error=str(e))
        return 0