"""
Cache Repository Implementation
==============================

Redis repository for general caching operations supporting multiple
cache patterns, TTL management, and cache analytics.

Features:
- Multiple cache patterns (simple, hash, list, set)
- TTL management with automatic expiration
- Cache invalidation strategies
- Multi-tenant cache isolation
- Cache hit/miss analytics
- Bulk cache operations
- Cache warming and preloading
"""

from datetime import datetime, timedelta
from typing import Optional, List, Dict, Any, Union, Set, Tuple
from redis.asyncio import Redis
import json
import pickle
import hashlib
import structlog
from dataclasses import dataclass
from enum import Enum
import asyncio

from .exceptions import (
    RepositoryError, ValidationError, TimeoutError
)

# Type definitions
TenantId = str
CacheKey = str
CacheValue = Union[str, int, float, bool, Dict[str, Any], List[Any]]


class CacheStrategy(str, Enum):
    """Cache invalidation strategies"""
    TTL = "ttl"  # Time-based expiration
    LRU = "lru"  # Least Recently Used
    LFU = "lfu"  # Least Frequently Used
    WRITE_THROUGH = "write_through"  # Update cache on write
    WRITE_BEHIND = "write_behind"  # Async cache update
    READ_THROUGH = "read_through"  # Load on cache miss


class CachePattern(str, Enum):
    """Cache storage patterns"""
    SIMPLE = "simple"  # Single key-value
    HASH = "hash"  # Hash table
    LIST = "list"  # Ordered list
    SET = "set"  # Unordered set
    SORTED_SET = "zset"  # Sorted set with scores


@dataclass
class CacheConfig:
    """Cache configuration"""
    ttl_seconds: int = 3600  # 1 hour default
    strategy: CacheStrategy = CacheStrategy.TTL
    pattern: CachePattern = CachePattern.SIMPLE
    max_size: Optional[int] = None  # For LRU/LFU
    compression: bool = False  # Compress large values
    serialization: str = "json"  # json, pickle, or string


class CacheStats:
    """Cache statistics tracking"""

    def __init__(self):
        self.hits = 0
        self.misses = 0
        self.sets = 0
        self.deletes = 0
        self.expired = 0
        self.total_size_bytes = 0

    @property
    def hit_rate(self) -> float:
        """Calculate cache hit rate"""
        total = self.hits + self.misses
        return (self.hits / total) if total > 0 else 0.0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "hits": self.hits,
            "misses": self.misses,
            "sets": self.sets,
            "deletes": self.deletes,
            "expired": self.expired,
            "hit_rate": self.hit_rate,
            "total_size_bytes": self.total_size_bytes
        }


class CacheRepository:
    """
    Repository for general caching operations in Redis

    Provides comprehensive caching functionality with:
    - Multiple storage patterns
    - Flexible TTL management
    - Multi-tenant isolation
    - Performance analytics
    - Bulk operations
    """

    def __init__(self, redis_client: Redis):
        """
        Initialize cache repository

        Args:
            redis_client: Redis async client instance
        """
        self.redis = redis_client
        self.logger = structlog.get_logger("CacheRepository")
        self._stats: Dict[TenantId, CacheStats] = {}

    # Simple Cache Operations

    async def get(
            self,
            tenant_id: TenantId,
            key: CacheKey,
            config: Optional[CacheConfig] = None
    ) -> Optional[CacheValue]:
        """
        Get value from cache

        Args:
            tenant_id: Tenant identifier
            key: Cache key
            config: Cache configuration

        Returns:
            Cached value if found, None otherwise
        """
        try:
            config = config or CacheConfig()
            cache_key = self._build_cache_key(tenant_id, key)

            # Get value from Redis
            raw_value = await self.redis.get(cache_key)

            if raw_value is None:
                self._record_miss(tenant_id)
                self.logger.debug("Cache miss", tenant_id=tenant_id, key=key)
                return None

            # Deserialize value
            value = self._deserialize(raw_value, config.serialization)

            self._record_hit(tenant_id)
            self.logger.debug("Cache hit", tenant_id=tenant_id, key=key)

            return value

        except Exception as e:
            self.logger.error(
                "Failed to get cache value",
                tenant_id=tenant_id,
                key=key,
                error=str(e)
            )
            self._record_miss(tenant_id)
            return None

    async def set(
            self,
            tenant_id: TenantId,
            key: CacheKey,
            value: CacheValue,
            config: Optional[CacheConfig] = None
    ) -> bool:
        """
        Set value in cache

        Args:
            tenant_id: Tenant identifier
            key: Cache key
            value: Value to cache
            config: Cache configuration

        Returns:
            True if set successfully
        """
        try:
            config = config or CacheConfig()
            cache_key = self._build_cache_key(tenant_id, key)

            # Serialize value
            serialized_value = self._serialize(value, config.serialization, config.compression)

            # Set value with TTL
            if config.ttl_seconds > 0:
                success = await self.redis.setex(cache_key, config.ttl_seconds, serialized_value)
            else:
                success = await self.redis.set(cache_key, serialized_value)

            if success:
                self._record_set(tenant_id,
                                 len(serialized_value.encode()) if isinstance(serialized_value, str) else len(
                                     serialized_value))
                self.logger.debug(
                    "Cache set successful",
                    tenant_id=tenant_id,
                    key=key,
                    ttl=config.ttl_seconds
                )

            return bool(success)

        except Exception as e:
            self.logger.error(
                "Failed to set cache value",
                tenant_id=tenant_id,
                key=key,
                error=str(e)
            )
            raise RepositoryError(f"Failed to set cache value: {e}", original_error=e)

    async def delete(
            self,
            tenant_id: TenantId,
            key: CacheKey
    ) -> bool:
        """
        Delete value from cache

        Args:
            tenant_id: Tenant identifier
            key: Cache key

        Returns:
            True if deleted, False if not found
        """
        try:
            cache_key = self._build_cache_key(tenant_id, key)
            deleted = await self.redis.delete(cache_key)

            if deleted:
                self._record_delete(tenant_id)
                self.logger.debug("Cache delete successful", tenant_id=tenant_id, key=key)

            return deleted > 0

        except Exception as e:
            self.logger.error(
                "Failed to delete cache value",
                tenant_id=tenant_id,
                key=key,
                error=str(e)
            )
            raise RepositoryError(f"Failed to delete cache value: {e}", original_error=e)

    async def exists(
            self,
            tenant_id: TenantId,
            key: CacheKey
    ) -> bool:
        """
        Check if key exists in cache

        Args:
            tenant_id: Tenant identifier
            key: Cache key

        Returns:
            True if key exists
        """
        try:
            cache_key = self._build_cache_key(tenant_id, key)
            exists = await self.redis.exists(cache_key)
            return exists > 0

        except Exception as e:
            self.logger.error(
                "Failed to check cache key existence",
                tenant_id=tenant_id,
                key=key,
                error=str(e)
            )
            return False

    async def expire(
            self,
            tenant_id: TenantId,
            key: CacheKey,
            seconds: int
    ) -> bool:
        """
        Set expiration for cache key

        Args:
            tenant_id: Tenant identifier
            key: Cache key
            seconds: Expiration time in seconds

        Returns:
            True if expiration was set
        """
        try:
            cache_key = self._build_cache_key(tenant_id, key)
            success = await self.redis.expire(cache_key, seconds)

            if success:
                self.logger.debug(
                    "Cache expiration set",
                    tenant_id=tenant_id,
                    key=key,
                    seconds=seconds
                )

            return bool(success)

        except Exception as e:
            self.logger.error(
                "Failed to set cache expiration",
                tenant_id=tenant_id,
                key=key,
                error=str(e)
            )
            raise RepositoryError(f"Failed to set cache expiration: {e}", original_error=e)

    async def ttl(
            self,
            tenant_id: TenantId,
            key: CacheKey
    ) -> int:
        """
        Get time-to-live for cache key

        Args:
            tenant_id: Tenant identifier
            key: Cache key

        Returns:
            TTL in seconds (-1 if no expiration, -2 if key doesn't exist)
        """
        try:
            cache_key = self._build_cache_key(tenant_id, key)
            return await self.redis.ttl(cache_key)

        except Exception as e:
            self.logger.error(
                "Failed to get cache TTL",
                tenant_id=tenant_id,
                key=key,
                error=str(e)
            )
            return -2

    # Hash Cache Operations

    async def hget(
            self,
            tenant_id: TenantId,
            key: CacheKey,
            field: str,
            config: Optional[CacheConfig] = None
    ) -> Optional[CacheValue]:
        """Get field from hash cache"""
        try:
            config = config or CacheConfig()
            cache_key = self._build_cache_key(tenant_id, key)

            raw_value = await self.redis.hget(cache_key, field)

            if raw_value is None:
                self._record_miss(tenant_id)
                return None

            value = self._deserialize(raw_value, config.serialization)
            self._record_hit(tenant_id)

            return value

        except Exception as e:
            self.logger.error(
                "Failed to get hash field",
                tenant_id=tenant_id,
                key=key,
                field=field,
                error=str(e)
            )
            self._record_miss(tenant_id)
            return None

    async def hset(
            self,
            tenant_id: TenantId,
            key: CacheKey,
            field: str,
            value: CacheValue,
            config: Optional[CacheConfig] = None
    ) -> bool:
        """Set field in hash cache"""
        try:
            config = config or CacheConfig()
            cache_key = self._build_cache_key(tenant_id, key)

            serialized_value = self._serialize(value, config.serialization, config.compression)

            success = await self.redis.hset(cache_key, field, serialized_value)

            if config.ttl_seconds > 0:
                await self.redis.expire(cache_key, config.ttl_seconds)

            if success:
                self._record_set(tenant_id,
                                 len(serialized_value.encode()) if isinstance(serialized_value, str) else len(
                                     serialized_value))

            return bool(success)

        except Exception as e:
            self.logger.error(
                "Failed to set hash field",
                tenant_id=tenant_id,
                key=key,
                field=field,
                error=str(e)
            )
            raise RepositoryError(f"Failed to set hash field: {e}", original_error=e)

    async def hgetall(
            self,
            tenant_id: TenantId,
            key: CacheKey,
            config: Optional[CacheConfig] = None
    ) -> Dict[str, CacheValue]:
        """Get all fields from hash cache"""
        try:
            config = config or CacheConfig()
            cache_key = self._build_cache_key(tenant_id, key)

            raw_hash = await self.redis.hgetall(cache_key)

            if not raw_hash:
                self._record_miss(tenant_id)
                return {}

            # Deserialize all values
            result = {}
            for field, raw_value in raw_hash.items():
                result[field] = self._deserialize(raw_value, config.serialization)

            self._record_hit(tenant_id)
            return result

        except Exception as e:
            self.logger.error(
                "Failed to get hash",
                tenant_id=tenant_id,
                key=key,
                error=str(e)
            )
            self._record_miss(tenant_id)
            return {}

    async def hmset(
            self,
            tenant_id: TenantId,
            key: CacheKey,
            mapping: Dict[str, CacheValue],
            config: Optional[CacheConfig] = None
    ) -> bool:
        """Set multiple fields in hash cache"""
        try:
            config = config or CacheConfig()
            cache_key = self._build_cache_key(tenant_id, key)

            # Serialize all values
            serialized_mapping = {}
            total_size = 0
            for field, value in mapping.items():
                serialized_value = self._serialize(value, config.serialization, config.compression)
                serialized_mapping[field] = serialized_value
                total_size += len(serialized_value.encode()) if isinstance(serialized_value, str) else len(
                    serialized_value)

            success = await self.redis.hset(cache_key, mapping=serialized_mapping)

            if config.ttl_seconds > 0:
                await self.redis.expire(cache_key, config.ttl_seconds)

            if success:
                self._record_set(tenant_id, total_size)

            return bool(success)

        except Exception as e:
            self.logger.error(
                "Failed to set hash fields",
                tenant_id=tenant_id,
                key=key,
                error=str(e)
            )
            raise RepositoryError(f"Failed to set hash fields: {e}", original_error=e)

    # List Cache Operations

    async def lpush(
            self,
            tenant_id: TenantId,
            key: CacheKey,
            value: CacheValue,
            config: Optional[CacheConfig] = None
    ) -> int:
        """Push value to left of list"""
        try:
            config = config or CacheConfig()
            cache_key = self._build_cache_key(tenant_id, key)

            serialized_value = self._serialize(value, config.serialization, config.compression)
            length = await self.redis.lpush(cache_key, serialized_value)

            if config.ttl_seconds > 0:
                await self.redis.expire(cache_key, config.ttl_seconds)

            self._record_set(tenant_id, len(serialized_value.encode()) if isinstance(serialized_value, str) else len(
                serialized_value))

            return length

        except Exception as e:
            self.logger.error(
                "Failed to push to list",
                tenant_id=tenant_id,
                key=key,
                error=str(e)
            )
            raise RepositoryError(f"Failed to push to list: {e}", original_error=e)

    async def lrange(
            self,
            tenant_id: TenantId,
            key: CacheKey,
            start: int = 0,
            end: int = -1,
            config: Optional[CacheConfig] = None
    ) -> List[CacheValue]:
        """Get range from list"""
        try:
            config = config or CacheConfig()
            cache_key = self._build_cache_key(tenant_id, key)

            raw_values = await self.redis.lrange(cache_key, start, end)

            if not raw_values:
                self._record_miss(tenant_id)
                return []

            # Deserialize all values
            values = [
                self._deserialize(raw_value, config.serialization)
                for raw_value in raw_values
            ]

            self._record_hit(tenant_id)
            return values

        except Exception as e:
            self.logger.error(
                "Failed to get list range",
                tenant_id=tenant_id,
                key=key,
                error=str(e)
            )
            self._record_miss(tenant_id)
            return []

    # Bulk Operations

    async def mget(
            self,
            tenant_id: TenantId,
            keys: List[CacheKey],
            config: Optional[CacheConfig] = None
    ) -> Dict[CacheKey, Optional[CacheValue]]:
        """Get multiple values from cache"""
        try:
            config = config or CacheConfig()
            cache_keys = [self._build_cache_key(tenant_id, key) for key in keys]

            raw_values = await self.redis.mget(cache_keys)

            result = {}
            hits = 0
            misses = 0

            for i, (key, raw_value) in enumerate(zip(keys, raw_values)):
                if raw_value is not None:
                    result[key] = self._deserialize(raw_value, config.serialization)
                    hits += 1
                else:
                    result[key] = None
                    misses += 1

            # Record stats
            for _ in range(hits):
                self._record_hit(tenant_id)
            for _ in range(misses):
                self._record_miss(tenant_id)

            self.logger.debug(
                "Bulk get completed",
                tenant_id=tenant_id,
                total_keys=len(keys),
                hits=hits,
                misses=misses
            )

            return result

        except Exception as e:
            self.logger.error(
                "Failed to get multiple cache values",
                tenant_id=tenant_id,
                keys=keys,
                error=str(e)
            )
            return {key: None for key in keys}

    async def mset(
            self,
            tenant_id: TenantId,
            mapping: Dict[CacheKey, CacheValue],
            config: Optional[CacheConfig] = None
    ) -> bool:
        """Set multiple values in cache"""
        try:
            config = config or CacheConfig()

            # Serialize all values
            serialized_mapping = {}
            total_size = 0
            for key, value in mapping.items():
                cache_key = self._build_cache_key(tenant_id, key)
                serialized_value = self._serialize(value, config.serialization, config.compression)
                serialized_mapping[cache_key] = serialized_value
                total_size += len(serialized_value.encode()) if isinstance(serialized_value, str) else len(
                    serialized_value)

            # Use pipeline for atomic operation
            pipe = self.redis.pipeline()

            # Set all values
            pipe.mset(serialized_mapping)

            # Set TTL for each key if specified
            if config.ttl_seconds > 0:
                for cache_key in serialized_mapping.keys():
                    pipe.expire(cache_key, config.ttl_seconds)

            results = await pipe.execute()
            success = results[0]  # mset result

            if success:
                self._record_set(tenant_id, total_size)
                self.logger.debug(
                    "Bulk set completed",
                    tenant_id=tenant_id,
                    total_keys=len(mapping),
                    total_size_bytes=total_size
                )

            return bool(success)

        except Exception as e:
            self.logger.error(
                "Failed to set multiple cache values",
                tenant_id=tenant_id,
                keys=list(mapping.keys()),
                error=str(e)
            )
            raise RepositoryError(f"Failed to set multiple cache values: {e}", original_error=e)

    # Cache Management

    async def invalidate_pattern(
            self,
            tenant_id: TenantId,
            pattern: str
    ) -> int:
        """
        Invalidate cache keys matching pattern

        Args:
            tenant_id: Tenant identifier
            pattern: Key pattern (supports wildcards)

        Returns:
            Number of keys deleted
        """
        try:
            cache_pattern = self._build_cache_key(tenant_id, pattern)

            deleted_count = 0
            async for key in self.redis.scan_iter(match=cache_pattern):
                await self.redis.delete(key)
                deleted_count += 1

            if deleted_count > 0:
                for _ in range(deleted_count):
                    self._record_delete(tenant_id)

                self.logger.info(
                    "Cache invalidation completed",
                    tenant_id=tenant_id,
                    pattern=pattern,
                    deleted_count=deleted_count
                )

            return deleted_count

        except Exception as e:
            self.logger.error(
                "Failed to invalidate cache pattern",
                tenant_id=tenant_id,
                pattern=pattern,
                error=str(e)
            )
            raise RepositoryError(f"Failed to invalidate cache pattern: {e}", original_error=e)

    async def clear_tenant_cache(self, tenant_id: TenantId) -> int:
        """Clear all cache for a tenant"""
        return await self.invalidate_pattern(tenant_id, "*")

    async def get_cache_size(self, tenant_id: TenantId) -> Dict[str, Any]:
        """Get cache size statistics for tenant"""
        try:
            pattern = self._build_cache_key(tenant_id, "*")

            total_keys = 0
            total_memory = 0
            key_types = {}

            async for key in self.redis.scan_iter(match=pattern):
                total_keys += 1

                # Get key type
                key_type = await self.redis.type(key)
                key_types[key_type] = key_types.get(key_type, 0) + 1

                # Get memory usage (approximate)
                try:
                    memory = await self.redis.memory_usage(key)
                    if memory:
                        total_memory += memory
                except:
                    # memory_usage command might not be available
                    pass

            return {
                "total_keys": total_keys,
                "total_memory_bytes": total_memory,
                "key_types": key_types,
                "tenant_id": tenant_id
            }

        except Exception as e:
            self.logger.error(
                "Failed to get cache size",
                tenant_id=tenant_id,
                error=str(e)
            )
            return {"error": str(e)}

    # Statistics and Analytics

    def get_stats(self, tenant_id: TenantId) -> Dict[str, Any]:
        """Get cache statistics for tenant"""
        if tenant_id not in self._stats:
            self._stats[tenant_id] = CacheStats()

        return self._stats[tenant_id].to_dict()

    def reset_stats(self, tenant_id: TenantId) -> None:
        """Reset cache statistics for tenant"""
        self._stats[tenant_id] = CacheStats()

    # Private Helper Methods

    def _build_cache_key(self, tenant_id: TenantId, key: CacheKey) -> str:
        """Build cache key with tenant isolation"""
        return f"cache:{tenant_id}:{key}"

    def _serialize(
            self,
            value: CacheValue,
            serialization: str,
            compression: bool = False
    ) -> Union[str, bytes]:
        """Serialize value for storage"""
        try:
            if serialization == "json":
                serialized = json.dumps(value, default=str)
            elif serialization == "pickle":
                serialized = pickle.dumps(value)
            else:  # string
                serialized = str(value)

            if compression and len(str(serialized)) > 1000:  # Compress large values
                import gzip
                if isinstance(serialized, str):
                    serialized = serialized.encode('utf-8')
                serialized = gzip.compress(serialized)

            return serialized

        except Exception as e:
            self.logger.warning(
                "Failed to serialize value, using string representation",
                error=str(e)
            )
            return str(value)

    def _deserialize(
            self,
            value: Union[str, bytes],
            serialization: str
    ) -> CacheValue:
        """Deserialize value from storage"""
        try:
            # Check if value is compressed
            if isinstance(value, bytes):
                try:
                    import gzip
                    value = gzip.decompress(value).decode('utf-8')
                except:
                    # Not compressed or different format
                    if serialization == "pickle":
                        return pickle.loads(value)
                    else:
                        value = value.decode('utf-8')

            if serialization == "json":
                return json.loads(value)
            elif serialization == "pickle":
                return pickle.loads(value)
            else:  # string
                return value

        except Exception as e:
            self.logger.warning(
                "Failed to deserialize value, returning as string",
                error=str(e)
            )
            return str(value)

    def _record_hit(self, tenant_id: TenantId) -> None:
        """Record cache hit"""
        if tenant_id not in self._stats:
            self._stats[tenant_id] = CacheStats()
        self._stats[tenant_id].hits += 1

    def _record_miss(self, tenant_id: TenantId) -> None:
        """Record cache miss"""
        if tenant_id not in self._stats:
            self._stats[tenant_id] = CacheStats()
        self._stats[tenant_id].misses += 1

    def _record_set(self, tenant_id: TenantId, size_bytes: int = 0) -> None:
        """Record cache set operation"""
        if tenant_id not in self._stats:
            self._stats[tenant_id] = CacheStats()
        self._stats[tenant_id].sets += 1
        self._stats[tenant_id].total_size_bytes += size_bytes

    def _record_delete(self, tenant_id: TenantId) -> None:
        """Record cache delete operation"""
        if tenant_id not in self._stats:
            self._stats[tenant_id] = CacheStats()
        self._stats[tenant_id].deletes += 1


# Dependency injection function
async def get_cache_repository() -> CacheRepository:
    """
    Get cache repository instance for dependency injection

    Returns:
        CacheRepository instance
    """
    from .session_repository import get_redis_client
    redis_client = await get_redis_client()
    return CacheRepository(redis_client)