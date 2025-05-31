"""
Rate Limit Repository Implementation
===================================

Redis repository for rate limiting and quota management using multiple
algorithms including sliding window, token bucket, and fixed window approaches.

Features:
- Multiple rate limiting algorithms
- Multi-tenant isolation
- API key and user-based limiting
- Burst handling with token buckets
- Rate limit analytics and monitoring
- Automatic cleanup of expired data
"""

from datetime import datetime, timedelta
from typing import Tuple, Optional, Dict, Any, List
from redis.asyncio import Redis
import time
import math
import json
import structlog
from dataclasses import dataclass, asdict
from enum import Enum

from .exceptions import (
    RepositoryError, ValidationError, TimeoutError
)

# Type definitions
TenantId = str
ApiKeyId = str
UserId = str


class RateLimitAlgorithm(str, Enum):
    SLIDING_WINDOW = "sliding_window"
    TOKEN_BUCKET = "token_bucket"
    FIXED_WINDOW = "fixed_window"


class RateLimitType(str, Enum):
    API_KEY = "api_key"
    USER = "user"
    IP_ADDRESS = "ip_address"
    TENANT = "tenant"
    ENDPOINT = "endpoint"


@dataclass
class RateLimitConfig:
    """Rate limit configuration"""
    limit: int  # Number of requests allowed
    window_seconds: int  # Time window in seconds
    algorithm: RateLimitAlgorithm = RateLimitAlgorithm.SLIDING_WINDOW
    burst_limit: Optional[int] = None  # For token bucket
    refill_rate: Optional[float] = None  # Tokens per second for token bucket

    def __post_init__(self):
        if self.algorithm == RateLimitAlgorithm.TOKEN_BUCKET:
            if self.burst_limit is None:
                self.burst_limit = self.limit * 2  # Default burst is 2x limit
            if self.refill_rate is None:
                self.refill_rate = self.limit / self.window_seconds


@dataclass
class RateLimitResult:
    """Result of a rate limit check"""
    allowed: bool
    current_count: int
    limit: int
    remaining: int
    reset_time: int  # Unix timestamp
    retry_after_seconds: Optional[int] = None

    def to_headers(self) -> Dict[str, str]:
        """Convert to HTTP headers"""
        headers = {
            "X-RateLimit-Limit": str(self.limit),
            "X-RateLimit-Remaining": str(self.remaining),
            "X-RateLimit-Reset": str(self.reset_time)
        }

        if self.retry_after_seconds:
            headers["Retry-After"] = str(self.retry_after_seconds)

        return headers


@dataclass
class TokenBucketState:
    """Token bucket state for rate limiting"""
    tokens: float
    last_refill: float  # Unix timestamp
    capacity: int
    refill_rate: float

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TokenBucketState":
        return cls(**data)


class RateLimitRepository:
    """
    Repository for rate limiting using Redis with multiple algorithms

    Supports:
    - Sliding window rate limiting (most accurate)
    - Token bucket rate limiting (allows bursts)
    - Fixed window rate limiting (most efficient)
    - Multi-dimensional rate limiting (by API key, user, IP, etc.)
    """

    def __init__(self, redis_client: Redis):
        """
        Initialize rate limit repository

        Args:
            redis_client: Redis async client instance
        """
        self.redis = redis_client
        self.logger = structlog.get_logger("RateLimitRepository")

    async def check_rate_limit(
            self,
            tenant_id: TenantId,
            identifier: str,  # API key, user ID, IP address, etc.
            config: RateLimitConfig,
            limit_type: RateLimitType = RateLimitType.API_KEY
    ) -> RateLimitResult:
        """
        Check if request is within rate limit

        Args:
            tenant_id: Tenant identifier
            identifier: Unique identifier for rate limiting
            config: Rate limit configuration
            limit_type: Type of rate limiting

        Returns:
            RateLimitResult with decision and metadata

        Raises:
            RepositoryError: If rate limit check fails
        """
        try:
            if config.algorithm == RateLimitAlgorithm.SLIDING_WINDOW:
                return await self._check_sliding_window(tenant_id, identifier, config, limit_type)
            elif config.algorithm == RateLimitAlgorithm.TOKEN_BUCKET:
                return await self._check_token_bucket(tenant_id, identifier, config, limit_type)
            elif config.algorithm == RateLimitAlgorithm.FIXED_WINDOW:
                return await self._check_fixed_window(tenant_id, identifier, config, limit_type)
            else:
                raise ValidationError(f"Unknown rate limit algorithm: {config.algorithm}")

        except Exception as e:
            self.logger.error(
                "Rate limit check failed",
                tenant_id=tenant_id,
                identifier=identifier,
                algorithm=config.algorithm,
                error=str(e)
            )
            raise RepositoryError(f"Failed to check rate limit: {e}", original_error=e)

    async def _check_sliding_window(
            self,
            tenant_id: TenantId,
            identifier: str,
            config: RateLimitConfig,
            limit_type: RateLimitType
    ) -> RateLimitResult:
        """Check rate limit using sliding window algorithm"""
        current_time = time.time()
        window_start = current_time - config.window_seconds

        # Use sorted set for sliding window
        key = self._get_rate_limit_key(tenant_id, identifier, limit_type, "sliding")

        # Lua script for atomic sliding window check
        lua_script = """
        local key = KEYS[1]
        local window_start = tonumber(ARGV[1])
        local current_time = tonumber(ARGV[2])
        local limit = tonumber(ARGV[3])
        local window_seconds = tonumber(ARGV[4])

        -- Remove expired entries
        redis.call('ZREMRANGEBYSCORE', key, 0, window_start)

        -- Count current requests in window
        local current_count = redis.call('ZCARD', key)

        -- Check if limit exceeded
        local allowed = current_count < limit
        local remaining = math.max(0, limit - current_count)

        if allowed then
            -- Add current request
            redis.call('ZADD', key, current_time, current_time)
            current_count = current_count + 1
            remaining = remaining - 1
        end

        -- Set expiration
        redis.call('EXPIRE', key, window_seconds)

        -- Calculate reset time (when oldest entry expires)
        local oldest = redis.call('ZRANGE', key, 0, 0, 'WITHSCORES')
        local reset_time = current_time + window_seconds
        if #oldest > 0 then
            reset_time = oldest[2] + window_seconds
        end

        return {allowed and 1 or 0, current_count, remaining, reset_time}
        """

        result = await self.redis.eval(
            lua_script,
            1,
            key,
            window_start,
            current_time,
            config.limit,
            config.window_seconds
        )

        allowed = bool(result[0])
        current_count = int(result[1])
        remaining = int(result[2])
        reset_time = int(result[3])

        # Calculate retry after if blocked
        retry_after = None
        if not allowed:
            retry_after = max(1, reset_time - current_time)

        self.logger.debug(
            "Sliding window rate limit check",
            tenant_id=tenant_id,
            identifier=identifier,
            allowed=allowed,
            current_count=current_count,
            limit=config.limit,
            remaining=remaining
        )

        return RateLimitResult(
            allowed=allowed,
            current_count=current_count,
            limit=config.limit,
            remaining=remaining,
            reset_time=reset_time,
            retry_after_seconds=retry_after
        )

    async def _check_token_bucket(
            self,
            tenant_id: TenantId,
            identifier: str,
            config: RateLimitConfig,
            limit_type: RateLimitType
    ) -> RateLimitResult:
        """Check rate limit using token bucket algorithm"""
        current_time = time.time()
        key = self._get_rate_limit_key(tenant_id, identifier, limit_type, "bucket")

        # Lua script for atomic token bucket check
        lua_script = """
        local key = KEYS[1]
        local current_time = tonumber(ARGV[1])
        local capacity = tonumber(ARGV[2])
        local refill_rate = tonumber(ARGV[3])
        local window_seconds = tonumber(ARGV[4])

        -- Get current bucket state
        local bucket_data = redis.call('HGETALL', key)
        local tokens = capacity
        local last_refill = current_time

        if #bucket_data > 0 then
            local bucket = {}
            for i = 1, #bucket_data, 2 do
                bucket[bucket_data[i]] = bucket_data[i + 1]
            end
            tokens = tonumber(bucket.tokens) or capacity
            last_refill = tonumber(bucket.last_refill) or current_time
        end

        -- Refill tokens based on time elapsed
        local time_elapsed = current_time - last_refill
        local new_tokens = math.min(capacity, tokens + (time_elapsed * refill_rate))

        -- Check if request is allowed
        local allowed = new_tokens >= 1
        local remaining = math.floor(new_tokens)

        if allowed then
            new_tokens = new_tokens - 1
            remaining = remaining - 1
        end

        -- Update bucket state
        redis.call('HSET', key, 
            'tokens', new_tokens,
            'last_refill', current_time,
            'capacity', capacity,
            'refill_rate', refill_rate
        )
        redis.call('EXPIRE', key, window_seconds * 2)

        -- Calculate reset time (when bucket will be full)
        local reset_time = current_time + ((capacity - new_tokens) / refill_rate)

        return {allowed and 1 or 0, capacity - remaining, remaining, reset_time}
        """

        result = await self.redis.eval(
            lua_script,
            1,
            key,
            current_time,
            config.burst_limit,
            config.refill_rate,
            config.window_seconds
        )

        allowed = bool(result[0])
        current_count = int(result[1])
        remaining = int(result[2])
        reset_time = int(result[3])

        # Calculate retry after if blocked
        retry_after = None
        if not allowed:
            retry_after = max(1, 1 / config.refill_rate)  # Time to get next token

        self.logger.debug(
            "Token bucket rate limit check",
            tenant_id=tenant_id,
            identifier=identifier,
            allowed=allowed,
            tokens_remaining=remaining,
            bucket_capacity=config.burst_limit
        )

        return RateLimitResult(
            allowed=allowed,
            current_count=current_count,
            limit=config.burst_limit,
            remaining=remaining,
            reset_time=reset_time,
            retry_after_seconds=retry_after
        )

    async def _check_fixed_window(
            self,
            tenant_id: TenantId,
            identifier: str,
            config: RateLimitConfig,
            limit_type: RateLimitType
    ) -> RateLimitResult:
        """Check rate limit using fixed window algorithm"""
        current_time = time.time()
        window_start = int(current_time // config.window_seconds) * config.window_seconds
        window_key = f"{window_start}"

        key = self._get_rate_limit_key(tenant_id, identifier, limit_type, f"fixed:{window_key}")

        # Lua script for atomic fixed window check
        lua_script = """
        local key = KEYS[1]
        local limit = tonumber(ARGV[1])
        local window_seconds = tonumber(ARGV[2])
        local reset_time = tonumber(ARGV[3])

        -- Get current count
        local current_count = tonumber(redis.call('GET', key)) or 0

        -- Check if limit exceeded
        local allowed = current_count < limit
        local remaining = math.max(0, limit - current_count)

        if allowed then
            -- Increment counter
            current_count = redis.call('INCR', key)
            remaining = remaining - 1
            -- Set expiration if this is the first request in the window
            if current_count == 1 then
                redis.call('EXPIRE', key, window_seconds)
            end
        end

        return {allowed and 1 or 0, current_count, remaining, reset_time}
        """

        reset_time = window_start + config.window_seconds

        result = await self.redis.eval(
            lua_script,
            1,
            key,
            config.limit,
            config.window_seconds,
            reset_time
        )

        allowed = bool(result[0])
        current_count = int(result[1])
        remaining = int(result[2])
        reset_time = int(result[3])

        # Calculate retry after if blocked
        retry_after = None
        if not allowed:
            retry_after = max(1, reset_time - current_time)

        self.logger.debug(
            "Fixed window rate limit check",
            tenant_id=tenant_id,
            identifier=identifier,
            allowed=allowed,
            current_count=current_count,
            window_start=window_start,
            reset_time=reset_time
        )

        return RateLimitResult(
            allowed=allowed,
            current_count=current_count,
            limit=config.limit,
            remaining=remaining,
            reset_time=reset_time,
            retry_after_seconds=retry_after
        )

    async def increment_counter(
            self,
            tenant_id: TenantId,
            identifier: str,
            config: RateLimitConfig,
            limit_type: RateLimitType = RateLimitType.API_KEY
    ) -> int:
        """
        Increment rate limit counter without checking limit

        Args:
            tenant_id: Tenant identifier
            identifier: Unique identifier
            config: Rate limit configuration
            limit_type: Type of rate limiting

        Returns:
            Current count after increment
        """
        try:
            if config.algorithm == RateLimitAlgorithm.SLIDING_WINDOW:
                return await self._increment_sliding_window(tenant_id, identifier, config, limit_type)
            elif config.algorithm == RateLimitAlgorithm.FIXED_WINDOW:
                return await self._increment_fixed_window(tenant_id, identifier, config, limit_type)
            else:
                # Token bucket doesn't support simple increment
                result = await self.check_rate_limit(tenant_id, identifier, config, limit_type)
                return result.current_count

        except Exception as e:
            self.logger.error(
                "Failed to increment counter",
                tenant_id=tenant_id,
                identifier=identifier,
                error=str(e)
            )
            raise RepositoryError(f"Failed to increment counter: {e}", original_error=e)

    async def _increment_sliding_window(
            self,
            tenant_id: TenantId,
            identifier: str,
            config: RateLimitConfig,
            limit_type: RateLimitType
    ) -> int:
        """Increment sliding window counter"""
        current_time = time.time()
        key = self._get_rate_limit_key(tenant_id, identifier, limit_type, "sliding")

        # Add current request and clean up old entries
        pipe = self.redis.pipeline()
        pipe.zadd(key, {str(current_time): current_time})
        pipe.zremrangebyscore(key, 0, current_time - config.window_seconds)
        pipe.zcard(key)
        pipe.expire(key, config.window_seconds)

        results = await pipe.execute()
        return results[2]  # ZCARD result

    async def _increment_fixed_window(
            self,
            tenant_id: TenantId,
            identifier: str,
            config: RateLimitConfig,
            limit_type: RateLimitType
    ) -> int:
        """Increment fixed window counter"""
        current_time = time.time()
        window_start = int(current_time // config.window_seconds) * config.window_seconds
        window_key = f"{window_start}"

        key = self._get_rate_limit_key(tenant_id, identifier, limit_type, f"fixed:{window_key}")

        # Increment and set expiration
        pipe = self.redis.pipeline()
        pipe.incr(key)
        pipe.expire(key, config.window_seconds)

        results = await pipe.execute()
        return results[0]  # INCR result

    async def get_remaining_quota(
            self,
            tenant_id: TenantId,
            identifier: str,
            config: RateLimitConfig,
            limit_type: RateLimitType = RateLimitType.API_KEY
    ) -> int:
        """
        Get remaining quota for identifier

        Args:
            tenant_id: Tenant identifier
            identifier: Unique identifier
            config: Rate limit configuration
            limit_type: Type of rate limiting

        Returns:
            Number of remaining requests
        """
        try:
            result = await self.check_rate_limit(tenant_id, identifier, config, limit_type)
            return result.remaining

        except Exception as e:
            self.logger.error(
                "Failed to get remaining quota",
                tenant_id=tenant_id,
                identifier=identifier,
                error=str(e)
            )
            raise RepositoryError(f"Failed to get remaining quota: {e}", original_error=e)

    async def reset_rate_limit(
            self,
            tenant_id: TenantId,
            identifier: str,
            limit_type: RateLimitType = RateLimitType.API_KEY
    ) -> bool:
        """
        Reset rate limit counter for identifier

        Args:
            tenant_id: Tenant identifier
            identifier: Unique identifier
            limit_type: Type of rate limiting

        Returns:
            True if reset successfully
        """
        try:
            # Delete all rate limit keys for this identifier
            patterns = [
                self._get_rate_limit_key(tenant_id, identifier, limit_type, "sliding"),
                self._get_rate_limit_key(tenant_id, identifier, limit_type, "bucket"),
                self._get_rate_limit_key(tenant_id, identifier, limit_type, "fixed:*")
            ]

            deleted_count = 0
            for pattern in patterns:
                if "*" in pattern:
                    # Handle wildcard patterns
                    async for key in self.redis.scan_iter(match=pattern):
                        await self.redis.delete(key)
                        deleted_count += 1
                else:
                    # Direct key deletion
                    deleted = await self.redis.delete(pattern)
                    deleted_count += deleted

            self.logger.info(
                "Rate limit reset",
                tenant_id=tenant_id,
                identifier=identifier,
                limit_type=limit_type,
                keys_deleted=deleted_count
            )

            return deleted_count > 0

        except Exception as e:
            self.logger.error(
                "Failed to reset rate limit",
                tenant_id=tenant_id,
                identifier=identifier,
                error=str(e)
            )
            raise RepositoryError(f"Failed to reset rate limit: {e}", original_error=e)

    async def get_rate_limit_info(
            self,
            tenant_id: TenantId,
            identifier: str,
            config: RateLimitConfig,
            limit_type: RateLimitType = RateLimitType.API_KEY
    ) -> Dict[str, Any]:
        """
        Get comprehensive rate limit information

        Args:
            tenant_id: Tenant identifier
            identifier: Unique identifier
            config: Rate limit configuration
            limit_type: Type of rate limiting

        Returns:
            Dictionary with rate limit details
        """
        try:
            result = await self.check_rate_limit(tenant_id, identifier, config, limit_type)

            info = {
                "algorithm": config.algorithm.value,
                "limit": config.limit,
                "window_seconds": config.window_seconds,
                "current_count": result.current_count,
                "remaining": result.remaining,
                "reset_time": result.reset_time,
                "reset_time_iso": datetime.fromtimestamp(result.reset_time).isoformat(),
                "allowed": result.allowed
            }

            if config.algorithm == RateLimitAlgorithm.TOKEN_BUCKET:
                info.update({
                    "burst_limit": config.burst_limit,
                    "refill_rate": config.refill_rate,
                    "tokens_available": result.remaining
                })

            return info

        except Exception as e:
            self.logger.error(
                "Failed to get rate limit info",
                tenant_id=tenant_id,
                identifier=identifier,
                error=str(e)
            )
            raise RepositoryError(f"Failed to get rate limit info: {e}", original_error=e)

    async def get_rate_limit_analytics(
            self,
            tenant_id: TenantId,
            limit_type: RateLimitType,
            hours: int = 24
    ) -> Dict[str, Any]:
        """
        Get rate limiting analytics for a tenant

        Args:
            tenant_id: Tenant identifier
            limit_type: Type of rate limiting
            hours: Number of hours to analyze

        Returns:
            Analytics data
        """
        try:
            analytics = {
                "total_requests": 0,
                "blocked_requests": 0,
                "unique_identifiers": set(),
                "top_identifiers": {},
                "hourly_breakdown": {}
            }

            # This is a simplified analytics implementation
            # In production, you'd want to use separate analytics storage
            pattern = f"rate_limit:{tenant_id}:*:{limit_type.value}:*"

            async for key in self.redis.scan_iter(match=pattern):
                try:
                    # Extract identifier from key
                    parts = key.split(":")
                    if len(parts) >= 4:
                        identifier = parts[3]
                        analytics["unique_identifiers"].add(identifier)

                        # Count requests (simplified)
                        if key.endswith("sliding"):
                            count = await self.redis.zcard(key)
                            analytics["total_requests"] += count
                            analytics["top_identifiers"][identifier] = count
                        elif key.endswith("bucket"):
                            # For token bucket, we can't easily get request count
                            pass
                        elif "fixed:" in key:
                            count = await self.redis.get(key)
                            if count:
                                analytics["total_requests"] += int(count)
                                analytics["top_identifiers"][identifier] = int(count)

                except Exception as e:
                    self.logger.warning(
                        "Failed to analyze rate limit key",
                        key=key,
                        error=str(e)
                    )
                    continue

            # Convert set to count
            analytics["unique_identifiers"] = len(analytics["unique_identifiers"])

            # Sort top identifiers
            analytics["top_identifiers"] = dict(
                sorted(analytics["top_identifiers"].items(),
                       key=lambda x: x[1], reverse=True)[:10]
            )

            return analytics

        except Exception as e:
            self.logger.error(
                "Failed to get rate limit analytics",
                tenant_id=tenant_id,
                error=str(e)
            )
            return {"error": str(e)}

    def _get_rate_limit_key(
            self,
            tenant_id: TenantId,
            identifier: str,
            limit_type: RateLimitType,
            suffix: str
    ) -> str:
        """Generate Redis key for rate limiting"""
        return f"rate_limit:{tenant_id}:{identifier}:{limit_type.value}:{suffix}"


# Dependency injection function
async def get_rate_limit_repository() -> RateLimitRepository:
    """
    Get rate limit repository instance for dependency injection

    Returns:
        RateLimitRepository instance
    """
    from .session_repository import get_redis_client
    redis_client = await get_redis_client()
    return RateLimitRepository(redis_client)