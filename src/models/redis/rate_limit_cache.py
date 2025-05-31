# src/models/redis/rate_limit_cache.py
"""
Redis data structures for rate limiting.
Implements sliding window and token bucket rate limiting algorithms.
"""

from datetime import datetime, timedelta
from typing import Dict, Any, Optional, List, Tuple
from pydantic import BaseModel, Field, validator
import time
import math

from src.models.base_model import BaseRedisModel
from src.models.types import TenantId, Priority


class RateLimitWindow(BaseRedisModel):
    """
    Sliding window rate limiting using Redis sorted sets.
    Tracks requests within time windows for accurate rate limiting.
    """

    # Identifiers
    tenant_id: TenantId
    api_key_id: Optional[str] = None
    user_id: Optional[str] = None
    endpoint: Optional[str] = None

    # Rate limit configuration
    limit_per_minute: int = Field(..., gt=0)
    limit_per_hour: int = Field(..., gt=0)
    limit_per_day: int = Field(..., gt=0)

    # Current window tracking
    current_minute_count: int = Field(default=0, ge=0)
    current_hour_count: int = Field(default=0, ge=0)
    current_day_count: int = Field(default=0, ge=0)

    # Window start times
    minute_window_start: datetime = Field(default_factory=datetime.utcnow)
    hour_window_start: datetime = Field(default_factory=datetime.utcnow)
    day_window_start: datetime = Field(default_factory=datetime.utcnow)

    # Metadata
    last_request_time: Optional[datetime] = None
    blocked_until: Optional[datetime] = None
    total_requests: int = Field(default=0, ge=0)
    total_blocked: int = Field(default=0, ge=0)

    @staticmethod
    def get_cache_key(
            tenant_id: TenantId,
            identifier: str,
            window_type: str = "minute"
    ) -> str:
        """Generate Redis cache key for rate limit window"""
        timestamp = int(time.time())

        if window_type == "minute":
            window_timestamp = timestamp // 60
        elif window_type == "hour":
            window_timestamp = timestamp // 3600
        elif window_type == "day":
            window_timestamp = timestamp // 86400
        else:
            window_timestamp = timestamp // 60

        return f"rate_limit:{tenant_id}:{identifier}:{window_type}:{window_timestamp}"

    @staticmethod
    def get_sorted_set_key(tenant_id: TenantId, identifier: str) -> str:
        """Generate Redis sorted set key for sliding window"""
        return f"rate_limit_window:{tenant_id}:{identifier}"

    def is_rate_limited(self) -> Tuple[bool, Optional[str]]:
        """
        Check if rate limit is exceeded.

        Returns:
            Tuple of (is_limited, reason)
        """
        now = datetime.utcnow()

        # Check if currently blocked
        if self.blocked_until and now < self.blocked_until:
            return True, f"Blocked until {self.blocked_until.isoformat()}"

        # Check minute limit
        if self.current_minute_count >= self.limit_per_minute:
            return True, f"Minute limit exceeded ({self.limit_per_minute})"

        # Check hour limit
        if self.current_hour_count >= self.limit_per_hour:
            return True, f"Hour limit exceeded ({self.limit_per_hour})"

        # Check day limit
        if self.current_day_count >= self.limit_per_day:
            return True, f"Day limit exceeded ({self.limit_per_day})"

        return False, None

    def increment_request_count(self) -> None:
        """Increment request counters"""
        now = datetime.utcnow()

        # Reset windows if needed
        self._reset_windows_if_needed(now)

        # Increment counters
        self.current_minute_count += 1
        self.current_hour_count += 1
        self.current_day_count += 1
        self.total_requests += 1
        self.last_request_time = now

        self.update_timestamp()

    def increment_blocked_count(self) -> None:
        """Increment blocked request counter"""
        self.total_blocked += 1
        self.update_timestamp()

    def apply_temporary_block(self, duration_minutes: int) -> None:
        """Apply temporary block for specified duration"""
        self.blocked_until = datetime.utcnow() + timedelta(minutes=duration_minutes)
        self.update_timestamp()

    def clear_temporary_block(self) -> None:
        """Clear temporary block"""
        self.blocked_until = None
        self.update_timestamp()

    def _reset_windows_if_needed(self, now: datetime) -> None:
        """Reset rate limit windows if time has passed"""
        # Reset minute window
        if now >= self.minute_window_start + timedelta(minutes=1):
            self.current_minute_count = 0
            self.minute_window_start = now.replace(second=0, microsecond=0)

        # Reset hour window
        if now >= self.hour_window_start + timedelta(hours=1):
            self.current_hour_count = 0
            self.hour_window_start = now.replace(minute=0, second=0, microsecond=0)

        # Reset day window
        if now >= self.day_window_start + timedelta(days=1):
            self.current_day_count = 0
            self.day_window_start = now.replace(hour=0, minute=0, second=0, microsecond=0)

    def get_reset_times(self) -> Dict[str, datetime]:
        """Get when each window will reset"""
        return {
            "minute": self.minute_window_start + timedelta(minutes=1),
            "hour": self.hour_window_start + timedelta(hours=1),
            "day": self.day_window_start + timedelta(days=1)
        }

    def get_remaining_requests(self) -> Dict[str, int]:
        """Get remaining requests for each window"""
        return {
            "minute": max(0, self.limit_per_minute - self.current_minute_count),
            "hour": max(0, self.limit_per_hour - self.current_hour_count),
            "day": max(0, self.limit_per_day - self.current_day_count)
        }


class TokenBucket(BaseRedisModel):
    """
    Token bucket rate limiting algorithm.
    Allows burst traffic while maintaining average rate limits.
    """

    # Identifiers
    tenant_id: TenantId
    bucket_id: str

    # Bucket configuration
    capacity: int = Field(..., gt=0)  # Maximum tokens
    refill_rate: float = Field(..., gt=0)  # Tokens per second

    # Current state
    current_tokens: float = Field(..., ge=0)
    last_refill: datetime = Field(default_factory=datetime.utcnow)

    # Statistics
    total_requests: int = Field(default=0, ge=0)
    total_denied: int = Field(default=0, ge=0)
    last_request_time: Optional[datetime] = None

    @staticmethod
    def get_cache_key(tenant_id: TenantId, bucket_id: str) -> str:
        """Generate Redis cache key for token bucket"""
        return f"token_bucket:{tenant_id}:{bucket_id}"

    def refill_tokens(self) -> None:
        """Refill tokens based on elapsed time"""
        now = datetime.utcnow()
        time_elapsed = (now - self.last_refill).total_seconds()

        # Calculate tokens to add
        tokens_to_add = time_elapsed * self.refill_rate

        # Add tokens up to capacity
        self.current_tokens = min(self.capacity, self.current_tokens + tokens_to_add)
        self.last_refill = now

        self.update_timestamp()

    def consume_tokens(self, tokens_requested: int = 1) -> bool:
        """
        Attempt to consume tokens from bucket.

        Args:
            tokens_requested: Number of tokens to consume

        Returns:
            True if tokens were available and consumed, False otherwise
        """
        # Refill tokens first
        self.refill_tokens()

        # Check if enough tokens available
        if self.current_tokens >= tokens_requested:
            self.current_tokens -= tokens_requested
            self.total_requests += 1
            self.last_request_time = datetime.utcnow()
            self.update_timestamp()
            return True
        else:
            self.total_denied += 1
            self.update_timestamp()
            return False

    def get_time_until_available(self, tokens_needed: int = 1) -> float:
        """
        Calculate time until specified tokens will be available.

        Args:
            tokens_needed: Number of tokens needed

        Returns:
            Time in seconds until tokens available
        """
        self.refill_tokens()

        if self.current_tokens >= tokens_needed:
            return 0.0

        tokens_deficit = tokens_needed - self.current_tokens
        return tokens_deficit / self.refill_rate

    def get_bucket_status(self) -> Dict[str, Any]:
        """Get current bucket status"""
        self.refill_tokens()

        return {
            "current_tokens": self.current_tokens,
            "capacity": self.capacity,
            "refill_rate": self.refill_rate,
            "fill_percentage": (self.current_tokens / self.capacity) * 100,
            "total_requests": self.total_requests,
            "total_denied": self.total_denied,
            "last_request": self.last_request_time
        }


class RateLimitTier(BaseRedisModel):
    """
    Rate limit tier configuration for different user/API key tiers.
    Defines limits and burst allowances for different service levels.
    """

    tier_name: str = Field(..., max_length=50)
    tenant_id: TenantId

    # Rate limits
    requests_per_minute: int = Field(..., gt=0)
    requests_per_hour: int = Field(..., gt=0)
    requests_per_day: int = Field(..., gt=0)

    # Burst configuration
    burst_capacity: int = Field(..., gt=0)
    burst_refill_rate: float = Field(..., gt=0)

    # Feature limits
    max_concurrent_requests: int = Field(default=10, gt=0)
    max_request_size_bytes: int = Field(default=1048576, gt=0)  # 1MB
    max_response_time_ms: int = Field(default=30000, gt=0)

    # Priority handling
    priority_queue_enabled: bool = Field(default=False)
    priority_multiplier: float = Field(default=1.0, gt=0)

    @staticmethod
    def get_cache_key(tenant_id: TenantId, tier_name: str) -> str:
        """Generate Redis cache key for rate limit tier"""
        return f"rate_limit_tier:{tenant_id}:{tier_name}"

    def create_token_bucket(self, bucket_id: str) -> TokenBucket:
        """Create a token bucket with this tier's configuration"""
        return TokenBucket(
            tenant_id=self.tenant_id,
            bucket_id=bucket_id,
            capacity=self.burst_capacity,
            refill_rate=self.burst_refill_rate,
            current_tokens=self.burst_capacity  # Start with full bucket
        )

    def create_rate_limit_window(
            self,
            api_key_id: Optional[str] = None,
            user_id: Optional[str] = None,
            endpoint: Optional[str] = None
    ) -> RateLimitWindow:
        """Create a rate limit window with this tier's configuration"""
        return RateLimitWindow(
            tenant_id=self.tenant_id,
            api_key_id=api_key_id,
            user_id=user_id,
            endpoint=endpoint,
            limit_per_minute=self.requests_per_minute,
            limit_per_hour=self.requests_per_hour,
            limit_per_day=self.requests_per_day
        )

    def calculate_priority_weight(self, base_priority: Priority) -> float:
        """Calculate priority weight based on tier and request priority"""
        priority_weights = {
            Priority.LOW: 0.5,
            Priority.NORMAL: 1.0,
            Priority.HIGH: 2.0,
            Priority.URGENT: 4.0,
            Priority.CRITICAL: 8.0
        }

        base_weight = priority_weights.get(base_priority, 1.0)
        return base_weight * self.priority_multiplier


class ConcurrentRequestTracker(BaseRedisModel):
    """
    Track concurrent requests per user/API key to enforce concurrency limits.
    """

    tenant_id: TenantId
    identifier: str  # user_id or api_key_id

    # Current state
    active_requests: Dict[str, datetime] = Field(default_factory=dict)  # request_id -> start_time
    max_concurrent: int = Field(..., gt=0)

    # Statistics
    total_requests: int = Field(default=0, ge=0)
    total_rejected: int = Field(default=0, ge=0)
    peak_concurrent: int = Field(default=0, ge=0)

    @staticmethod
    def get_cache_key(tenant_id: TenantId, identifier: str) -> str:
        """Generate Redis cache key for concurrent request tracking"""
        return f"concurrent_requests:{tenant_id}:{identifier}"

    def start_request(self, request_id: str) -> bool:
        """
        Start tracking a new request.

        Args:
            request_id: Unique request identifier

        Returns:
            True if request can proceed, False if limit exceeded
        """
        # Clean up old requests first
        self._cleanup_old_requests()

        # Check if limit would be exceeded
        if len(self.active_requests) >= self.max_concurrent:
            self.total_rejected += 1
            self.update_timestamp()
            return False

        # Add request
        self.active_requests[request_id] = datetime.utcnow()
        self.total_requests += 1

        # Update peak if necessary
        current_count = len(self.active_requests)
        if current_count > self.peak_concurrent:
            self.peak_concurrent = current_count

        self.update_timestamp()
        return True

    def end_request(self, request_id: str) -> None:
        """End tracking for a request"""
        self.active_requests.pop(request_id, None)
        self.update_timestamp()

    def get_current_count(self) -> int:
        """Get current number of active requests"""
        self._cleanup_old_requests()
        return len(self.active_requests)

    def get_remaining_capacity(self) -> int:
        """Get number of additional requests that can be started"""
        current = self.get_current_count()
        return max(0, self.max_concurrent - current)

    def _cleanup_old_requests(self, timeout_minutes: int = 10) -> None:
        """Remove requests that have been active too long"""
        cutoff_time = datetime.utcnow() - timedelta(minutes=timeout_minutes)

        old_requests = [
            req_id for req_id, start_time in self.active_requests.items()
            if start_time < cutoff_time
        ]

        for req_id in old_requests:
            del self.active_requests[req_id]

        if old_requests:
            self.update_timestamp()


class RateLimitViolation(BaseRedisModel):
    """
    Track rate limit violations for monitoring and potential blocking.
    """

    tenant_id: TenantId
    identifier: str  # user_id or api_key_id

    # Violation tracking
    violations_last_hour: int = Field(default=0, ge=0)
    violations_last_day: int = Field(default=0, ge=0)
    total_violations: int = Field(default=0, ge=0)

    # Timestamps
    first_violation: Optional[datetime] = None
    last_violation: Optional[datetime] = None

    # Current penalties
    warning_level: int = Field(default=0, ge=0)  # 0=none, 1=warning, 2=temp_block, 3=investigation
    blocked_until: Optional[datetime] = None

    @staticmethod
    def get_cache_key(tenant_id: TenantId, identifier: str) -> str:
        """Generate Redis cache key for rate limit violations"""
        return f"rate_violations:{tenant_id}:{identifier}"

    def record_violation(self, violation_type: str = "rate_limit") -> None:
        """Record a new rate limit violation"""
        now = datetime.utcnow()

        # Update counters
        self.violations_last_hour += 1
        self.violations_last_day += 1
        self.total_violations += 1

        # Update timestamps
        if not self.first_violation:
            self.first_violation = now
        self.last_violation = now

        # Update warning level based on frequency
        self._update_warning_level()

        self.update_timestamp()

    def reset_hourly_violations(self) -> None:
        """Reset hourly violation counter"""
        self.violations_last_hour = 0
        self.update_timestamp()

    def reset_daily_violations(self) -> None:
        """Reset daily violation counter"""
        self.violations_last_day = 0
        self.update_timestamp()

    def apply_temporary_block(self, duration_minutes: int) -> None:
        """Apply temporary block due to violations"""
        self.blocked_until = datetime.utcnow() + timedelta(minutes=duration_minutes)
        self.warning_level = max(self.warning_level, 2)
        self.update_timestamp()

    def clear_block(self) -> None:
        """Clear temporary block"""
        self.blocked_until = None
        if self.warning_level == 2:
            self.warning_level = 1  # Reduce to warning
        self.update_timestamp()

    def is_blocked(self) -> bool:
        """Check if currently blocked"""
        if not self.blocked_until:
            return False
        return datetime.utcnow() < self.blocked_until

    def _update_warning_level(self) -> None:
        """Update warning level based on violation frequency"""
        # Escalate based on violations in last hour
        if self.violations_last_hour >= 100:
            self.warning_level = 3  # Investigation required
        elif self.violations_last_hour >= 20:
            self.warning_level = 2  # Temporary block
        elif self.violations_last_hour >= 5:
            self.warning_level = 1  # Warning
        else:
            self.warning_level = 0  # No warning

    def get_violation_summary(self) -> Dict[str, Any]:
        """Get summary of violations and current status"""
        return {
            "total_violations": self.total_violations,
            "violations_last_hour": self.violations_last_hour,
            "violations_last_day": self.violations_last_day,
            "warning_level": self.warning_level,
            "is_blocked": self.is_blocked(),
            "blocked_until": self.blocked_until,
            "first_violation": self.first_violation,
            "last_violation": self.last_violation
        }


# Default rate limit tiers
DEFAULT_RATE_LIMIT_TIERS = {
    "basic": {
        "requests_per_minute": 100,
        "requests_per_hour": 1000,
        "requests_per_day": 10000,
        "burst_capacity": 50,
        "burst_refill_rate": 1.67,  # ~100/minute
        "max_concurrent_requests": 5
    },
    "standard": {
        "requests_per_minute": 1000,
        "requests_per_hour": 10000,
        "requests_per_day": 100000,
        "burst_capacity": 200,
        "burst_refill_rate": 16.67,  # ~1000/minute
        "max_concurrent_requests": 20
    },
    "premium": {
        "requests_per_minute": 5000,
        "requests_per_hour": 50000,
        "requests_per_day": 500000,
        "burst_capacity": 1000,
        "burst_refill_rate": 83.33,  # ~5000/minute
        "max_concurrent_requests": 50
    },
    "enterprise": {
        "requests_per_minute": 20000,
        "requests_per_hour": 200000,
        "requests_per_day": 2000000,
        "burst_capacity": 5000,
        "burst_refill_rate": 333.33,  # ~20000/minute
        "max_concurrent_requests": 200
    }
}

# Export classes
__all__ = [
    "RateLimitWindow",
    "TokenBucket",
    "RateLimitTier",
    "ConcurrentRequestTracker",
    "RateLimitViolation",
    "DEFAULT_RATE_LIMIT_TIERS"
]