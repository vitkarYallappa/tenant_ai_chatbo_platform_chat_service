# src/models/redis/__init__.py
"""
Redis cache models for the Chat Service.
Provides caching structures for sessions, rate limiting, and conversation state.
"""

from src.models.redis.session_cache import (
    SessionCache,
    ConversationState,
    ActiveConversations,
    SessionCacheManager
)

from src.models.redis.rate_limit_cache import (
    RateLimitWindow,
    TokenBucket,
    RateLimitTier,
    ConcurrentRequestTracker,
    RateLimitViolation,
    DEFAULT_RATE_LIMIT_TIERS
)

__all__ = [
    # Session and conversation state
    "SessionCache",
    "ConversationState",
    "ActiveConversations",
    "SessionCacheManager",

    # Rate limiting
    "RateLimitWindow",
    "TokenBucket",
    "RateLimitTier",
    "ConcurrentRequestTracker",
    "RateLimitViolation",
    "DEFAULT_RATE_LIMIT_TIERS"
]