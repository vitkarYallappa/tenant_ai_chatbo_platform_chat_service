# src/config/redis_config.py
"""
Redis-specific configuration settings.
Provides configuration for Redis connections, caching, and rate limiting.
"""

from typing import Optional, Dict, Any, List
from pydantic import Field, validator
from pydantic_settings import BaseSettings
import os


class RedisMasterConfig(BaseSettings):
    """Redis connection and caching configuration."""

    # Connection settings
    REDIS_URL: str = Field(
        default="redis://localhost:6379",
        description="Redis connection URL"
    )
    REDIS_HOST: str = Field(
        default="localhost",
        description="Redis host"
    )
    REDIS_PORT: int = Field(
        default=6379,
        ge=1,
        le=65535,
        description="Redis port"
    )
    REDIS_DB: int = Field(
        default=0,
        ge=0,
        le=15,
        description="Redis database number"
    )
    REDIS_PASSWORD: Optional[str] = Field(
        default=None,
        description="Redis password"
    )

    # Connection pool settings
    MAX_CONNECTIONS_REDIS: int = Field(
        default=50,
        ge=1,
        le=1000,
        description="Maximum connections in pool"
    )
    MIN_CONNECTIONS_REDIS: int = Field(
        default=5,
        ge=1,
        description="Minimum connections in pool"
    )

    # Timeout settings (in seconds)
    SOCKET_TIMEOUT: int = Field(
        default=10,
        ge=1,
        le=60,
        description="Socket timeout"
    )
    SOCKET_CONNECT_TIMEOUT: int = Field(
        default=10,
        ge=1,
        le=60,
        description="Connection timeout"
    )

    # Retry settings
    RETRY_ON_TIMEOUT: bool = Field(
        default=True,
        description="Retry operations on timeout"
    )
    MAX_RETRY_ATTEMPTS: int = Field(
        default=3,
        ge=1,
        le=10,
        description="Maximum retry attempts"
    )

    # Encoding settings
    DECODE_RESPONSES: bool = Field(
        default=True,
        description="Automatically decode responses"
    )
    ENCODING: str = Field(
        default="utf-8",
        description="Character encoding"
    )

    # Health check settings
    HEALTH_CHECK_INTERVAL: int = Field(
        default=30,
        ge=10,
        le=300,
        description="Health check interval in seconds"
    )
    PING_TIMEOUT: int = Field(
        default=1,
        ge=1,
        le=10,
        description="Ping timeout for health checks"
    )

    # Performance settings
    ENABLE_COMPRESSION: bool = Field(
        default=False,
        description="Enable data compression"
    )
    COMPRESSION_THRESHOLD: int = Field(
        default=1024,
        ge=100,
        description="Minimum size for compression (bytes)"
    )

    @validator('REDIS_URL')
    def validate_redis_url(cls, v):
        if not v.startswith(('redis://', 'rediss://')):
            raise ValueError('Redis URL must start with redis:// or rediss://')
        return v

    @validator('ENCODING')
    def validate_encoding(cls, v):
        try:
            "test".encode(v)
            return v
        except LookupError:
            raise ValueError(f'Invalid encoding: {v}')

    class Config:
        env_prefix = "REDIS_"
        case_sensitive = True


class RedisCacheConfig(BaseSettings):
    """Redis caching configuration."""

    # Default TTL settings (in seconds)
    DEFAULT_TTL: int = Field(
        default=3600,
        ge=60,
        le=86400,
        description="Default cache TTL"
    )
    SESSION_TTL: int = Field(
        default=3600,
        ge=300,
        le=86400,
        description="Session cache TTL"
    )
    CONVERSATION_STATE_TTL: int = Field(
        default=86400,
        ge=3600,
        le=604800,
        description="Conversation state TTL"
    )
    RATE_LIMIT_TTL: int = Field(
        default=3600,
        ge=60,
        le=86400,
        description="Rate limit window TTL"
    )

    # Cache key prefixes
    SESSION_PREFIX: str = Field(
        default="session",
        description="Session cache key prefix"
    )
    CONVERSATION_PREFIX: str = Field(
        default="conversation_state",
        description="Conversation state key prefix"
    )
    RATE_LIMIT_PREFIX: str = Field(
        default="rate_limit",
        description="Rate limit key prefix"
    )
    LOCK_PREFIX: str = Field(
        default="lock",
        description="Distributed lock key prefix"
    )

    # Cache policies
    EVICTION_POLICY: str = Field(
        default="allkeys-lru",
        description="Redis eviction policy"
    )
    MAX_MEMORY_POLICY: str = Field(
        default="allkeys-lru",
        description="Max memory policy"
    )

    # Serialization settings
    SERIALIZE_JSON: bool = Field(
        default=True,
        description="Use JSON serialization"
    )
    COMPRESS_VALUES: bool = Field(
        default=False,
        description="Compress cached values"
    )

    @validator('EVICTION_POLICY')
    def validate_eviction_policy(cls, v):
        valid_policies = [
            'noeviction', 'allkeys-lru', 'volatile-lru',
            'allkeys-random', 'volatile-random', 'volatile-ttl'
        ]
        if v not in valid_policies:
            raise ValueError(f'Eviction policy must be one of: {valid_policies}')
        return v

    class Config:
        env_prefix = "REDIS_CACHE_"
        case_sensitive = True


class RedisRateLimitConfig(BaseSettings):
    """Redis rate limiting configuration."""

    # Default rate limits
    DEFAULT_RATE_LIMIT_PER_MINUTE: int = Field(
        default=1000,
        ge=1,
        description="Default rate limit per minute"
    )
    DEFAULT_RATE_LIMIT_PER_HOUR: int = Field(
        default=10000,
        ge=1,
        description="Default rate limit per hour"
    )
    DEFAULT_RATE_LIMIT_PER_DAY: int = Field(
        default=100000,
        ge=1,
        description="Default rate limit per day"
    )

    # Token bucket settings
    DEFAULT_BUCKET_CAPACITY: int = Field(
        default=100,
        ge=1,
        description="Default token bucket capacity"
    )
    DEFAULT_REFILL_RATE: float = Field(
        default=1.67,
        gt=0,
        description="Default token refill rate per second"
    )

    # Rate limit windows
    SLIDING_WINDOW_SIZE: int = Field(
        default=60,
        ge=10,
        le=3600,
        description="Sliding window size in seconds"
    )
    CLEANUP_INTERVAL: int = Field(
        default=300,
        ge=60,
        le=3600,
        description="Rate limit cleanup interval in seconds"
    )

    # Concurrency limits
    DEFAULT_MAX_CONCURRENT: int = Field(
        default=10,
        ge=1,
        description="Default max concurrent requests"
    )
    CONCURRENT_TIMEOUT: int = Field(
        default=600,
        ge=60,
        le=3600,
        description="Concurrent request timeout in seconds"
    )

    # Violation tracking
    VIOLATION_THRESHOLD: int = Field(
        default=5,
        ge=1,
        description="Violations before warning"
    )
    BLOCK_DURATION_MINUTES: int = Field(
        default=15,
        ge=1,
        le=1440,
        description="Block duration for violations"
    )

    class Config:
        env_prefix = "REDIS_RATE_LIMIT_"
        case_sensitive = True


class RedisClusterConfig(BaseSettings):
    """Redis Cluster configuration."""

    # Cluster settings
    CLUSTER_ENABLED: bool = Field(
        default=False,
        description="Enable Redis Cluster mode"
    )
    CLUSTER_NODES: List[str] = Field(
        default=["localhost:7000", "localhost:7001", "localhost:7002"],
        description="Redis cluster node addresses"
    )

    # Cluster behavior
    SKIP_FULL_COVERAGE_CHECK: bool = Field(
        default=False,
        description="Skip full coverage check in cluster"
    )
    MAX_CONNECTIONS_PER_NODE: int = Field(
        default=50,
        ge=1,
        description="Max connections per cluster node"
    )

    # Failover settings
    RETRY_ON_CLUSTER_DOWN: bool = Field(
        default=True,
        description="Retry operations when cluster is down"
    )
    CLUSTER_DOWN_RETRY_ATTEMPTS: int = Field(
        default=3,
        ge=1,
        le=10,
        description="Retry attempts when cluster is down"
    )

    class Config:
        env_prefix = "REDIS_CLUSTER_"
        case_sensitive = True


class RedisConfig(BaseSettings):
    """Master Redis configuration."""

    # Environment and general settings
    ENVIRONMENT: str = Field(
        default="development",
        description="Application environment"
    )
    DEBUG_REDIS: bool = Field(
        default=False,
        description="Enable Redis debug logging"
    )

    # Base Redis configuration
    base: RedisConfig = Field(default_factory=RedisConfig)
    cache: RedisCacheConfig = Field(default_factory=RedisCacheConfig)
    rate_limit: RedisRateLimitConfig = Field(default_factory=RedisRateLimitConfig)
    cluster: RedisClusterConfig = Field(default_factory=RedisClusterConfig)

    # Monitoring and alerting
    ENABLE_MONITORING: bool = Field(
        default=True,
        description="Enable Redis monitoring"
    )
    SLOW_LOG_THRESHOLD_MS: int = Field(
        default=100,
        ge=1,
        description="Slow query threshold in milliseconds"
    )

    # Memory management
    MAX_MEMORY: Optional[str] = Field(
        default=None,
        description="Maximum memory limit (e.g., '1gb')"
    )
    MEMORY_POLICY: str = Field(
        default="allkeys-lru",
        description="Memory eviction policy"
    )

    # Persistence settings
    ENABLE_AOF: bool = Field(
        default=True,
        description="Enable Append Only File persistence"
    )
    AOF_SYNC: str = Field(
        default="everysec",
        description="AOF sync policy"
    )
    ENABLE_RDB: bool = Field(
        default=True,
        description="Enable RDB snapshots"
    )
    RDB_SAVE_CONFIG: List[str] = Field(
        default=["900 1", "300 10", "60 10000"],
        description="RDB save configuration"
    )

    @validator('ENVIRONMENT')
    def validate_environment(cls, v):
        valid_envs = ['development', 'testing', 'staging', 'production']
        if v not in valid_envs:
            raise ValueError(f'Environment must be one of: {valid_envs}')
        return v

    @validator('MEMORY_POLICY')
    def validate_memory_policy(cls, v):
        valid_policies = [
            'noeviction', 'allkeys-lru', 'volatile-lru',
            'allkeys-random', 'volatile-random', 'volatile-ttl'
        ]
        if v not in valid_policies:
            raise ValueError(f'Memory policy must be one of: {valid_policies}')
        return v

    @validator('AOF_SYNC')
    def validate_aof_sync(cls, v):
        valid_sync = ['always', 'everysec', 'no']
        if v not in valid_sync:
            raise ValueError(f'AOF sync must be one of: {valid_sync}')
        return v

    def is_production(self) -> bool:
        """Check if running in production environment."""
        return self.ENVIRONMENT == "production"

    def is_development(self) -> bool:
        """Check if running in development environment."""
        return self.ENVIRONMENT == "development"

    def get_redis_url(self) -> str:
        """Get Redis connection URL."""
        return self.base.REDIS_URL

    def get_connection_settings(self) -> Dict[str, Any]:
        """Get Redis connection settings as dictionary."""
        return {
            "host": self.base.REDIS_HOST,
            "port": self.base.REDIS_PORT,
            "db": self.base.REDIS_DB,
            "password": self.base.REDIS_PASSWORD,
            "max_connections": self.base.MAX_CONNECTIONS_REDIS,
            "socket_timeout": self.base.SOCKET_TIMEOUT,
            "socket_connect_timeout": self.base.SOCKET_CONNECT_TIMEOUT,
            "retry_on_timeout": self.base.RETRY_ON_TIMEOUT,
            "decode_responses": self.base.DECODE_RESPONSES,
            "encoding": self.base.ENCODING
        }

    def get_cache_settings(self) -> Dict[str, Any]:
        """Get cache configuration as dictionary."""
        return {
            "default_ttl": self.cache.DEFAULT_TTL,
            "session_ttl": self.cache.SESSION_TTL,
            "conversation_state_ttl": self.cache.CONVERSATION_STATE_TTL,
            "rate_limit_ttl": self.cache.RATE_LIMIT_TTL,
            "session_prefix": self.cache.SESSION_PREFIX,
            "conversation_prefix": self.cache.CONVERSATION_PREFIX,
            "rate_limit_prefix": self.cache.RATE_LIMIT_PREFIX,
            "lock_prefix": self.cache.LOCK_PREFIX
        }

    def get_cluster_settings(self) -> Dict[str, Any]:
        """Get cluster configuration as dictionary."""
        return {
            "enabled": self.cluster.CLUSTER_ENABLED,
            "nodes": self.cluster.CLUSTER_NODES,
            "skip_full_coverage_check": self.cluster.SKIP_FULL_COVERAGE_CHECK,
            "max_connections_per_node": self.cluster.MAX_CONNECTIONS_PER_NODE,
            "retry_on_cluster_down": self.cluster.RETRY_ON_CLUSTER_DOWN,
            "cluster_down_retry_attempts": self.cluster.CLUSTER_DOWN_RETRY_ATTEMPTS
        }

    class Config:
        env_prefix = "REDIS_"
        case_sensitive = True
        env_file = ".env"
        env_file_encoding = "utf-8"


# Global configuration instances
def get_redis_config() -> RedisMasterConfig:
    """
    Get Redis configuration instance.

    Returns:
        RedisMasterConfig instance with all settings
    """
    return RedisMasterConfig()


# Environment-specific configurations
def get_development_redis_config() -> RedisMasterConfig:
    """Get development Redis configuration."""
    config = RedisMasterConfig()
    config.ENVIRONMENT = "development"
    config.DEBUG_REDIS = True
    config.base.REDIS_DB = 0
    config.ENABLE_MONITORING = True
    return config


def get_production_redis_config() -> RedisMasterConfig:
    """Get production Redis configuration."""
    config = RedisMasterConfig()
    config.ENVIRONMENT = "production"
    config.DEBUG_REDIS = False
    config.ENABLE_AOF = True
    config.ENABLE_RDB = True
    config.MAX_MEMORY = "2gb"
    config.cluster.CLUSTER_ENABLED = True
    return config


def get_testing_redis_config() -> RedisMasterConfig:
    """Get testing Redis configuration."""
    config = RedisMasterConfig()
    config.ENVIRONMENT = "testing"
    config.DEBUG_REDIS = True
    config.base.REDIS_DB = 1  # Use different DB for testing
    config.cache.DEFAULT_TTL = 60  # Shorter TTL for tests
    config.ENABLE_AOF = False
    config.ENABLE_RDB = False
    return config


# Configuration validation
def validate_redis_config(config: RedisMasterConfig) -> List[str]:
    """
    Validate Redis configuration.

    Args:
        config: RedisMasterConfig instance to validate

    Returns:
        List of validation errors (empty if valid)
    """
    errors = []

    # Check Redis URL
    if not config.base.REDIS_URL:
        errors.append("Redis URL is required")

    # Check connection pool settings
    if config.base.MAX_CONNECTIONS_REDIS < config.base.MIN_CONNECTIONS_REDIS:
        errors.append("Max connections must be >= min connections")

    # Check timeout settings
    if config.base.SOCKET_TIMEOUT < 1:
        errors.append("Socket timeout must be at least 1 second")

    # Check cache TTL settings
    if config.cache.SESSION_TTL < 300:
        errors.append("Session TTL should be at least 5 minutes")

    # Cluster-specific validations
    if config.cluster.CLUSTER_ENABLED:
        if len(config.cluster.CLUSTER_NODES) < 3:
            errors.append("Cluster requires at least 3 nodes")

        # Validate node addresses
        for node in config.cluster.CLUSTER_NODES:
            if ':' not in node:
                errors.append(f"Invalid cluster node address: {node}")

    # Production-specific validations
    if config.is_production():
        if config.DEBUG_REDIS:
            errors.append("Debug mode should be disabled in production")
        if not config.base.REDIS_PASSWORD:
            errors.append("Password should be set in production")
        if not config.ENABLE_AOF and not config.ENABLE_RDB:
            errors.append("At least one persistence method should be enabled in production")

    return errors


# Default rate limit tiers for Redis configuration
DEFAULT_REDIS_RATE_LIMIT_TIERS = {
    "basic": {
        "requests_per_minute": 100,
        "requests_per_hour": 1000,
        "requests_per_day": 10000,
        "bucket_capacity": 50,
        "refill_rate": 1.67,
        "max_concurrent": 5
    },
    "standard": {
        "requests_per_minute": 1000,
        "requests_per_hour": 10000,
        "requests_per_day": 100000,
        "bucket_capacity": 200,
        "refill_rate": 16.67,
        "max_concurrent": 20
    },
    "premium": {
        "requests_per_minute": 5000,
        "requests_per_hour": 50000,
        "requests_per_day": 500000,
        "bucket_capacity": 1000,
        "refill_rate": 83.33,
        "max_concurrent": 50
    },
    "enterprise": {
        "requests_per_minute": 20000,
        "requests_per_hour": 200000,
        "requests_per_day": 2000000,
        "bucket_capacity": 5000,
        "refill_rate": 333.33,
        "max_concurrent": 200
    }
}

# Export configuration classes and functions
__all__ = [
    "RedisConfig",
    "RedisCacheConfig",
    "RedisRateLimitConfig",
    "RedisClusterConfig",
    "RedisMasterConfig",
    "get_redis_config",
    "get_development_redis_config",
    "get_production_redis_config",
    "get_testing_redis_config",
    "validate_redis_config",
    "DEFAULT_REDIS_RATE_LIMIT_TIERS"
]