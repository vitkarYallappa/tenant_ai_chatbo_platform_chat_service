"""
Application configuration settings using Pydantic Settings.

This module provides centralized configuration management with
environment variable support, validation, and type safety.
"""

import secrets
from typing import List, Optional, Dict, Any
from enum import Enum
from functools import lru_cache

from pydantic import Field, field_validator, model_validator, FieldValidationInfo
from pydantic_settings import BaseSettings
from pydantic.networks import AnyHttpUrl, PostgresDsn, RedisDsn


class Environment(str, Enum):
    """Application environment options."""
    DEVELOPMENT = "development"
    TESTING = "testing"
    STAGING = "staging"
    PRODUCTION = "production"


class LogLevel(str, Enum):
    """Logging level options."""
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"


class Settings(BaseSettings):
    """Application settings with environment variable support."""

    # Environment Configuration
    ENVIRONMENT: Environment = Field(
        default=Environment.DEVELOPMENT,
        description="Application environment"
    )
    DEBUG: bool = Field(
        default=False,
        description="Enable debug mode"
    )
    TESTING: bool = Field(
        default=False,
        description="Enable testing mode"
    )

    # Server Configuration
    HOST: str = Field(
        default="0.0.0.0",
        description="Server host address"
    )
    PORT: int = Field(
        default=8001,
        ge=1,
        le=65535,
        description="Server port number"
    )
    WORKERS: int = Field(
        default=1,
        ge=1,
        le=32,
        description="Number of worker processes"
    )

    # Logging Configuration
    LOG_LEVEL: LogLevel = Field(
        default=LogLevel.INFO,
        description="Logging level"
    )
    LOG_FORMAT: str = Field(
        default="json",
        pattern=r"^(json|text)$",
        description="Log output format"
    )

    # Database Configuration
    MONGODB_URI: str = Field(
        default="mongodb://localhost:27017",
        description="MongoDB connection URI"
    )
    MONGODB_DATABASE: str = Field(
        default="chatbot_conversations",
        min_length=1,
        max_length=64,
        description="MongoDB database name"
    )
    MONGODB_MAX_CONNECTIONS: int = Field(
        default=100,
        ge=1,
        le=1000,
        description="MongoDB maximum connections"
    )
    MONGODB_MIN_CONNECTIONS: int = Field(
        default=10,
        ge=1,
        le=100,
        description="MongoDB minimum connections"
    )

    # Redis Configuration
    REDIS_URL: RedisDsn = Field(
        default="redis://localhost:6379",
        description="Redis connection URL"
    )
    REDIS_DB: int = Field(
        default=0,
        ge=0,
        le=15,
        description="Redis database number"
    )
    REDIS_MAX_CONNECTIONS: int = Field(
        default=50,
        ge=1,
        le=500,
        description="Redis maximum connections"
    )
    REDIS_SOCKET_TIMEOUT: int = Field(
        default=5,
        ge=1,
        le=60,
        description="Redis socket timeout in seconds"
    )

    # PostgreSQL Configuration
    POSTGRESQL_URI: PostgresDsn = Field(
        default="postgresql://postgres:postgres@localhost:5432/chatbot_config",
        description="PostgreSQL connection URI"
    )
    POSTGRESQL_MAX_CONNECTIONS: int = Field(
        default=20,
        ge=1,
        le=100,
        description="PostgreSQL maximum connections"
    )
    POSTGRESQL_MIN_CONNECTIONS: int = Field(
        default=5,
        ge=1,
        le=50,
        description="PostgreSQL minimum connections"
    )

    # Message Queue Configuration
    KAFKA_BROKERS: List[str] = Field(
        default=["localhost:9092"],
        description="Kafka broker addresses"
    )
    KAFKA_TOPIC_PREFIX: str = Field(
        default="chatbot.platform",
        min_length=1,
        max_length=100,
        description="Kafka topic prefix"
    )
    KAFKA_CONSUMER_GROUP: str = Field(
        default="chat-service",
        min_length=1,
        max_length=100,
        description="Kafka consumer group ID"
    )
    KAFKA_BATCH_SIZE: int = Field(
        default=100,
        ge=1,
        le=10000,
        description="Kafka batch size"
    )

    # External Services Configuration
    MCP_ENGINE_URL: str = Field(
        default="localhost:50051",
        description="MCP Engine gRPC endpoint"
    )
    SECURITY_HUB_URL: str = Field(
        default="localhost:50052",
        description="Security Hub gRPC endpoint"
    )
    ANALYTICS_ENGINE_URL: Optional[str] = Field(
        default=None,
        description="Analytics Engine HTTP endpoint"
    )

    # Security Configuration
    JWT_SECRET_KEY: str = Field(
        default_factory=lambda: secrets.token_urlsafe(32),
        min_length=32,
        description="JWT signing secret key"
    )
    JWT_ALGORITHM: str = Field(
        default="HS256",
        pattern=r"^(HS256|HS384|HS512|RS256|RS384|RS512)$",
        description="JWT signing algorithm"
    )
    JWT_EXPIRE_MINUTES: int = Field(
        default=60,
        ge=1,
        le=10080,  # 1 week
        description="JWT token expiration in minutes"
    )
    API_KEY_LENGTH: int = Field(
        default=32,
        ge=16,
        le=64,
        description="API key length in bytes"
    )

    # CORS Configuration
    ALLOWED_ORIGINS: List[str] = Field(
        default=["*"],
        description="CORS allowed origins"
    )
    ALLOWED_HOSTS: List[str] = Field(
        default=["*"],
        description="Trusted host middleware allowed hosts"
    )

    # Performance Configuration
    REQUEST_TIMEOUT_MS: int = Field(
        default=30000,
        ge=1000,
        le=300000,  # 5 minutes
        description="Request timeout in milliseconds"
    )
    MAX_REQUEST_SIZE_MB: int = Field(
        default=16,
        ge=1,
        le=100,
        description="Maximum request size in MB"
    )
    MAX_UPLOAD_SIZE_MB: int = Field(
        default=50,
        ge=1,
        le=500,
        description="Maximum file upload size in MB"
    )

    # Rate Limiting Configuration
    RATE_LIMIT_ENABLED: bool = Field(
        default=True,
        description="Enable rate limiting"
    )
    DEFAULT_RATE_LIMIT_PER_MINUTE: int = Field(
        default=1000,
        ge=1,
        le=100000,
        description="Default rate limit per minute"
    )
    BURST_RATE_MULTIPLIER: float = Field(
        default=2.0,
        ge=1.0,
        le=10.0,
        description="Burst rate multiplier for rate limiting"
    )

    # Caching Configuration
    CACHE_ENABLED: bool = Field(
        default=True,
        description="Enable caching"
    )
    CACHE_DEFAULT_TTL: int = Field(
        default=300,
        ge=1,
        le=86400,  # 24 hours
        description="Default cache TTL in seconds"
    )
    CACHE_MAX_SIZE_MB: int = Field(
        default=100,
        ge=1,
        le=1000,
        description="Maximum cache size in MB"
    )

    # Monitoring Configuration
    METRICS_ENABLED: bool = Field(
        default=True,
        description="Enable metrics collection"
    )
    TRACING_ENABLED: bool = Field(
        default=False,
        description="Enable distributed tracing"
    )
    JAEGER_ENDPOINT: Optional[str] = Field(
        default=None,
        description="Jaeger tracing endpoint"
    )

    # Feature Flags
    ENABLE_API_DOCS: bool = Field(
        default=True,
        description="Enable API documentation endpoints"
    )
    ENABLE_ADMIN_ENDPOINTS: bool = Field(
        default=False,
        description="Enable admin endpoints"
    )
    ENABLE_WEBHOOKS: bool = Field(
        default=True,
        description="Enable webhook functionality"
    )

    # Development Configuration
    RELOAD_ON_CHANGE: bool = Field(
        default=False,
        description="Enable auto-reload on file changes"
    )
    MOCK_EXTERNAL_SERVICES: bool = Field(
        default=False,
        description="Use mock external services for testing"
    )

    class Config:
        """Pydantic configuration."""
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = True
        validate_assignment = True
        extra = "forbid"  # Forbid extra fields

        # Custom configuration for field descriptions
        json_schema_extra = {
            "example": {
                "ENVIRONMENT": "development",
                "HOST": "0.0.0.0",
                "PORT": 8001,
                "MONGODB_URI": "mongodb://localhost:27017",
                "REDIS_URL": "redis://localhost:6379",
                "LOG_LEVEL": "INFO"
            }
        }

    @field_validator("KAFKA_BROKERS")
    @classmethod
    def validate_kafka_brokers(cls, v):
        """Validate Kafka broker format."""
        for broker in v:
            if not broker or ":" not in broker:
                raise ValueError(f"Invalid Kafka broker format: {broker}")
        return v

    @model_validator(mode='after')
    def validate_environment_consistency(self):
        """Validate environment-specific consistency."""
        # Production-specific validations
        if self.ENVIRONMENT == Environment.PRODUCTION:
            if self.DEBUG:
                raise ValueError("Debug mode should not be enabled in production")

            if "*" in self.ALLOWED_ORIGINS:
                raise ValueError("Wildcard CORS origins not allowed in production")

            if len(self.JWT_SECRET_KEY) < 32:
                raise ValueError("JWT secret key must be at least 32 characters in production")

            if self.JWT_SECRET_KEY == "dev-secret-change-in-production":
                raise ValueError("Default JWT secret key not allowed in production")

            if self.MOCK_EXTERNAL_SERVICES:
                raise ValueError("Mock external services not allowed in production")

            if self.RELOAD_ON_CHANGE:
                raise ValueError("Auto-reload not allowed in production")

            if not self.CACHE_ENABLED:
                raise ValueError("Caching should be enabled in production")

        # Testing-specific settings
        elif self.ENVIRONMENT == Environment.TESTING:
            self.TESTING = True
            self.MOCK_EXTERNAL_SERVICES = True

        # Connection limit validations
        if self.MONGODB_MIN_CONNECTIONS > self.MONGODB_MAX_CONNECTIONS:
            raise ValueError("MongoDB min connections cannot exceed max connections")

        if self.POSTGRESQL_MIN_CONNECTIONS > self.POSTGRESQL_MAX_CONNECTIONS:
            raise ValueError("PostgreSQL min connections cannot exceed max connections")

        return self

    def get_database_url(self, db_type: str) -> str:
        """Get database URL by type."""
        urls = {
            "mongodb": self.MONGODB_URI,
            "redis": str(self.REDIS_URL),
            "postgresql": str(self.POSTGRESQL_URI),
        }
        return urls.get(db_type, "")

    def is_production(self) -> bool:
        """Check if running in production environment."""
        return self.ENVIRONMENT == Environment.PRODUCTION

    def is_development(self) -> bool:
        """Check if running in development environment."""
        return self.ENVIRONMENT == Environment.DEVELOPMENT

    def is_testing(self) -> bool:
        """Check if running in testing environment."""
        return self.ENVIRONMENT == Environment.TESTING or self.TESTING

    def get_kafka_topic(self, topic_name: str) -> str:
        """Get full Kafka topic name with prefix."""
        return f"{self.KAFKA_TOPIC_PREFIX}.{topic_name}"

    def get_redis_key(self, key: str, tenant_id: Optional[str] = None) -> str:
        """Get Redis key with optional tenant prefix."""
        if tenant_id:
            return f"tenant:{tenant_id}:{key}"
        return key


# Global settings instance with caching
_settings: Optional[Settings] = None


@lru_cache()
def get_settings() -> Settings:
    """
    Get application settings (cached singleton).

    Returns:
        Settings: Configured application settings instance

    Note:
        Settings are cached using functools.lru_cache to avoid
        re-parsing environment variables on every call.
    """
    global _settings
    if _settings is None:
        _settings = Settings()
    return _settings


def reload_settings() -> Settings:
    """
    Force reload of application settings.

    Returns:
        Settings: New settings instance

    Note:
        This clears the cache and creates a new settings instance.
        Useful for testing or dynamic configuration updates.
    """
    global _settings
    _settings = None
    get_settings.cache_clear()
    return get_settings()