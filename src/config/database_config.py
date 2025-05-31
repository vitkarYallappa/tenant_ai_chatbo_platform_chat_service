# src/config/database_config.py
"""
Database configuration settings.
Provides configuration for MongoDB, Redis, and other database connections.
"""

from typing import Optional, Dict, Any, List
from pydantic import  Field, validator
from pydantic_settings import BaseSettings
import os


class MongoDBConfig(BaseSettings):
    """MongoDB configuration settings."""

    # Connection settings
    MONGODB_URI: str = Field(
        default="mongodb://localhost:27017",
        description="MongoDB connection URI"
    )
    MONGODB_DATABASE: str = Field(
        default="chatbot_platform",
        description="Database name"
    )

    # Connection pool settings
    MAX_CONNECTIONS_MONGO: int = Field(
        default=100,
        ge=1,
        le=1000,
        description="Maximum connections in pool"
    )
    MIN_CONNECTIONS_MONGO: int = Field(
        default=10,
        ge=1,
        description="Minimum connections in pool"
    )

    # Timeout settings (in milliseconds)
    SERVER_SELECTION_TIMEOUT_MS: int = Field(
        default=5000,
        ge=1000,
        le=30000,
        description="Server selection timeout"
    )
    CONNECT_TIMEOUT_MS: int = Field(
        default=10000,
        ge=1000,
        le=60000,
        description="Connection timeout"
    )
    SOCKET_TIMEOUT_MS: int = Field(
        default=20000,
        ge=1000,
        le=120000,
        description="Socket timeout"
    )

    # Write concern settings
    WRITE_CONCERN: str = Field(
        default="majority",
        description="Write concern level"
    )
    READ_PREFERENCE: str = Field(
        default="secondaryPreferred",
        description="Read preference"
    )
    READ_CONCERN: str = Field(
        default="majority",
        description="Read concern level"
    )

    # Retry settings
    RETRY_WRITES: bool = Field(
        default=True,
        description="Enable retry writes"
    )
    RETRY_READS: bool = Field(
        default=True,
        description="Enable retry reads"
    )

    # Monitoring and health check settings
    HEARTBEAT_FREQUENCY_MS: int = Field(
        default=10000,
        ge=500,
        le=60000,
        description="Heartbeat frequency"
    )
    HEALTH_CHECK_INTERVAL_SECONDS: int = Field(
        default=30,
        ge=10,
        le=300,
        description="Health check interval"
    )

    # Index management
    AUTO_CREATE_INDEXES: bool = Field(
        default=True,
        description="Automatically create indexes on startup"
    )
    INDEX_CREATION_TIMEOUT_SECONDS: int = Field(
        default=300,
        ge=30,
        le=1800,
        description="Timeout for index creation"
    )

    # Backup and maintenance
    BACKUP_ENABLED: bool = Field(
        default=False,
        description="Enable automated backups"
    )
    BACKUP_INTERVAL_HOURS: int = Field(
        default=24,
        ge=1,
        le=168,
        description="Backup interval in hours"
    )
    BACKUP_RETENTION_DAYS: int = Field(
        default=30,
        ge=1,
        le=365,
        description="Backup retention period"
    )

    # Data cleanup settings
    CLEANUP_ENABLED: bool = Field(
        default=True,
        description="Enable automatic data cleanup"
    )
    CLEANUP_INTERVAL_HOURS: int = Field(
        default=6,
        ge=1,
        le=24,
        description="Cleanup interval in hours"
    )
    MESSAGE_RETENTION_DAYS: int = Field(
        default=90,
        ge=1,
        le=365,
        description="Default message retention period"
    )
    SESSION_RETENTION_DAYS: int = Field(
        default=30,
        ge=1,
        le=90,
        description="Default session retention period"
    )

    @validator('MONGODB_URI')
    def validate_mongodb_uri(cls, v):
        if not v.startswith(('mongodb://', 'mongodb+srv://')):
            raise ValueError('MongoDB URI must start with mongodb:// or mongodb+srv://')
        return v

    @validator('WRITE_CONCERN')
    def validate_write_concern(cls, v):
        valid_concerns = ['majority', 'acknowledged', 'unacknowledged', '1', '2', '3']
        if v not in valid_concerns:
            raise ValueError(f'Write concern must be one of: {valid_concerns}')
        return v

    @validator('READ_PREFERENCE')
    def validate_read_preference(cls, v):
        valid_prefs = ['primary', 'primaryPreferred', 'secondary', 'secondaryPreferred', 'nearest']
        if v not in valid_prefs:
            raise ValueError(f'Read preference must be one of: {valid_prefs}')
        return v

    class Config:
        env_prefix = "MONGO_"
        case_sensitive = True


class PostgreSQLConfig(BaseSettings):
    """PostgreSQL configuration settings (for future use)."""

    # Connection settings
    POSTGRESQL_URI: str = Field(
        default="postgresql://localhost:5432/chatbot_platform",
        description="PostgreSQL connection URI"
    )
    POSTGRESQL_DATABASE: str = Field(
        default="chatbot_platform",
        description="Database name"
    )

    # Connection pool settings
    MAX_CONNECTIONS_POSTGRES: int = Field(
        default=20,
        ge=1,
        le=100,
        description="Maximum connections in pool"
    )
    MIN_CONNECTIONS_POSTGRES: int = Field(
        default=5,
        ge=1,
        description="Minimum connections in pool"
    )

    # Timeout settings
    CONNECT_TIMEOUT_SECONDS: int = Field(
        default=30,
        ge=5,
        le=120,
        description="Connection timeout"
    )
    QUERY_TIMEOUT_SECONDS: int = Field(
        default=30,
        ge=5,
        le=300,
        description="Query timeout"
    )

    # SSL settings
    SSL_MODE: str = Field(
        default="prefer",
        description="SSL mode"
    )
    SSL_CERT_PATH: Optional[str] = Field(
        default=None,
        description="Path to SSL certificate"
    )

    @validator('POSTGRESQL_URI')
    def validate_postgresql_uri(cls, v):
        if not v.startswith(('postgresql://', 'postgres://')):
            raise ValueError('PostgreSQL URI must start with postgresql:// or postgres://')
        return v

    @validator('SSL_MODE')
    def validate_ssl_mode(cls, v):
        valid_modes = ['disable', 'allow', 'prefer', 'require', 'verify-ca', 'verify-full']
        if v not in valid_modes:
            raise ValueError(f'SSL mode must be one of: {valid_modes}')
        return v

    class Config:
        env_prefix = "POSTGRES_"
        case_sensitive = True


class TimescaleDBConfig(BaseSettings):
    """TimescaleDB configuration settings (extends PostgreSQL)."""

    # Hypertable settings
    CHUNK_TIME_INTERVAL: str = Field(
        default="1 hour",
        description="Chunk time interval for hypertables"
    )
    COMPRESSION_ENABLED: bool = Field(
        default=True,
        description="Enable compression for old chunks"
    )
    COMPRESSION_AFTER: str = Field(
        default="7 days",
        description="Compress chunks older than this"
    )

    # Retention settings
    RETENTION_ENABLED: bool = Field(
        default=True,
        description="Enable automatic data retention"
    )
    RETENTION_POLICIES: Dict[str, str] = Field(
        default={
            "system_metrics": "90 days",
            "conversation_analytics": "2 years",
            "model_usage_analytics": "1 year",
            "custom_metrics": "6 months"
        },
        description="Retention policies for different data types"
    )

    class Config:
        env_prefix = "TIMESCALE_"
        case_sensitive = True


class DatabaseConfig(BaseSettings):
    """Master database configuration."""

    # Environment settings
    ENVIRONMENT: str = Field(
        default="development",
        description="Application environment"
    )
    DEBUG_MODE: bool = Field(
        default=False,
        description="Enable debug mode"
    )

    # Database configurations
    mongodb: MongoDBConfig = Field(default_factory=MongoDBConfig)
    postgresql: PostgreSQLConfig = Field(default_factory=PostgreSQLConfig)
    timescaledb: TimescaleDBConfig = Field(default_factory=TimescaleDBConfig)

    # General database settings
    CONNECTION_RETRY_ATTEMPTS: int = Field(
        default=3,
        ge=1,
        le=10,
        description="Number of connection retry attempts"
    )
    CONNECTION_RETRY_DELAY_SECONDS: int = Field(
        default=5,
        ge=1,
        le=60,
        description="Delay between connection retries"
    )

    # Health monitoring
    HEALTH_CHECK_ENABLED: bool = Field(
        default=True,
        description="Enable health check monitoring"
    )
    HEALTH_CHECK_INTERVAL_SECONDS: int = Field(
        default=60,
        ge=10,
        le=300,
        description="Health check interval"
    )
    HEALTH_CHECK_TIMEOUT_SECONDS: int = Field(
        default=10,
        ge=1,
        le=30,
        description="Health check timeout"
    )

    # Performance monitoring
    SLOW_QUERY_THRESHOLD_MS: int = Field(
        default=1000,
        ge=100,
        le=10000,
        description="Slow query threshold in milliseconds"
    )
    ENABLE_QUERY_LOGGING: bool = Field(
        default=False,
        description="Enable query logging"
    )

    # Security settings
    ENABLE_SSL: bool = Field(
        default=True,
        description="Enable SSL connections"
    )
    VERIFY_SSL_CERTIFICATES: bool = Field(
        default=True,
        description="Verify SSL certificates"
    )

    # Migration settings
    AUTO_MIGRATE: bool = Field(
        default=False,
        description="Run migrations automatically on startup"
    )
    MIGRATION_TIMEOUT_SECONDS: int = Field(
        default=600,
        ge=60,
        le=3600,
        description="Migration timeout"
    )

    @validator('ENVIRONMENT')
    def validate_environment(cls, v):
        valid_envs = ['development', 'testing', 'staging', 'production']
        if v not in valid_envs:
            raise ValueError(f'Environment must be one of: {valid_envs}')
        return v

    def is_production(self) -> bool:
        """Check if running in production environment."""
        return self.ENVIRONMENT == "production"

    def is_development(self) -> bool:
        """Check if running in development environment."""
        return self.ENVIRONMENT == "development"

    def get_mongodb_settings(self) -> Dict[str, Any]:
        """Get MongoDB connection settings as dictionary."""
        return {
            "uri": self.mongodb.MONGODB_URI,
            "database": self.mongodb.MONGODB_DATABASE,
            "max_pool_size": self.mongodb.MAX_CONNECTIONS_MONGO,
            "min_pool_size": self.mongodb.MIN_CONNECTIONS_MONGO,
            "server_selection_timeout_ms": self.mongodb.SERVER_SELECTION_TIMEOUT_MS,
            "connect_timeout_ms": self.mongodb.CONNECT_TIMEOUT_MS,
            "socket_timeout_ms": self.mongodb.SOCKET_TIMEOUT_MS,
            "retry_writes": self.mongodb.RETRY_WRITES,
            "retry_reads": self.mongodb.RETRY_READS,
            "w": self.mongodb.WRITE_CONCERN,
            "read_preference": self.mongodb.READ_PREFERENCE,
            "read_concern": self.mongodb.READ_CONCERN
        }

    def get_postgresql_settings(self) -> Dict[str, Any]:
        """Get PostgreSQL connection settings as dictionary."""
        return {
            "uri": self.postgresql.POSTGRESQL_URI,
            "database": self.postgresql.POSTGRESQL_DATABASE,
            "max_connections": self.postgresql.MAX_CONNECTIONS_POSTGRES,
            "min_connections": self.postgresql.MIN_CONNECTIONS_POSTGRES,
            "connect_timeout": self.postgresql.CONNECT_TIMEOUT_SECONDS,
            "query_timeout": self.postgresql.QUERY_TIMEOUT_SECONDS,
            "ssl_mode": self.postgresql.SSL_MODE,
            "ssl_cert_path": self.postgresql.SSL_CERT_PATH
        }

    class Config:
        env_prefix = "DB_"
        case_sensitive = True
        env_file = ".env"
        env_file_encoding = "utf-8"


# Global configuration instance
def get_database_config() -> DatabaseConfig:
    """
    Get database configuration instance.

    Returns:
        DatabaseConfig instance with all settings
    """
    return DatabaseConfig()


# Environment-specific configurations
def get_development_config() -> DatabaseConfig:
    """Get development database configuration."""
    config = DatabaseConfig()
    config.ENVIRONMENT = "development"
    config.DEBUG_MODE = True
    config.mongodb.AUTO_CREATE_INDEXES = True
    config.ENABLE_QUERY_LOGGING = True
    return config


def get_production_config() -> DatabaseConfig:
    """Get production database configuration."""
    config = DatabaseConfig()
    config.ENVIRONMENT = "production"
    config.DEBUG_MODE = False
    config.ENABLE_SSL = True
    config.VERIFY_SSL_CERTIFICATES = True
    config.mongodb.BACKUP_ENABLED = True
    config.ENABLE_QUERY_LOGGING = False
    return config


def get_testing_config() -> DatabaseConfig:
    """Get testing database configuration."""
    config = DatabaseConfig()
    config.ENVIRONMENT = "testing"
    config.DEBUG_MODE = True
    config.mongodb.MONGODB_DATABASE = "chatbot_platform_test"
    config.postgresql.POSTGRESQL_DATABASE = "chatbot_platform_test"
    config.mongodb.AUTO_CREATE_INDEXES = True
    config.mongodb.CLEANUP_ENABLED = False
    return config


# Configuration validation
def validate_database_config(config: DatabaseConfig) -> List[str]:
    """
    Validate database configuration.

    Args:
        config: DatabaseConfig instance to validate

    Returns:
        List of validation errors (empty if valid)
    """
    errors = []

    # Check MongoDB URI accessibility
    if not config.mongodb.MONGODB_URI:
        errors.append("MongoDB URI is required")

    # Check connection pool settings
    if config.mongodb.MAX_CONNECTIONS_MONGO < config.mongodb.MIN_CONNECTIONS_MONGO:
        errors.append("Max connections must be >= min connections")

    # Check timeout settings
    if config.mongodb.CONNECT_TIMEOUT_MS < 1000:
        errors.append("Connect timeout should be at least 1000ms")

    # Production-specific validations
    if config.is_production():
        if config.DEBUG_MODE:
            errors.append("Debug mode should be disabled in production")
        if not config.ENABLE_SSL:
            errors.append("SSL should be enabled in production")
        if not config.mongodb.BACKUP_ENABLED:
            errors.append("Backups should be enabled in production")

    return errors


# Export configuration classes and functions
__all__ = [
    "MongoDBConfig",
    "PostgreSQLConfig",
    "TimescaleDBConfig",
    "DatabaseConfig",
    "get_database_config",
    "get_development_config",
    "get_production_config",
    "get_testing_config",
    "validate_database_config"
]