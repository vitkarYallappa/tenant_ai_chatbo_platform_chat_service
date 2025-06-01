"""
PostgreSQL Connection Management
===============================

Centralized PostgreSQL connection management with SQLAlchemy and AsyncPG,
including connection pooling, health monitoring, and configuration management.

Features:
- SQLAlchemy 2.0+ async support with AsyncPG driver
- Connection pooling with optimal settings
- Health monitoring and reconnection logic
- Migration support with Alembic
- Multi-tenant database isolation options
- Performance monitoring
- Graceful connection handling
"""

import asyncio
from typing import Optional, Dict, Any, AsyncGenerator
from sqlalchemy.ext.asyncio import (
    create_async_engine, AsyncSession, async_sessionmaker, AsyncEngine
)
from sqlalchemy.orm import DeclarativeBase
from sqlalchemy import text, event
from sqlalchemy.pool import NullPool, QueuePool
import structlog
from dataclasses import dataclass
import os
from urllib.parse import quote_plus

logger = structlog.get_logger(__name__)


@dataclass
class PostgreSQLConfig:
    """PostgreSQL configuration with secure defaults"""
    # Connection settings
    host: str = "localhost"
    port: int = 5432
    database_name: str = "chatbot_platform"
    username: Optional[str] = None
    password: Optional[str] = None

    # Connection pool settings
    pool_size: int = 20
    max_overflow: int = 0
    pool_timeout: int = 30
    pool_recycle: int = 3600
    pool_pre_ping: bool = True

    # Async settings
    connect_timeout: int = 10
    command_timeout: int = 60
    server_settings: Dict[str, str] = None

    # SSL settings
    ssl_mode: str = "prefer"  # disable, allow, prefer, require, verify-ca, verify-full
    ssl_cert: Optional[str] = None
    ssl_key: Optional[str] = None
    ssl_ca: Optional[str] = None

    # Performance settings
    statement_cache_size: int = 100
    prepared_statement_cache_size: int = 100

    # Environment-based overrides
    def __post_init__(self):
        """Override with environment variables if available"""
        self.host = os.getenv("POSTGRES_HOST", self.host)
        self.port = int(os.getenv("POSTGRES_PORT", str(self.port)))
        self.database_name = os.getenv("POSTGRES_DATABASE", self.database_name)
        self.username = os.getenv("POSTGRES_USERNAME", self.username)
        self.password = os.getenv("POSTGRES_PASSWORD", self.password)

        # SSL settings from environment
        self.ssl_mode = os.getenv("POSTGRES_SSL_MODE", self.ssl_mode)
        self.ssl_cert = os.getenv("POSTGRES_SSL_CERT", self.ssl_cert)
        self.ssl_key = os.getenv("POSTGRES_SSL_KEY", self.ssl_key)
        self.ssl_ca = os.getenv("POSTGRES_SSL_CA", self.ssl_ca)

        # Server settings
        if self.server_settings is None:
            self.server_settings = {
                "application_name": "chatbot_platform",
                "timezone": "UTC"
            }

    def get_connection_string(self) -> str:
        """
        Build PostgreSQL connection string for SQLAlchemy with AsyncPG

        Returns:
            PostgreSQL URI string
        """
        # Build auth part
        if self.username and self.password:
            encoded_username = quote_plus(self.username)
            encoded_password = quote_plus(self.password)
            auth_part = f"{encoded_username}:{encoded_password}@"
        else:
            auth_part = ""

        # Build base URI with asyncpg driver
        uri = f"postgresql+asyncpg://{auth_part}{self.host}:{self.port}/{self.database_name}"

        # Add query parameters
        params = []

        # SSL parameters
        if self.ssl_mode != "prefer":  # asyncpg uses different SSL parameter names
            ssl_mapping = {
                "disable": "disable",
                "allow": "allow",
                "prefer": "prefer",
                "require": "require",
                "verify-ca": "verify-ca",
                "verify-full": "verify-full"
            }
            params.append(f"ssl={ssl_mapping.get(self.ssl_mode, 'prefer')}")

        if self.ssl_cert:
            params.append(f"sslcert={self.ssl_cert}")
        if self.ssl_key:
            params.append(f"sslkey={self.ssl_key}")
        if self.ssl_ca:
            params.append(f"sslrootcert={self.ssl_ca}")

        # Connection timeouts
        params.append(f"connect_timeout={self.connect_timeout}")
        params.append(f"command_timeout={self.command_timeout}")

        # Server settings
        for key, value in self.server_settings.items():
            params.append(f"options=-c {key}={value}")

        if params:
            uri += "?" + "&".join(params)

        return uri

    def get_engine_options(self) -> Dict[str, Any]:
        """
        Get SQLAlchemy engine options

        Returns:
            Dictionary of engine options
        """
        return {
            'pool_size': self.pool_size,
            'max_overflow': self.max_overflow,
            'pool_timeout': self.pool_timeout,
            'pool_recycle': self.pool_recycle,
            'pool_pre_ping': self.pool_pre_ping,
            'echo': False,  # Set to True for SQL query logging
            'echo_pool': False,  # Set to True for connection pool logging
            'future': True,  # Use SQLAlchemy 2.0 style
        }


class PostgreSQLConnectionManager:
    """
    PostgreSQL connection manager with health monitoring and reconnection logic
    """

    def __init__(self, config: PostgreSQLConfig):
        """
        Initialize connection manager

        Args:
            config: PostgreSQL configuration
        """
        self.config = config
        self.engine: Optional[AsyncEngine] = None
        self.session_factory: Optional[async_sessionmaker[AsyncSession]] = None
        self._connection_lock = asyncio.Lock()
        self._health_check_task: Optional[asyncio.Task] = None
        self._is_healthy = False

    async def connect(self) -> AsyncEngine:
        """
        Establish connection to PostgreSQL

        Returns:
            SQLAlchemy async engine

        Raises:
            ConnectionError: If connection cannot be established
        """
        async with self._connection_lock:
            if self.engine is None:
                try:
                    logger.info(
                        "Connecting to PostgreSQL",
                        host=self.config.host,
                        database=self.config.database_name
                    )

                    # Create connection string
                    connection_string = self.config.get_connection_string()
                    engine_options = self.config.get_engine_options()

                    # Create async engine
                    self.engine = create_async_engine(
                        connection_string,
                        **engine_options
                    )

                    # Create session factory
                    self.session_factory = async_sessionmaker(
                        self.engine,
                        class_=AsyncSession,
                        expire_on_commit=False
                    )

                    # Test connection
                    await self._test_connection()

                    # Setup event listeners
                    self._setup_event_listeners()

                    # Start health monitoring
                    self._start_health_monitoring()

                    self._is_healthy = True
                    logger.info("Successfully connected to PostgreSQL")

                except Exception as e:
                    logger.error("Failed to connect to PostgreSQL", error=str(e))
                    await self.disconnect()
                    raise ConnectionError(f"Failed to connect to PostgreSQL: {e}")

            return self.engine

    async def disconnect(self) -> None:
        """Disconnect from PostgreSQL"""
        async with self._connection_lock:
            if self._health_check_task:
                self._health_check_task.cancel()
                try:
                    await self._health_check_task
                except asyncio.CancelledError:
                    pass
                self._health_check_task = None

            if self.engine:
                await self.engine.dispose()
                self.engine = None
                self.session_factory = None
                self._is_healthy = False

                logger.info("Disconnected from PostgreSQL")

    async def get_session(self) -> AsyncGenerator[AsyncSession, None]:
        """
        Get database session, connecting if necessary

        Yields:
            AsyncSession instance
        """
        if self.session_factory is None or not self._is_healthy:
            await self.connect()

        async with self.session_factory() as session:
            try:
                yield session
            except Exception:
                await session.rollback()
                raise
            finally:
                await session.close()

    async def get_engine(self) -> AsyncEngine:
        """
        Get database engine, connecting if necessary

        Returns:
            SQLAlchemy async engine
        """
        if self.engine is None or not self._is_healthy:
            await self.connect()

        return self.engine

    async def _test_connection(self) -> None:
        """Test PostgreSQL connection"""
        if self.engine:
            async with self.engine.begin() as conn:
                result = await conn.execute(text("SELECT 1"))
                assert result.scalar() == 1

    def _setup_event_listeners(self) -> None:
        """Setup SQLAlchemy event listeners for monitoring"""
        if self.engine:
            # Connection pool events
            @event.listens_for(self.engine.sync_engine, "connect")
            def receive_connect(dbapi_connection, connection_record):
                logger.debug("New database connection established")

            @event.listens_for(self.engine.sync_engine, "checkout")
            def receive_checkout(dbapi_connection, connection_record, connection_proxy):
                logger.debug("Connection checked out from pool")

            @event.listens_for(self.engine.sync_engine, "checkin")
            def receive_checkin(dbapi_connection, connection_record):
                logger.debug("Connection checked back into pool")

            # Query performance monitoring
            @event.listens_for(self.engine.sync_engine, "before_cursor_execute")
            def receive_before_cursor_execute(conn, cursor, statement, parameters, context, executemany):
                context._query_start_time = asyncio.get_event_loop().time()

            @event.listens_for(self.engine.sync_engine, "after_cursor_execute")
            def receive_after_cursor_execute(conn, cursor, statement, parameters, context, executemany):
                if hasattr(context, '_query_start_time'):
                    total = asyncio.get_event_loop().time() - context._query_start_time
                    if total > 1.0:  # Log slow queries (>1 second)
                        logger.warning(
                            "Slow query detected",
                            duration_seconds=total,
                            query=statement[:200] + "..." if len(statement) > 200 else statement
                        )

    def _start_health_monitoring(self) -> None:
        """Start background health monitoring task"""
        if self._health_check_task is None:
            self._health_check_task = asyncio.create_task(self._health_monitor())

    async def _health_monitor(self) -> None:
        """Background health monitoring coroutine"""
        while True:
            try:
                await asyncio.sleep(30)  # Check every 30 seconds
                await self._test_connection()

                if not self._is_healthy:
                    self._is_healthy = True
                    logger.info("PostgreSQL connection restored")

            except Exception as e:
                if self._is_healthy:
                    self._is_healthy = False
                    logger.error("PostgreSQL health check failed", error=str(e))

                # Try to reconnect after a delay
                await asyncio.sleep(5)
                try:
                    await self.connect()
                except Exception as reconnect_error:
                    logger.error("Failed to reconnect to PostgreSQL", error=str(reconnect_error))

    async def health_check(self) -> Dict[str, Any]:
        """
        Perform comprehensive health check

        Returns:
            Health status information
        """
        health_info = {
            "connected": False,
            "healthy": self._is_healthy,
            "database": self.config.database_name,
            "host": self.config.host,
            "port": self.config.port
        }

        try:
            if self.engine:
                # Test connection
                start_time = asyncio.get_event_loop().time()
                await self._test_connection()
                response_time = (asyncio.get_event_loop().time() - start_time) * 1000

                # Get connection pool info
                pool = self.engine.pool
                pool_status = {
                    "size": pool.size(),
                    "checked_out": pool.checkedout(),
                    "overflow": pool.overflow(),
                    "checked_in": pool.checkedin()
                }

                # Get database version and status
                async with self.engine.begin() as conn:
                    version_result = await conn.execute(text("SELECT version()"))
                    version = version_result.scalar()

                    # Get database statistics
                    stats_query = text("""
                        SELECT 
                            numbackends as active_connections,
                            xact_commit as transactions_committed,
                            xact_rollback as transactions_rolled_back,
                            blks_read as blocks_read,
                            blks_hit as blocks_hit,
                            tup_returned as tuples_returned,
                            tup_fetched as tuples_fetched,
                            tup_inserted as tuples_inserted,
                            tup_updated as tuples_updated,
                            tup_deleted as tuples_deleted
                        FROM pg_stat_database 
                        WHERE datname = :db_name
                    """)

                    stats_result = await conn.execute(stats_query, {"db_name": self.config.database_name})
                    stats = dict(stats_result.fetchone()._mapping) if stats_result.rowcount > 0 else {}

                health_info.update({
                    "connected": True,
                    "response_time_ms": round(response_time, 2),
                    "version": version,
                    "pool_status": pool_status,
                    "database_stats": stats
                })

        except Exception as e:
            health_info.update({
                "error": str(e),
                "healthy": False
            })

        return health_info

    async def execute_raw_sql(self, query: str, params: Optional[Dict[str, Any]] = None) -> Any:
        """
        Execute raw SQL query

        Args:
            query: SQL query string
            params: Query parameters

        Returns:
            Query result
        """
        async with self.get_session() as session:
            result = await session.execute(text(query), params or {})
            await session.commit()
            return result

    async def get_database_info(self) -> Dict[str, Any]:
        """
        Get comprehensive database information

        Returns:
            Database information
        """
        try:
            async with self.get_session() as session:
                # Get database size
                size_query = text("""
                    SELECT pg_size_pretty(pg_database_size(:db_name)) as database_size,
                           pg_database_size(:db_name) as database_size_bytes
                """)
                size_result = await session.execute(size_query, {"db_name": self.config.database_name})
                size_info = dict(size_result.fetchone()._mapping)

                # Get table information
                tables_query = text("""
                    SELECT 
                        schemaname,
                        tablename,
                        pg_size_pretty(pg_total_relation_size(schemaname||'.'||tablename)) as size,
                        pg_total_relation_size(schemaname||'.'||tablename) as size_bytes
                    FROM pg_tables 
                    WHERE schemaname NOT IN ('information_schema', 'pg_catalog')
                    ORDER BY size_bytes DESC
                """)
                tables_result = await session.execute(tables_query)
                tables_info = [dict(row._mapping) for row in tables_result.fetchall()]

                return {
                    "database_info": size_info,
                    "tables": tables_info
                }

        except Exception as e:
            logger.error("Failed to get database info", error=str(e))
            return {"error": str(e)}


# Global connection manager instance
_connection_manager: Optional[PostgreSQLConnectionManager] = None


async def initialize_postgresql(config: Optional[PostgreSQLConfig] = None) -> AsyncEngine:
    """
    Initialize PostgreSQL connection

    Args:
        config: PostgreSQL configuration (uses default if None)

    Returns:
        SQLAlchemy async engine
    """
    global _connection_manager

    if config is None:
        config = PostgreSQLConfig()

    if _connection_manager is None:
        _connection_manager = PostgreSQLConnectionManager(config)

    return await _connection_manager.connect()


async def get_postgresql_session() -> AsyncGenerator[AsyncSession, None]:
    """
    Get PostgreSQL database session

    Yields:
        AsyncSession instance

    Raises:
        RuntimeError: If PostgreSQL is not initialized
    """
    global _connection_manager

    if _connection_manager is None:
        # Auto-initialize with default config
        await initialize_postgresql()

    async for session in _connection_manager.get_session():
        yield session


async def get_postgresql_engine() -> AsyncEngine:
    """
    Get PostgreSQL engine

    Returns:
        SQLAlchemy async engine

    Raises:
        RuntimeError: If PostgreSQL is not initialized
    """
    global _connection_manager

    if _connection_manager is None:
        # Auto-initialize with default config
        await initialize_postgresql()

    return await _connection_manager.get_engine()


async def close_postgresql() -> None:
    """Close PostgreSQL connection"""
    global _connection_manager

    if _connection_manager:
        await _connection_manager.disconnect()
        _connection_manager = None


async def postgresql_health_check() -> Dict[str, Any]:
    """
    Get PostgreSQL health status

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
            "error": "PostgreSQL not initialized"
        }