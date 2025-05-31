"""
Session Repository Implementation
================================

Redis repository for session management and user state caching with
comprehensive session lifecycle management and multi-tenant isolation.

Features:
- Session CRUD operations with TTL management
- User session indexing and lookup
- Session extension and cleanup
- Multi-tenant isolation
- Performance monitoring
- Session analytics and tracking
"""

from datetime import datetime, timedelta
from typing import Optional, List, Dict, Any, Set
from redis.asyncio import Redis
import json
import structlog
from dataclasses import dataclass, asdict
from enum import Enum

from .base_repository import BaseRepository
from .exceptions import (
    RepositoryError, EntityNotFoundError, ValidationError,
    ConnectionError, TimeoutError
)

# Type definitions
TenantId = str
SessionId = str
UserId = str


class SessionStatus(str, Enum):
    ACTIVE = "active"
    EXPIRED = "expired"
    TERMINATED = "terminated"


@dataclass
class SessionData:
    """Session data structure for Redis storage"""
    session_id: SessionId
    tenant_id: TenantId
    user_id: UserId
    conversation_id: Optional[str] = None
    channel: str = "web"
    created_at: datetime = None
    last_activity_at: datetime = None
    expires_at: datetime = None
    status: SessionStatus = SessionStatus.ACTIVE

    # Session context and state
    context: Dict[str, Any] = None
    user_preferences: Dict[str, Any] = None
    device_info: Dict[str, Any] = None
    location_info: Dict[str, Any] = None

    # Security and tracking
    ip_address: Optional[str] = None
    user_agent: Optional[str] = None
    last_seen_ip: Optional[str] = None

    # Business context
    customer_tier: Optional[str] = None
    session_tags: List[str] = None

    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.utcnow()
        if self.last_activity_at is None:
            self.last_activity_at = datetime.utcnow()
        if self.expires_at is None:
            self.expires_at = datetime.utcnow() + timedelta(hours=1)  # Default 1 hour
        if self.context is None:
            self.context = {}
        if self.user_preferences is None:
            self.user_preferences = {}
        if self.device_info is None:
            self.device_info = {}
        if self.location_info is None:
            self.location_info = {}
        if self.session_tags is None:
            self.session_tags = []

    @staticmethod
    def get_cache_key(tenant_id: TenantId, session_id: SessionId) -> str:
        """Generate Redis cache key for session"""
        return f"session:{tenant_id}:{session_id}"

    @staticmethod
    def get_user_sessions_key(tenant_id: TenantId, user_id: UserId) -> str:
        """Generate Redis key for user sessions index"""
        return f"user_sessions:{tenant_id}:{user_id}"

    def to_redis_hash(self) -> Dict[str, str]:
        """Convert to Redis hash format (all string values)"""
        data = asdict(self)

        # Convert datetime objects to ISO format
        for key, value in data.items():
            if isinstance(value, datetime):
                data[key] = value.isoformat()
            elif isinstance(value, (dict, list)):
                data[key] = json.dumps(value)
            elif isinstance(value, Enum):
                data[key] = value.value
            elif value is None:
                data[key] = ""
            else:
                data[key] = str(value)

        return data

    @classmethod
    def from_redis_hash(cls, data: Dict[str, str]) -> "SessionData":
        """Create from Redis hash data"""
        # Convert string values back to proper types
        converted_data = {}

        for key, value in data.items():
            if not value:  # Empty string means None
                converted_data[key] = None
                continue

            if key in ['created_at', 'last_activity_at', 'expires_at']:
                converted_data[key] = datetime.fromisoformat(value)
            elif key in ['context', 'user_preferences', 'device_info', 'location_info', 'session_tags']:
                converted_data[key] = json.loads(value) if value else {}
            elif key == 'status':
                converted_data[key] = SessionStatus(value)
            else:
                converted_data[key] = value

        return cls(**converted_data)

    def is_expired(self) -> bool:
        """Check if session is expired"""
        return datetime.utcnow() > self.expires_at

    def update_activity(self) -> None:
        """Update last activity timestamp"""
        self.last_activity_at = datetime.utcnow()

    def extend_session(self, hours: int = 1) -> None:
        """Extend session expiration"""
        self.expires_at = datetime.utcnow() + timedelta(hours=hours)
        self.update_activity()

    def add_context(self, key: str, value: Any) -> None:
        """Add context data to session"""
        self.context[key] = value
        self.update_activity()

    def get_context(self, key: str, default: Any = None) -> Any:
        """Get context data from session"""
        return self.context.get(key, default)

    def get_ttl_seconds(self) -> int:
        """Calculate TTL in seconds"""
        delta = self.expires_at - datetime.utcnow()
        return max(0, int(delta.total_seconds()))


class SessionRepository:
    """
    Repository for session management in Redis

    Provides session lifecycle management with:
    - TTL-based expiration
    - User session indexing
    - Multi-tenant isolation
    - Performance monitoring
    - Session analytics
    """

    def __init__(self, redis_client: Redis):
        """
        Initialize session repository

        Args:
            redis_client: Redis async client instance
        """
        self.redis = redis_client
        self.default_ttl = 3600  # 1 hour in seconds
        self.logger = structlog.get_logger("SessionRepository")

    async def create_session(self, session_data: SessionData) -> bool:
        """
        Create a new session in Redis

        Args:
            session_data: Session data to store

        Returns:
            True if session was created successfully

        Raises:
            RepositoryError: If creation fails
            ValidationError: If session data is invalid
        """
        try:
            # Validate session data
            self._validate_session_data(session_data)

            cache_key = SessionData.get_cache_key(
                session_data.tenant_id,
                session_data.session_id
            )

            # Convert to Redis hash format
            hash_data = session_data.to_redis_hash()

            # Calculate TTL
            ttl_seconds = session_data.get_ttl_seconds()
            if ttl_seconds <= 0:
                ttl_seconds = self.default_ttl

            # Use pipeline for atomic operations
            pipe = self.redis.pipeline()

            # Store session data
            pipe.hset(cache_key, mapping=hash_data)
            pipe.expire(cache_key, ttl_seconds)

            # Add to user sessions index
            user_sessions_key = SessionData.get_user_sessions_key(
                session_data.tenant_id,
                session_data.user_id
            )
            pipe.sadd(user_sessions_key, session_data.session_id)
            pipe.expire(user_sessions_key, 86400)  # 24 hours

            # Update session stats
            stats_key = f"session_stats:{session_data.tenant_id}:daily:{datetime.utcnow().strftime('%Y-%m-%d')}"
            pipe.incr(stats_key)
            pipe.expire(stats_key, 86400 * 7)  # Keep for 7 days

            # Execute pipeline
            results = await pipe.execute()
            success = all(results[:2])  # Check main operations

            if success:
                self.logger.info(
                    "Session created successfully",
                    session_id=session_data.session_id,
                    tenant_id=session_data.tenant_id,
                    user_id=session_data.user_id,
                    ttl_seconds=ttl_seconds
                )

            return success

        except Exception as e:
            self.logger.error(
                "Failed to create session",
                session_id=session_data.session_id,
                error=str(e)
            )
            raise RepositoryError(f"Failed to create session: {e}", original_error=e)

    async def get_session(
            self,
            tenant_id: TenantId,
            session_id: SessionId
    ) -> Optional[SessionData]:
        """
        Get session data by ID

        Args:
            tenant_id: Tenant identifier
            session_id: Session identifier

        Returns:
            Session data if found and not expired, None otherwise

        Raises:
            RepositoryError: If retrieval fails
        """
        try:
            cache_key = SessionData.get_cache_key(tenant_id, session_id)
            hash_data = await self.redis.hgetall(cache_key)

            if not hash_data:
                self.logger.debug(
                    "Session not found",
                    tenant_id=tenant_id,
                    session_id=session_id
                )
                return None

            # Convert Redis hash to session data
            session_data = SessionData.from_redis_hash(hash_data)

            # Check if session is expired
            if session_data.is_expired():
                self.logger.info(
                    "Session expired, cleaning up",
                    session_id=session_id,
                    tenant_id=tenant_id,
                    expired_at=session_data.expires_at
                )
                await self.delete_session(tenant_id, session_id)
                return None

            self.logger.debug(
                "Session retrieved successfully",
                session_id=session_id,
                tenant_id=tenant_id,
                expires_at=session_data.expires_at
            )

            return session_data

        except Exception as e:
            self.logger.error(
                "Failed to get session",
                tenant_id=tenant_id,
                session_id=session_id,
                error=str(e)
            )
            raise RepositoryError(f"Failed to get session: {e}", original_error=e)

    async def update_session(self, session_data: SessionData) -> bool:
        """
        Update existing session

        Args:
            session_data: Updated session data

        Returns:
            True if updated successfully

        Raises:
            EntityNotFoundError: If session doesn't exist
            RepositoryError: If update fails
        """
        try:
            # Validate session data
            self._validate_session_data(session_data)

            cache_key = SessionData.get_cache_key(
                session_data.tenant_id,
                session_data.session_id
            )

            # Check if session exists
            exists = await self.redis.exists(cache_key)
            if not exists:
                raise EntityNotFoundError("Session", session_data.session_id)

            # Update last activity
            session_data.update_activity()

            # Convert to Redis hash format
            hash_data = session_data.to_redis_hash()

            # Calculate new TTL
            ttl_seconds = session_data.get_ttl_seconds()
            if ttl_seconds <= 0:
                ttl_seconds = self.default_ttl

            # Update session data and TTL
            pipe = self.redis.pipeline()
            pipe.hset(cache_key, mapping=hash_data)
            pipe.expire(cache_key, ttl_seconds)
            results = await pipe.execute()

            success = all(results)

            if success:
                self.logger.debug(
                    "Session updated successfully",
                    session_id=session_data.session_id,
                    tenant_id=session_data.tenant_id,
                    new_ttl=ttl_seconds
                )

            return success

        except EntityNotFoundError:
            raise
        except Exception as e:
            self.logger.error(
                "Failed to update session",
                session_id=session_data.session_id,
                error=str(e)
            )
            raise RepositoryError(f"Failed to update session: {e}", original_error=e)

    async def delete_session(
            self,
            tenant_id: TenantId,
            session_id: SessionId
    ) -> bool:
        """
        Delete session by ID

        Args:
            tenant_id: Tenant identifier
            session_id: Session identifier

        Returns:
            True if deleted, False if not found

        Raises:
            RepositoryError: If deletion fails
        """
        try:
            # Get session data first to clean up indexes
            session_data = await self.get_session(tenant_id, session_id)

            cache_key = SessionData.get_cache_key(tenant_id, session_id)

            # Use pipeline for atomic operations
            pipe = self.redis.pipeline()

            # Delete session data
            pipe.delete(cache_key)

            # Clean up user sessions index
            if session_data:
                user_sessions_key = SessionData.get_user_sessions_key(
                    tenant_id,
                    session_data.user_id
                )
                pipe.srem(user_sessions_key, session_id)

            # Execute pipeline
            results = await pipe.execute()
            deleted = results[0] > 0  # First result is delete count

            if deleted:
                self.logger.info(
                    "Session deleted successfully",
                    session_id=session_id,
                    tenant_id=tenant_id
                )

            return deleted

        except Exception as e:
            self.logger.error(
                "Failed to delete session",
                tenant_id=tenant_id,
                session_id=session_id,
                error=str(e)
            )
            raise RepositoryError(f"Failed to delete session: {e}", original_error=e)

    async def extend_session(
            self,
            tenant_id: TenantId,
            session_id: SessionId,
            hours: int = 1
    ) -> bool:
        """
        Extend session expiration

        Args:
            tenant_id: Tenant identifier
            session_id: Session identifier
            hours: Hours to extend by

        Returns:
            True if extended successfully

        Raises:
            EntityNotFoundError: If session doesn't exist
            RepositoryError: If extension fails
        """
        try:
            session_data = await self.get_session(tenant_id, session_id)
            if not session_data:
                raise EntityNotFoundError("Session", session_id)

            # Extend expiration
            session_data.extend_session(hours)

            # Update in Redis
            success = await self.update_session(session_data)

            if success:
                self.logger.info(
                    "Session extended successfully",
                    session_id=session_id,
                    tenant_id=tenant_id,
                    extended_hours=hours,
                    new_expires_at=session_data.expires_at
                )

            return success

        except EntityNotFoundError:
            raise
        except Exception as e:
            self.logger.error(
                "Failed to extend session",
                tenant_id=tenant_id,
                session_id=session_id,
                error=str(e)
            )
            raise RepositoryError(f"Failed to extend session: {e}", original_error=e)

    async def get_user_sessions(
            self,
            tenant_id: TenantId,
            user_id: UserId,
            include_expired: bool = False
    ) -> List[SessionData]:
        """
        Get all sessions for a user

        Args:
            tenant_id: Tenant identifier
            user_id: User identifier
            include_expired: Include expired sessions

        Returns:
            List of user sessions

        Raises:
            RepositoryError: If retrieval fails
        """
        try:
            user_sessions_key = SessionData.get_user_sessions_key(tenant_id, user_id)
            session_ids = await self.redis.smembers(user_sessions_key)

            sessions = []
            expired_sessions = []

            for session_id in session_ids:
                session_data = await self.get_session(tenant_id, session_id.decode())

                if session_data:
                    if session_data.is_expired():
                        if include_expired:
                            sessions.append(session_data)
                        expired_sessions.append(session_id.decode())
                    else:
                        sessions.append(session_data)
                else:
                    # Session not found, remove from index
                    expired_sessions.append(session_id.decode())

            # Clean up expired sessions from index
            if expired_sessions:
                await self.redis.srem(user_sessions_key, *expired_sessions)

            self.logger.debug(
                "Retrieved user sessions",
                tenant_id=tenant_id,
                user_id=user_id,
                active_sessions=len(sessions),
                cleaned_expired=len(expired_sessions)
            )

            return sessions

        except Exception as e:
            self.logger.error(
                "Failed to get user sessions",
                tenant_id=tenant_id,
                user_id=user_id,
                error=str(e)
            )
            raise RepositoryError(f"Failed to get user sessions: {e}", original_error=e)

    async def cleanup_expired_sessions(self, batch_size: int = 100) -> int:
        """
        Clean up expired sessions (background task)

        Args:
            batch_size: Number of sessions to process in each batch

        Returns:
            Number of sessions cleaned up

        Raises:
            RepositoryError: If cleanup fails
        """
        try:
            cleaned_count = 0

            # Get all session keys
            pattern = "session:*"
            async for key in self.redis.scan_iter(match=pattern, count=batch_size):
                try:
                    # Check TTL
                    ttl = await self.redis.ttl(key)

                    if ttl == -2:  # Key doesn't exist
                        cleaned_count += 1
                    elif ttl == -1:  # Key exists but no TTL
                        # Set default TTL for orphaned keys
                        await self.redis.expire(key, self.default_ttl)
                        self.logger.warning(
                            "Found session without TTL, setting default",
                            key=key.decode()
                        )

                except Exception as e:
                    self.logger.warning(
                        "Failed to check session key",
                        key=key.decode(),
                        error=str(e)
                    )
                    continue

            self.logger.info(
                "Session cleanup completed",
                cleaned_sessions=cleaned_count
            )

            return cleaned_count

        except Exception as e:
            self.logger.error(
                "Failed to cleanup expired sessions",
                error=str(e)
            )
            raise RepositoryError(f"Failed to cleanup expired sessions: {e}", original_error=e)

    async def get_session_stats(
            self,
            tenant_id: TenantId,
            days: int = 7
    ) -> Dict[str, Any]:
        """
        Get session statistics for a tenant

        Args:
            tenant_id: Tenant identifier
            days: Number of days to include

        Returns:
            Session statistics dictionary
        """
        try:
            stats = {
                "daily_sessions": {},
                "total_sessions": 0,
                "active_sessions": 0
            }

            # Get daily session counts
            for i in range(days):
                date = (datetime.utcnow() - timedelta(days=i)).strftime('%Y-%m-%d')
                stats_key = f"session_stats:{tenant_id}:daily:{date}"
                count = await self.redis.get(stats_key)
                stats["daily_sessions"][date] = int(count) if count else 0
                stats["total_sessions"] += stats["daily_sessions"][date]

            # Count currently active sessions
            pattern = f"session:{tenant_id}:*"
            active_count = 0
            async for key in self.redis.scan_iter(match=pattern):
                ttl = await self.redis.ttl(key)
                if ttl > 0:  # Active session
                    active_count += 1

            stats["active_sessions"] = active_count

            return stats

        except Exception as e:
            self.logger.error(
                "Failed to get session stats",
                tenant_id=tenant_id,
                error=str(e)
            )
            return {"error": str(e)}

    # Private helper methods

    def _validate_session_data(self, session_data: SessionData) -> None:
        """
        Validate session data before operations

        Args:
            session_data: Session data to validate

        Raises:
            ValidationError: If validation fails
        """
        if not session_data.session_id:
            raise ValidationError("session_id is required", field="session_id")

        if not session_data.tenant_id:
            raise ValidationError("tenant_id is required", field="tenant_id")

        if not session_data.user_id:
            raise ValidationError("user_id is required", field="user_id")

        if session_data.expires_at <= datetime.utcnow():
            raise ValidationError("expires_at must be in the future", field="expires_at")

        # Validate session duration (max 24 hours)
        max_duration = timedelta(hours=24)
        if session_data.expires_at - session_data.created_at > max_duration:
            raise ValidationError("Session duration cannot exceed 24 hours", field="expires_at")


# Redis connection management
_redis_client: Optional[Redis] = None


async def get_redis_client() -> Redis:
    """
    Get Redis client instance

    Returns:
        Redis client instance

    Raises:
        ConnectionError: If Redis connection fails
    """
    global _redis_client

    if _redis_client is None:
        try:
            from redis.asyncio import Redis
            # This would normally come from configuration
            _redis_client = Redis.from_url("redis://localhost:6379", decode_responses=True)

            # Test connection
            await _redis_client.ping()

        except Exception as e:
            raise ConnectionError("Redis", original_error=e)

    return _redis_client


# Dependency injection function
async def get_session_repository() -> SessionRepository:
    """
    Get session repository instance for dependency injection

    Returns:
        SessionRepository instance
    """
    redis_client = await get_redis_client()
    return SessionRepository(redis_client)