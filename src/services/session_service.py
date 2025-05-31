"""
Session Service

Manage user sessions, conversation context, and state persistence.
Handles session lifecycle, context updates, and session-based routing.
"""

from datetime import datetime, timedelta
from typing import Optional, Dict, Any, List
from uuid import uuid4

from src.services.base_service import BaseService
from src.services.exceptions import ServiceError, ValidationError, NotFoundError, SessionError, UnauthorizedError
from src.models.redis.session_cache import SessionData
from src.models.types import TenantId, UserId, SessionId, ChannelType
from src.repositories.session_repository import SessionRepository


class SessionService(BaseService):
    """Service for managing user sessions and conversation state"""

    def __init__(self, session_repo: SessionRepository):
        super().__init__()
        self.session_repo = session_repo
        self.default_session_duration_hours = 2
        self.max_session_duration_hours = 24

    async def create_session(
            self,
            user_id: UserId,
            tenant_id: TenantId,
            channel: ChannelType,
            conversation_id: Optional[str] = None,
            initial_context: Optional[Dict[str, Any]] = None
    ) -> SessionData:
        """
        Create a new user session

        Args:
            user_id: User identifier
            tenant_id: Tenant identifier
            channel: Communication channel
            conversation_id: Optional existing conversation ID
            initial_context: Optional initial session context

        Returns:
            Created SessionData
        """
        try:
            await self.validate_tenant_access(tenant_id)

            # Generate session ID
            session_id = str(uuid4())

            # Create session data
            session_data = SessionData(
                session_id=session_id,
                tenant_id=tenant_id,
                user_id=user_id,
                conversation_id=conversation_id,
                channel=channel,
                context=initial_context or {},
                expires_at=datetime.utcnow() + timedelta(
                    hours=self.default_session_duration_hours
                )
            )

            # Store in repository
            success = await self.session_repo.create_session(session_data)

            if not success:
                raise ServiceError("Failed to create session in repository")

            self.log_operation(
                "create_session",
                tenant_id=tenant_id,
                user_id=user_id,
                session_id=session_id,
                channel=channel.value
            )

            return session_data

        except Exception as e:
            error = self.handle_service_error(
                e, "create_session",
                tenant_id=tenant_id,
                user_id=user_id
            )
            raise error

    async def get_session(
            self,
            tenant_id: TenantId,
            session_id: SessionId
    ) -> Optional[SessionData]:
        """
        Retrieve session by ID

        Args:
            tenant_id: Tenant identifier
            session_id: Session identifier

        Returns:
            SessionData if found, None otherwise
        """
        try:
            await self.validate_tenant_access(tenant_id)

            session_data = await self.session_repo.get_session(tenant_id, session_id)

            if session_data:
                # Check if session is expired
                if session_data.is_expired():
                    await self.session_repo.delete_session(tenant_id, session_id)
                    return None

                self.log_operation(
                    "get_session",
                    tenant_id=tenant_id,
                    session_id=session_id,
                    user_id=session_data.user_id
                )

            return session_data

        except Exception as e:
            error = self.handle_service_error(
                e, "get_session",
                tenant_id=tenant_id,
                session_id=session_id
            )
            raise error

    async def update_session_context(
            self,
            tenant_id: TenantId,
            session_id: SessionId,
            context_updates: Dict[str, Any]
    ) -> bool:
        """
        Update session context

        Args:
            tenant_id: Tenant identifier
            session_id: Session identifier
            context_updates: Context updates to apply

        Returns:
            True if successful
        """
        try:
            await self.validate_tenant_access(tenant_id)

            # Get existing session
            session_data = await self.get_session(tenant_id, session_id)

            if not session_data:
                raise NotFoundError(f"Session {session_id} not found")

            # Update context
            session_data.context.update(context_updates)

            # Save updated session
            success = await self.session_repo.update_session(session_data)

            if success:
                self.log_operation(
                    "update_session_context",
                    tenant_id=tenant_id,
                    session_id=session_id,
                    updates_count=len(context_updates)
                )

            return success

        except NotFoundError:
            raise
        except Exception as e:
            error = self.handle_service_error(
                e, "update_session_context",
                tenant_id=tenant_id,
                session_id=session_id
            )
            raise error

    async def extend_session(
            self,
            tenant_id: TenantId,
            session_id: SessionId,
            hours: int = None
    ) -> bool:
        """
        Extend session expiration

        Args:
            tenant_id: Tenant identifier
            session_id: Session identifier
            hours: Hours to extend (default: default_session_duration_hours)

        Returns:
            True if successful
        """
        try:
            await self.validate_tenant_access(tenant_id)

            hours = hours or self.default_session_duration_hours

            if hours > self.max_session_duration_hours:
                raise ValidationError(
                    f"Cannot extend session beyond {self.max_session_duration_hours} hours"
                )

            success = await self.session_repo.extend_session(
                tenant_id, session_id, hours
            )

            if success:
                self.log_operation(
                    "extend_session",
                    tenant_id=tenant_id,
                    session_id=session_id,
                    extended_hours=hours
                )

            return success

        except ValidationError:
            raise
        except Exception as e:
            error = self.handle_service_error(
                e, "extend_session",
                tenant_id=tenant_id,
                session_id=session_id
            )
            raise error

    async def delete_session(
            self,
            tenant_id: TenantId,
            session_id: SessionId
    ) -> bool:
        """
        Delete session

        Args:
            tenant_id: Tenant identifier
            session_id: Session identifier

        Returns:
            True if successful
        """
        try:
            await self.validate_tenant_access(tenant_id)

            success = await self.session_repo.delete_session(tenant_id, session_id)

            if success:
                self.log_operation(
                    "delete_session",
                    tenant_id=tenant_id,
                    session_id=session_id
                )

            return success

        except Exception as e:
            error = self.handle_service_error(
                e, "delete_session",
                tenant_id=tenant_id,
                session_id=session_id
            )
            raise error

    async def get_user_sessions(
            self,
            tenant_id: TenantId,
            user_id: UserId
    ) -> List[SessionData]:
        """
        Get all active sessions for a user

        Args:
            tenant_id: Tenant identifier
            user_id: User identifier

        Returns:
            List of active sessions
        """
        try:
            await self.validate_tenant_access(tenant_id)

            sessions = await self.session_repo.get_user_sessions(tenant_id, user_id)

            # Filter out expired sessions
            active_sessions = []
            for session in sessions:
                if not session.is_expired():
                    active_sessions.append(session)
                else:
                    # Clean up expired session
                    await self.session_repo.delete_session(
                        tenant_id, session.session_id
                    )

            self.log_operation(
                "get_user_sessions",
                tenant_id=tenant_id,
                user_id=user_id,
                sessions_count=len(active_sessions)
            )

            return active_sessions

        except Exception as e:
            error = self.handle_service_error(
                e, "get_user_sessions",
                tenant_id=tenant_id,
                user_id=user_id
            )
            raise error

    async def cleanup_expired_sessions(self) -> int:
        """
        Clean up expired sessions (background task)

        Returns:
            Number of sessions cleaned up
        """
        try:
            cleaned_count = await self.session_repo.cleanup_expired_sessions()

            self.log_operation(
                "cleanup_expired_sessions",
                cleaned_count=cleaned_count
            )

            return cleaned_count

        except Exception as e:
            error = self.handle_service_error(e, "cleanup_expired_sessions")
            raise error

    async def get_session_by_conversation(
            self,
            tenant_id: TenantId,
            conversation_id: str
    ) -> Optional[SessionData]:
        """
        Get session associated with a conversation

        Args:
            tenant_id: Tenant identifier
            conversation_id: Conversation identifier

        Returns:
            SessionData if found, None otherwise
        """
        try:
            await self.validate_tenant_access(tenant_id)

            session_data = await self.session_repo.get_session_by_conversation(
                tenant_id, conversation_id
            )

            if session_data and session_data.is_expired():
                await self.session_repo.delete_session(tenant_id, session_data.session_id)
                return None

            return session_data

        except Exception as e:
            error = self.handle_service_error(
                e, "get_session_by_conversation",
                tenant_id=tenant_id
            )
            raise error

    async def update_session_activity(
            self,
            tenant_id: TenantId,
            session_id: SessionId
    ) -> bool:
        """
        Update session last activity timestamp

        Args:
            tenant_id: Tenant identifier
            session_id: Session identifier

        Returns:
            True if successful
        """
        try:
            await self.validate_tenant_access(tenant_id)

            success = await self.session_repo.update_activity(tenant_id, session_id)

            if success:
                self.log_operation(
                    "update_session_activity",
                    tenant_id=tenant_id,
                    session_id=session_id
                )

            return success

        except Exception as e:
            error = self.handle_service_error(
                e, "update_session_activity",
                tenant_id=tenant_id,
                session_id=session_id
            )
            raise error

    async def get_session_statistics(
            self,
            tenant_id: TenantId,
            time_window_hours: int = 24
    ) -> Dict[str, Any]:
        """
        Get session statistics for a tenant

        Args:
            tenant_id: Tenant identifier
            time_window_hours: Time window for statistics in hours

        Returns:
            Dictionary with session statistics
        """
        try:
            await self.validate_tenant_access(tenant_id)

            stats = await self.session_repo.get_session_stats(
                tenant_id, time_window_hours
            )

            self.log_operation(
                "get_session_statistics",
                tenant_id=tenant_id,
                time_window_hours=time_window_hours
            )

            return stats

        except Exception as e:
            error = self.handle_service_error(
                e, "get_session_statistics",
                tenant_id=tenant_id
            )
            raise error

    async def validate_session_access(
            self,
            tenant_id: TenantId,
            session_id: SessionId,
            user_id: UserId
    ) -> bool:
        """
        Validate that a user has access to a specific session

        Args:
            tenant_id: Tenant identifier
            session_id: Session identifier
            user_id: User identifier

        Returns:
            True if user has access to session

        Raises:
            UnauthorizedError: If access is denied
            NotFoundError: If session doesn't exist
        """
        try:
            await self.validate_tenant_access(tenant_id)

            session_data = await self.get_session(tenant_id, session_id)

            if not session_data:
                raise NotFoundError(f"Session {session_id} not found")

            if session_data.user_id != user_id:
                raise UnauthorizedError(
                    f"User {user_id} does not have access to session {session_id}"
                )

            return True

        except (NotFoundError, UnauthorizedError):
            raise
        except Exception as e:
            error = self.handle_service_error(
                e, "validate_session_access",
                tenant_id=tenant_id,
                session_id=session_id,
                user_id=user_id
            )
            raise error

    async def transfer_session(
            self,
            tenant_id: TenantId,
            session_id: SessionId,
            new_channel: ChannelType,
            channel_metadata: Optional[Dict[str, Any]] = None
    ) -> bool:
        """
        Transfer session to a new channel (e.g., web to WhatsApp)

        Args:
            tenant_id: Tenant identifier
            session_id: Session identifier
            new_channel: New channel type
            channel_metadata: Optional metadata for new channel

        Returns:
            True if successful
        """
        try:
            await self.validate_tenant_access(tenant_id)

            session_data = await self.get_session(tenant_id, session_id)

            if not session_data:
                raise NotFoundError(f"Session {session_id} not found")

            # Update session with new channel
            session_data.channel = new_channel
            if channel_metadata:
                session_data.context.update({"channel_metadata": channel_metadata})

            success = await self.session_repo.update_session(session_data)

            if success:
                self.log_operation(
                    "transfer_session",
                    tenant_id=tenant_id,
                    session_id=session_id,
                    old_channel=session_data.channel.value,
                    new_channel=new_channel.value
                )

            return success

        except (NotFoundError, ValidationError):
            raise
        except Exception as e:
            error = self.handle_service_error(
                e, "transfer_session",
                tenant_id=tenant_id,
                session_id=session_id
            )
            raise error