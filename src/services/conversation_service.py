"""
Conversation Service

Manages conversation lifecycle, metadata, and business operations.
Handles conversation creation, updates, retrieval, and analytics.
"""

from datetime import datetime, timedelta
from typing import Dict, Any, Optional, List
from uuid import uuid4

from src.services.base_service import BaseService
from src.services.exceptions import ServiceError, ValidationError, NotFoundError, ConflictError
from src.models.mongo.conversation_model import ConversationDocument
from src.models.mongo.message_model import MessageDocument
from src.models.types import TenantId, UserId, ConversationStatus, ChannelType
from src.repositories.conversation_repository import ConversationRepository
from src.repositories.message_repository import MessageRepository


class ConversationService(BaseService):
    """Service for managing conversation lifecycle and operations"""

    def __init__(
            self,
            conversation_repo: ConversationRepository,
            message_repo: MessageRepository
    ):
        super().__init__()
        self.conversation_repo = conversation_repo
        self.message_repo = message_repo

    async def create_conversation(
            self,
            tenant_id: TenantId,
            user_id: UserId,
            channel: ChannelType,
            session_id: Optional[str] = None,
            initial_context: Optional[Dict[str, Any]] = None,
            channel_metadata: Optional[Dict[str, Any]] = None,
            user_context: Optional[Dict[str, Any]] = None
    ) -> ConversationDocument:
        """
        Create a new conversation

        Args:
            tenant_id: Tenant identifier
            user_id: User identifier
            channel: Communication channel
            session_id: Optional session identifier
            initial_context: Optional initial conversation context
            channel_metadata: Optional channel-specific metadata
            user_context: User authentication context

        Returns:
            Created conversation document
        """
        try:
            await self.validate_tenant_access(tenant_id, user_context)

            conversation_id = str(uuid4())

            conversation = ConversationDocument(
                conversation_id=conversation_id,
                tenant_id=tenant_id,
                user_id=user_id,
                session_id=session_id,
                channel=channel,
                channel_metadata=channel_metadata or {},
                status=ConversationStatus.ACTIVE,
                context=initial_context or {}
            )

            created_conversation = await self.conversation_repo.create(conversation)

            self.log_operation(
                "create_conversation",
                tenant_id=tenant_id,
                user_id=user_id,
                conversation_id=conversation_id,
                channel=channel.value
            )

            return created_conversation

        except Exception as e:
            error = self.handle_service_error(
                e, "create_conversation",
                tenant_id=tenant_id
            )
            raise error

    async def get_conversation(
            self,
            tenant_id: TenantId,
            conversation_id: str,
            user_context: Optional[Dict[str, Any]] = None
    ) -> Optional[ConversationDocument]:
        """
        Get conversation by ID

        Args:
            tenant_id: Tenant identifier
            conversation_id: Conversation identifier
            user_context: User authentication context

        Returns:
            Conversation document if found, None otherwise
        """
        try:
            await self.validate_tenant_access(tenant_id, user_context)

            conversation = await self.conversation_repo.get_by_id(conversation_id)

            # Verify tenant ownership
            if conversation and conversation.tenant_id != tenant_id:
                raise NotFoundError(f"Conversation {conversation_id} not found")

            if conversation:
                self.log_operation(
                    "get_conversation",
                    tenant_id=tenant_id,
                    conversation_id=conversation_id
                )

            return conversation

        except Exception as e:
            error = self.handle_service_error(
                e, "get_conversation",
                tenant_id=tenant_id
            )
            raise error

    async def update_conversation(
            self,
            tenant_id: TenantId,
            conversation_id: str,
            updates: Dict[str, Any],
            user_context: Optional[Dict[str, Any]] = None
    ) -> ConversationDocument:
        """
        Update conversation

        Args:
            tenant_id: Tenant identifier
            conversation_id: Conversation identifier
            updates: Fields to update
            user_context: User authentication context

        Returns:
            Updated conversation document
        """
        try:
            await self.validate_tenant_access(tenant_id, user_context)

            conversation = await self.get_conversation(tenant_id, conversation_id, user_context)
            if not conversation:
                raise NotFoundError(f"Conversation {conversation_id} not found")

            # Apply updates
            for field, value in updates.items():
                if hasattr(conversation, field):
                    setattr(conversation, field, value)

            # Update timestamp
            conversation.last_activity_at = datetime.utcnow()

            updated_conversation = await self.conversation_repo.update(conversation)

            self.log_operation(
                "update_conversation",
                tenant_id=tenant_id,
                conversation_id=conversation_id,
                updated_fields=list(updates.keys())
            )

            return updated_conversation

        except (NotFoundError, ValidationError):
            raise
        except Exception as e:
            error = self.handle_service_error(
                e, "update_conversation",
                tenant_id=tenant_id
            )
            raise error

    async def close_conversation(
            self,
            tenant_id: TenantId,
            conversation_id: str,
            reason: str = "completed",
            summary: Optional[str] = None,
            user_context: Optional[Dict[str, Any]] = None
    ) -> ConversationDocument:
        """
        Close a conversation

        Args:
            tenant_id: Tenant identifier
            conversation_id: Conversation identifier
            reason: Reason for closing
            summary: Optional conversation summary
            user_context: User authentication context

        Returns:
            Updated conversation document
        """
        try:
            await self.validate_tenant_access(tenant_id, user_context)

            conversation = await self.get_conversation(tenant_id, conversation_id, user_context)
            if not conversation:
                raise NotFoundError(f"Conversation {conversation_id} not found")

            if conversation.status == ConversationStatus.COMPLETED:
                raise ConflictError("Conversation is already closed")

            # Update conversation status
            conversation.status = ConversationStatus.COMPLETED
            conversation.completed_at = datetime.utcnow()
            conversation.last_activity_at = datetime.utcnow()

            # Calculate duration
            if conversation.started_at:
                duration = conversation.completed_at - conversation.started_at
                conversation.duration_seconds = int(duration.total_seconds())

            # Add closing metadata
            closing_metadata = {
                "reason": reason,
                "closed_by": user_context.get("user_id") if user_context else "system",
                "closed_at": conversation.completed_at.isoformat()
            }

            if summary:
                closing_metadata["summary"] = summary

            conversation.business_context = conversation.business_context or {}
            conversation.business_context.update(closing_metadata)

            updated_conversation = await self.conversation_repo.update(conversation)

            self.log_operation(
                "close_conversation",
                tenant_id=tenant_id,
                conversation_id=conversation_id,
                reason=reason,
                duration_seconds=conversation.duration_seconds
            )

            return updated_conversation

        except (NotFoundError, ConflictError):
            raise
        except Exception as e:
            error = self.handle_service_error(
                e, "close_conversation",
                tenant_id=tenant_id
            )
            raise error

    async def list_conversations(
            self,
            tenant_id: TenantId,
            filters: Optional[Dict[str, Any]] = None,
            limit: int = 50,
            offset: int = 0,
            sort_by: str = "last_activity_at",
            sort_order: str = "desc",
            user_context: Optional[Dict[str, Any]] = None
    ) -> List[ConversationDocument]:
        """
        List conversations with filtering and pagination

        Args:
            tenant_id: Tenant identifier
            filters: Optional filters to apply
            limit: Maximum number of conversations to return
            offset: Number of conversations to skip
            sort_by: Field to sort by
            sort_order: Sort order (asc or desc)
            user_context: User authentication context

        Returns:
            List of conversation documents
        """
        try:
            await self.validate_tenant_access(tenant_id, user_context)

            # Add tenant filter
            search_filters = filters or {}
            search_filters["tenant_id"] = tenant_id

            conversations = await self.conversation_repo.list_conversations(
                search_filters, limit, offset, sort_by, sort_order
            )

            self.log_operation(
                "list_conversations",
                tenant_id=tenant_id,
                filter_count=len(search_filters),
                result_count=len(conversations)
            )

            return conversations

        except Exception as e:
            error = self.handle_service_error(
                e, "list_conversations",
                tenant_id=tenant_id
            )
            raise error

    async def get_conversation_summary(
            self,
            tenant_id: TenantId,
            conversation_id: str,
            user_context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Get conversation summary with metrics

        Args:
            tenant_id: Tenant identifier
            conversation_id: Conversation identifier
            user_context: User authentication context

        Returns:
            Conversation summary dictionary
        """
        try:
            await self.validate_tenant_access(tenant_id, user_context)

            conversation = await self.get_conversation(tenant_id, conversation_id, user_context)
            if not conversation:
                raise NotFoundError(f"Conversation {conversation_id} not found")

            # Get message statistics
            message_stats = await self.message_repo.get_conversation_stats(conversation_id)

            summary = {
                "conversation_id": conversation_id,
                "status": conversation.status.value,
                "channel": conversation.channel.value,
                "started_at": conversation.started_at.isoformat() if conversation.started_at else None,
                "last_activity_at": conversation.last_activity_at.isoformat() if conversation.last_activity_at else None,
                "completed_at": conversation.completed_at.isoformat() if conversation.completed_at else None,
                "duration_seconds": conversation.duration_seconds,
                "message_count": message_stats.get("total_messages", 0),
                "user_messages": message_stats.get("user_messages", 0),
                "bot_messages": message_stats.get("bot_messages", 0),
                "primary_intent": conversation.context.get("current_intent"),
                "resolution_status": conversation.business_context.get("outcome"),
                "user_satisfaction": conversation.metrics.get("user_satisfaction"),
                "escalated": conversation.status == ConversationStatus.ESCALATED
            }

            self.log_operation(
                "get_conversation_summary",
                tenant_id=tenant_id,
                conversation_id=conversation_id
            )

            return summary

        except (NotFoundError, ValidationError):
            raise
        except Exception as e:
            error = self.handle_service_error(
                e, "get_conversation_summary",
                tenant_id=tenant_id
            )
            raise error

    async def update_conversation_context(
            self,
            tenant_id: TenantId,
            conversation_id: str,
            context_updates: Dict[str, Any],
            user_context: Optional[Dict[str, Any]] = None
    ) -> ConversationDocument:
        """
        Update conversation context

        Args:
            tenant_id: Tenant identifier
            conversation_id: Conversation identifier
            context_updates: Context updates to apply
            user_context: User authentication context

        Returns:
            Updated conversation document
        """
        try:
            await self.validate_tenant_access(tenant_id, user_context)

            conversation = await self.get_conversation(tenant_id, conversation_id, user_context)
            if not conversation:
                raise NotFoundError(f"Conversation {conversation_id} not found")

            # Update context
            conversation.context = conversation.context or {}
            conversation.context.update(context_updates)
            conversation.last_activity_at = datetime.utcnow()

            updated_conversation = await self.conversation_repo.update(conversation)

            self.log_operation(
                "update_conversation_context",
                tenant_id=tenant_id,
                conversation_id=conversation_id,
                context_keys=list(context_updates.keys())
            )

            return updated_conversation

        except (NotFoundError, ValidationError):
            raise
        except Exception as e:
            error = self.handle_service_error(
                e, "update_conversation_context",
                tenant_id=tenant_id
            )
            raise error

    async def escalate_conversation(
            self,
            tenant_id: TenantId,
            conversation_id: str,
            reason: str,
            escalation_data: Optional[Dict[str, Any]] = None,
            user_context: Optional[Dict[str, Any]] = None
    ) -> ConversationDocument:
        """
        Escalate conversation to human agent

        Args:
            tenant_id: Tenant identifier
            conversation_id: Conversation identifier
            reason: Reason for escalation
            escalation_data: Optional escalation metadata
            user_context: User authentication context

        Returns:
            Updated conversation document
        """
        try:
            await self.validate_tenant_access(tenant_id, user_context)

            conversation = await self.get_conversation(tenant_id, conversation_id, user_context)
            if not conversation:
                raise NotFoundError(f"Conversation {conversation_id} not found")

            if conversation.status == ConversationStatus.ESCALATED:
                raise ConflictError("Conversation is already escalated")

            # Update conversation status
            conversation.status = ConversationStatus.ESCALATED
            conversation.last_activity_at = datetime.utcnow()

            # Add escalation metadata
            escalation_metadata = {
                "escalated_at": datetime.utcnow().isoformat(),
                "escalation_reason": reason,
                "escalated_by": user_context.get("user_id") if user_context else "system"
            }

            if escalation_data:
                escalation_metadata.update(escalation_data)

            conversation.business_context = conversation.business_context or {}
            conversation.business_context.update(escalation_metadata)

            updated_conversation = await self.conversation_repo.update(conversation)

            self.log_operation(
                "escalate_conversation",
                tenant_id=tenant_id,
                conversation_id=conversation_id,
                reason=reason
            )

            return updated_conversation

        except (NotFoundError, ConflictError):
            raise
        except Exception as e:
            error = self.handle_service_error(
                e, "escalate_conversation",
                tenant_id=tenant_id
            )
            raise error

    async def get_conversation_analytics(
            self,
            tenant_id: TenantId,
            start_date: Optional[datetime] = None,
            end_date: Optional[datetime] = None,
            filters: Optional[Dict[str, Any]] = None,
            user_context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Get conversation analytics for a tenant

        Args:
            tenant_id: Tenant identifier
            start_date: Optional start date for analytics
            end_date: Optional end date for analytics
            filters: Optional additional filters
            user_context: User authentication context

        Returns:
            Analytics data dictionary
        """
        try:
            await self.validate_tenant_access(tenant_id, user_context)

            # Set default date range if not provided
            if not end_date:
                end_date = datetime.utcnow()
            if not start_date:
                start_date = end_date - timedelta(days=30)

            analytics_filters = filters or {}
            analytics_filters["tenant_id"] = tenant_id
            analytics_filters["start_date"] = start_date
            analytics_filters["end_date"] = end_date

            analytics = await self.conversation_repo.get_analytics(analytics_filters)

            self.log_operation(
                "get_conversation_analytics",
                tenant_id=tenant_id,
                start_date=start_date.isoformat(),
                end_date=end_date.isoformat()
            )

            return analytics

        except Exception as e:
            error = self.handle_service_error(
                e, "get_conversation_analytics",
                tenant_id=tenant_id
            )
            raise error

    async def export_conversation(
            self,
            tenant_id: TenantId,
            conversation_id: str,
            format_type: str = "json",
            include_metadata: bool = True,
            user_context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Export conversation data

        Args:
            tenant_id: Tenant identifier
            conversation_id: Conversation identifier
            format_type: Export format (json, csv, etc.)
            include_metadata: Whether to include metadata
            user_context: User authentication context

        Returns:
            Exported conversation data
        """
        try:
            await self.validate_tenant_access(tenant_id, user_context)

            conversation = await self.get_conversation(tenant_id, conversation_id, user_context)
            if not conversation:
                raise NotFoundError(f"Conversation {conversation_id} not found")

            # Get all messages for the conversation
            messages = await self.message_repo.get_conversation_messages(conversation_id)

            export_data = {
                "conversation": conversation.dict() if include_metadata else {
                    "conversation_id": conversation.conversation_id,
                    "started_at": conversation.started_at,
                    "completed_at": conversation.completed_at,
                    "status": conversation.status.value,
                    "channel": conversation.channel.value
                },
                "messages": [
                    msg.dict() if include_metadata else {
                        "message_id": msg.message_id,
                        "timestamp": msg.timestamp,
                        "direction": msg.direction,
                        "content": msg.content
                    }
                    for msg in messages
                ],
                "export_metadata": {
                    "exported_at": datetime.utcnow().isoformat(),
                    "format": format_type,
                    "exported_by": user_context.get("user_id") if user_context else "system"
                }
            }

            self.log_operation(
                "export_conversation",
                tenant_id=tenant_id,
                conversation_id=conversation_id,
                format_type=format_type,
                message_count=len(messages)
            )

            return export_data

        except (NotFoundError, ValidationError):
            raise
        except Exception as e:
            error = self.handle_service_error(
                e, "export_conversation",
                tenant_id=tenant_id
            )
            raise error