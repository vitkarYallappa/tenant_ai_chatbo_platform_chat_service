"""
Conversation Repository Implementation
=====================================

MongoDB repository for conversation document operations with comprehensive
conversation lifecycle management, multi-tenant isolation, and performance optimization.

Features:
- Full CRUD operations for conversations
- Tenant isolation and security
- Advanced querying and filtering
- Conversation state management
- Performance monitoring and caching
- Business analytics support
"""

from datetime import datetime, timedelta
from typing import Optional, List, Dict, Any, Tuple
from motor.motor_asyncio import AsyncIOMotorDatabase, AsyncIOMotorCollection
from pymongo.errors import DuplicateKeyError, ConnectionFailure
from pymongo import ReturnDocument
import structlog

from .base_repository import BaseRepository, Pagination, PaginatedResult
from .exceptions import (
    RepositoryError, EntityNotFoundError, DuplicateEntityError,
    ConnectionError, QueryError, ValidationError
)

# Import models from the models package (these would be defined in phase 2)
# For now, I'll create placeholder classes that match the schema
from dataclasses import dataclass, asdict
from enum import Enum


# Placeholder model classes based on the MongoDB schema
class ConversationStatus(str, Enum):
    ACTIVE = "active"
    COMPLETED = "completed"
    ABANDONED = "abandoned"
    ESCALATED = "escalated"
    ERROR = "error"


class ChannelType(str, Enum):
    WEB = "web"
    WHATSAPP = "whatsapp"
    MESSENGER = "messenger"
    SLACK = "slack"
    TEAMS = "teams"
    VOICE = "voice"
    SMS = "sms"


@dataclass
class ConversationDocument:
    """Conversation document model matching MongoDB schema"""
    conversation_id: str
    tenant_id: str
    user_id: str
    session_id: Optional[str] = None
    channel: ChannelType = ChannelType.WEB
    status: ConversationStatus = ConversationStatus.ACTIVE
    started_at: datetime = None
    last_activity_at: datetime = None
    completed_at: Optional[datetime] = None
    flow_id: Optional[str] = None
    current_state: Optional[str] = None
    context: Dict[str, Any] = None
    metrics: Dict[str, Any] = None
    channel_metadata: Dict[str, Any] = None
    id: Optional[str] = None  # MongoDB ObjectId

    def __post_init__(self):
        if self.started_at is None:
            self.started_at = datetime.utcnow()
        if self.last_activity_at is None:
            self.last_activity_at = datetime.utcnow()
        if self.context is None:
            self.context = {}
        if self.metrics is None:
            self.metrics = {}
        if self.channel_metadata is None:
            self.channel_metadata = {}

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for MongoDB storage"""
        data = asdict(self)
        if data.get('id') is None:
            data.pop('id', None)
        # Convert enum values to strings
        if isinstance(data.get('channel'), ChannelType):
            data['channel'] = data['channel'].value
        if isinstance(data.get('status'), ConversationStatus):
            data['status'] = data['status'].value
        return data

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ConversationDocument":
        """Create from MongoDB document"""
        # Handle MongoDB ObjectId
        if '_id' in data:
            data['id'] = str(data['_id'])
            data.pop('_id', None)

        # Convert string enums back to enum objects
        if 'channel' in data and isinstance(data['channel'], str):
            data['channel'] = ChannelType(data['channel'])
        if 'status' in data and isinstance(data['status'], str):
            data['status'] = ConversationStatus(data['status'])

        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


# Type aliases
TenantId = str
UserId = str
ConversationId = str


class ConversationRepository(BaseRepository[ConversationDocument]):
    """
    Repository for conversation documents in MongoDB

    Provides comprehensive conversation management with:
    - Multi-tenant isolation
    - Advanced filtering and pagination
    - Performance optimization
    - Business analytics support
    - Conversation lifecycle management
    """

    def __init__(self, database: AsyncIOMotorDatabase):
        """
        Initialize conversation repository

        Args:
            database: MongoDB database instance
        """
        super().__init__()
        self.database = database
        self.collection: AsyncIOMotorCollection = database.conversations
        self.logger = structlog.get_logger("ConversationRepository")

    async def create(self, conversation: ConversationDocument) -> ConversationDocument:
        """
        Create a new conversation document

        Args:
            conversation: Conversation to create

        Returns:
            Created conversation with assigned ID

        Raises:
            DuplicateEntityError: If conversation_id already exists
            RepositoryError: If creation fails
        """
        try:
            async with self._timed_operation("create_conversation"):
                # Validate conversation data
                self._validate_conversation(conversation)

                # Prepare document for insertion
                document = conversation.to_dict()
                document['created_at'] = datetime.utcnow()
                document['updated_at'] = datetime.utcnow()

                # Insert document
                result = await self.collection.insert_one(document)
                conversation.id = str(result.inserted_id)

                self._log_operation(
                    "create_conversation",
                    conversation_id=conversation.conversation_id,
                    tenant_id=conversation.tenant_id,
                    channel=conversation.channel.value if isinstance(conversation.channel,
                                                                     ChannelType) else conversation.channel
                )

                return conversation

        except DuplicateKeyError as e:
            raise DuplicateEntityError(
                entity_type="Conversation",
                conflicting_fields={"conversation_id": conversation.conversation_id}
            )
        except Exception as e:
            self._log_error("create_conversation", e, conversation_id=conversation.conversation_id)
            raise RepositoryError(f"Failed to create conversation: {e}", original_error=e)

    async def get_by_id(self, conversation_id: ConversationId) -> Optional[ConversationDocument]:
        """
        Get conversation by conversation_id

        Args:
            conversation_id: Unique conversation identifier

        Returns:
            Conversation document if found, None otherwise

        Raises:
            RepositoryError: If query fails
        """
        try:
            async with self._timed_operation("get_conversation"):
                document = await self.collection.find_one(
                    {"conversation_id": conversation_id}
                )

                if document:
                    conversation = ConversationDocument.from_dict(document)
                    self._log_operation("get_conversation", conversation_id=conversation_id, found=True)
                    return conversation

                self._log_operation("get_conversation", conversation_id=conversation_id, found=False)
                return None

        except Exception as e:
            self._log_error("get_conversation", e, conversation_id=conversation_id)
            raise RepositoryError(f"Failed to get conversation: {e}", original_error=e)

    async def update(self, conversation: ConversationDocument) -> ConversationDocument:
        """
        Update existing conversation

        Args:
            conversation: Conversation with updated data

        Returns:
            Updated conversation document

        Raises:
            EntityNotFoundError: If conversation doesn't exist
            RepositoryError: If update fails
        """
        try:
            async with self._timed_operation("update_conversation"):
                # Validate conversation data
                self._validate_conversation(conversation)

                # Prepare update document
                document = conversation.to_dict()
                document.pop("_id", None)  # Remove ID from update document
                document['updated_at'] = datetime.utcnow()

                # Update document
                result = await self.collection.find_one_and_update(
                    {"conversation_id": conversation.conversation_id},
                    {"$set": document},
                    return_document=ReturnDocument.AFTER
                )

                if result is None:
                    raise EntityNotFoundError("Conversation", conversation.conversation_id)

                updated_conversation = ConversationDocument.from_dict(result)

                self._log_operation(
                    "update_conversation",
                    conversation_id=conversation.conversation_id,
                    status=conversation.status
                )

                return updated_conversation

        except EntityNotFoundError:
            raise
        except Exception as e:
            self._log_error("update_conversation", e, conversation_id=conversation.conversation_id)
            raise RepositoryError(f"Failed to update conversation: {e}", original_error=e)

    async def delete(self, conversation_id: ConversationId) -> bool:
        """
        Delete conversation by ID (soft delete by setting status)

        Args:
            conversation_id: Conversation ID to delete

        Returns:
            True if deleted, False if not found

        Raises:
            RepositoryError: If deletion fails
        """
        try:
            async with self._timed_operation("delete_conversation"):
                # Soft delete by updating status
                result = await self.collection.update_one(
                    {"conversation_id": conversation_id},
                    {
                        "$set": {
                            "status": ConversationStatus.ABANDONED.value,
                            "deleted_at": datetime.utcnow(),
                            "updated_at": datetime.utcnow()
                        }
                    }
                )

                success = result.modified_count > 0
                self._log_operation(
                    "delete_conversation",
                    conversation_id=conversation_id,
                    deleted=success
                )

                return success

        except Exception as e:
            self._log_error("delete_conversation", e, conversation_id=conversation_id)
            raise RepositoryError(f"Failed to delete conversation: {e}", original_error=e)

    async def list(
            self,
            filters: Optional[Dict[str, Any]] = None,
            pagination: Optional[Pagination] = None
    ) -> PaginatedResult[ConversationDocument]:
        """
        List conversations with filters and pagination

        Args:
            filters: Query filters
            pagination: Pagination parameters

        Returns:
            Paginated result set

        Raises:
            RepositoryError: If query fails
        """
        try:
            async with self._timed_operation("list_conversations"):
                filters = filters or {}
                pagination = await self._validate_pagination(pagination)

                # Build MongoDB query
                query = self._build_query(filters)
                sort_criteria = self._build_sort_criteria(pagination)

                # Get total count
                total = await self.collection.count_documents(query)

                # Get paginated results
                cursor = self.collection.find(query)
                cursor = cursor.sort(sort_criteria)
                cursor = cursor.skip(pagination.offset)
                cursor = cursor.limit(pagination.page_size)

                documents = await cursor.to_list(length=pagination.page_size)
                conversations = [
                    ConversationDocument.from_dict(doc) for doc in documents
                ]

                self._log_operation(
                    "list_conversations",
                    filters=filters,
                    total=total,
                    returned=len(conversations)
                )

                return PaginatedResult.create(conversations, total, pagination)

        except Exception as e:
            self._log_error("list_conversations", e, filters=filters)
            raise RepositoryError(f"Failed to list conversations: {e}", original_error=e)

    async def exists(self, conversation_id: ConversationId) -> bool:
        """
        Check if conversation exists

        Args:
            conversation_id: Conversation ID to check

        Returns:
            True if conversation exists
        """
        try:
            count = await self.collection.count_documents(
                {"conversation_id": conversation_id}, limit=1
            )
            return count > 0
        except Exception as e:
            self._log_error("check_conversation_exists", e, conversation_id=conversation_id)
            return False

    async def count(self, filters: Optional[Dict[str, Any]] = None) -> int:
        """
        Count conversations matching filters

        Args:
            filters: Query filters

        Returns:
            Number of matching conversations

        Raises:
            RepositoryError: If query fails
        """
        try:
            query = self._build_query(filters or {})
            return await self.collection.count_documents(query)
        except Exception as e:
            self._log_error("count_conversations", e, filters=filters)
            raise RepositoryError(f"Failed to count conversations: {e}", original_error=e)

    # Conversation-specific methods

    async def get_active_conversations(
            self,
            tenant_id: TenantId,
            user_id: UserId,
            limit: int = 10
    ) -> List[ConversationDocument]:
        """
        Get active conversations for a user

        Args:
            tenant_id: Tenant identifier
            user_id: User identifier
            limit: Maximum number of conversations to return

        Returns:
            List of active conversations sorted by last activity
        """
        try:
            query = {
                "tenant_id": tenant_id,
                "user_id": user_id,
                "status": ConversationStatus.ACTIVE.value
            }

            cursor = self.collection.find(query)
            cursor = cursor.sort("last_activity_at", -1)
            cursor = cursor.limit(limit)

            documents = await cursor.to_list(length=limit)
            conversations = [ConversationDocument.from_dict(doc) for doc in documents]

            self._log_operation(
                "get_active_conversations",
                tenant_id=tenant_id,
                user_id=user_id,
                count=len(conversations)
            )

            return conversations

        except Exception as e:
            self._log_error(
                "get_active_conversations",
                e,
                tenant_id=tenant_id,
                user_id=user_id
            )
            raise RepositoryError(f"Failed to get active conversations: {e}", original_error=e)

    async def update_conversation_status(
            self,
            conversation_id: ConversationId,
            status: ConversationStatus,
            completed_at: Optional[datetime] = None
    ) -> bool:
        """
        Update conversation status

        Args:
            conversation_id: Conversation ID to update
            status: New status
            completed_at: Completion timestamp for completed conversations

        Returns:
            True if updated successfully
        """
        try:
            update_doc = {
                "status": status.value if isinstance(status, ConversationStatus) else status,
                "updated_at": datetime.utcnow()
            }

            if status == ConversationStatus.COMPLETED and completed_at:
                update_doc["completed_at"] = completed_at

            result = await self.collection.update_one(
                {"conversation_id": conversation_id},
                {"$set": update_doc}
            )

            success = result.modified_count > 0
            self._log_operation(
                "update_conversation_status",
                conversation_id=conversation_id,
                status=status,
                success=success
            )

            return success

        except Exception as e:
            self._log_error(
                "update_conversation_status",
                e,
                conversation_id=conversation_id,
                status=status
            )
            raise RepositoryError(f"Failed to update conversation status: {e}", original_error=e)

    async def get_conversations_by_tenant(
            self,
            tenant_id: TenantId,
            filters: Optional[Dict[str, Any]] = None,
            pagination: Optional[Pagination] = None
    ) -> PaginatedResult[ConversationDocument]:
        """
        Get conversations for a specific tenant

        Args:
            tenant_id: Tenant identifier
            filters: Additional filters
            pagination: Pagination parameters

        Returns:
            Paginated conversations for the tenant
        """
        base_filters = {"tenant_id": tenant_id}
        if filters:
            base_filters.update(filters)

        return await self.list(filters=base_filters, pagination=pagination)

    async def update_last_activity(self, conversation_id: ConversationId) -> bool:
        """
        Update last activity timestamp for a conversation

        Args:
            conversation_id: Conversation ID to update

        Returns:
            True if updated successfully
        """
        try:
            result = await self.collection.update_one(
                {"conversation_id": conversation_id},
                {
                    "$set": {
                        "last_activity_at": datetime.utcnow(),
                        "updated_at": datetime.utcnow()
                    }
                }
            )

            return result.modified_count > 0

        except Exception as e:
            self._log_error("update_last_activity", e, conversation_id=conversation_id)
            raise RepositoryError(f"Failed to update last activity: {e}", original_error=e)

    async def get_conversations_by_channel(
            self,
            tenant_id: TenantId,
            channel: ChannelType,
            start_date: Optional[datetime] = None,
            end_date: Optional[datetime] = None,
            pagination: Optional[Pagination] = None
    ) -> PaginatedResult[ConversationDocument]:
        """
        Get conversations by channel within date range

        Args:
            tenant_id: Tenant identifier
            channel: Channel type
            start_date: Start date for filtering
            end_date: End date for filtering
            pagination: Pagination parameters

        Returns:
            Paginated conversations for the channel
        """
        filters = {
            "tenant_id": tenant_id,
            "channel": channel.value if isinstance(channel, ChannelType) else channel
        }

        if start_date or end_date:
            date_filter = {}
            if start_date:
                date_filter["$gte"] = start_date
            if end_date:
                date_filter["$lte"] = end_date
            filters["started_at"] = date_filter

        return await self.list(filters=filters, pagination=pagination)

    # Private helper methods

    def _validate_conversation(self, conversation: ConversationDocument) -> None:
        """
        Validate conversation data before database operations

        Args:
            conversation: Conversation to validate

        Raises:
            ValidationError: If validation fails
        """
        if not conversation.conversation_id:
            raise ValidationError("conversation_id is required", field="conversation_id")

        if not conversation.tenant_id:
            raise ValidationError("tenant_id is required", field="tenant_id")

        if not conversation.user_id:
            raise ValidationError("user_id is required", field="user_id")

        # Validate enum values
        if isinstance(conversation.channel, str):
            try:
                ChannelType(conversation.channel)
            except ValueError:
                raise ValidationError(f"Invalid channel: {conversation.channel}", field="channel")

        if isinstance(conversation.status, str):
            try:
                ConversationStatus(conversation.status)
            except ValueError:
                raise ValidationError(f"Invalid status: {conversation.status}", field="status")

    def _build_query(self, filters: Dict[str, Any]) -> Dict[str, Any]:
        """
        Build MongoDB query from filters

        Args:
            filters: Filter dictionary

        Returns:
            MongoDB query document
        """
        query = {}

        # Direct field mappings
        direct_fields = [
            "tenant_id", "user_id", "conversation_id", "status", "channel",
            "flow_id", "current_state", "session_id"
        ]
        for field in direct_fields:
            if field in filters:
                query[field] = filters[field]

        # Date range filters
        if "start_date" in filters or "end_date" in filters:
            date_query = {}
            if "start_date" in filters:
                date_query["$gte"] = filters["start_date"]
            if "end_date" in filters:
                date_query["$lte"] = filters["end_date"]
            query["started_at"] = date_query

        # Activity date range
        if "activity_start" in filters or "activity_end" in filters:
            activity_query = {}
            if "activity_start" in filters:
                activity_query["$gte"] = filters["activity_start"]
            if "activity_end" in filters:
                activity_query["$lte"] = filters["activity_end"]
            query["last_activity_at"] = activity_query

        # Text search (simplified)
        if "search" in filters:
            query["$text"] = {"$search": filters["search"]}

        # Status list filter
        if "status_list" in filters:
            query["status"] = {"$in": filters["status_list"]}

        # Channel list filter
        if "channel_list" in filters:
            query["channel"] = {"$in": filters["channel_list"]}

        # Exclude deleted conversations by default
        if "include_deleted" not in filters or not filters["include_deleted"]:
            query["status"] = {"$ne": "deleted"}

        return query


# Database connection management
_mongodb_database: Optional[AsyncIOMotorDatabase] = None


async def get_mongodb_database() -> AsyncIOMotorDatabase:
    """
    Get MongoDB database instance

    Returns:
        MongoDB database instance

    Raises:
        ConnectionError: If database connection fails
    """
    global _mongodb_database

    if _mongodb_database is None:
        try:
            from motor.motor_asyncio import AsyncIOMotorClient
            # This would normally come from configuration
            client = AsyncIOMotorClient("mongodb://localhost:27017")
            _mongodb_database = client.chatbot_platform
        except Exception as e:
            raise ConnectionError("MongoDB", original_error=e)

    return _mongodb_database


# Dependency injection function
async def get_conversation_repository() -> ConversationRepository:
    """
    Get conversation repository instance for dependency injection

    Returns:
        ConversationRepository instance
    """
    database = await get_mongodb_database()
    return ConversationRepository(database)