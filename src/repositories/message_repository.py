"""
Message Repository Implementation
================================

MongoDB repository for message document operations with comprehensive
message lifecycle management, content analysis, and performance optimization.

Features:
- Full CRUD operations for messages
- Conversation-based message retrieval
- Content analysis and metadata
- Multi-tenant isolation
- Performance optimization with indexing
- Message threading and sequence management
"""

from datetime import datetime, timedelta
from typing import Optional, List, Dict, Any, Tuple
from motor.motor_asyncio import AsyncIOMotorDatabase, AsyncIOMotorCollection
from pymongo.errors import DuplicateKeyError, ConnectionFailure
from pymongo import ReturnDocument, ASCENDING, DESCENDING
import structlog

from .base_repository import BaseRepository, Pagination, PaginatedResult
from .exceptions import (
    RepositoryError, EntityNotFoundError, DuplicateEntityError,
    ConnectionError, QueryError, ValidationError
)

# Import models (placeholder implementation)
from dataclasses import dataclass, asdict, field
from enum import Enum


# Type definitions
class MessageType(str, Enum):
    TEXT = "text"
    IMAGE = "image"
    FILE = "file"
    AUDIO = "audio"
    VIDEO = "video"
    LOCATION = "location"
    QUICK_REPLY = "quick_reply"
    CAROUSEL = "carousel"
    FORM = "form"
    SYSTEM = "system"


class MessageDirection(str, Enum):
    INBOUND = "inbound"
    OUTBOUND = "outbound"


class DeliveryStatus(str, Enum):
    SENT = "sent"
    DELIVERED = "delivered"
    READ = "read"
    FAILED = "failed"


@dataclass
class MediaContent:
    """Media content structure"""
    url: str
    type: str  # MIME type
    size_bytes: int
    duration_ms: Optional[int] = None
    dimensions: Optional[Dict[str, int]] = None
    thumbnail_url: Optional[str] = None
    alt_text: Optional[str] = None
    caption: Optional[str] = None


@dataclass
class MessageContent:
    """Message content structure"""
    type: MessageType
    text: Optional[str] = None
    language: str = "en"
    media: Optional[MediaContent] = None
    location: Optional[Dict[str, Any]] = None
    quick_replies: Optional[List[Dict[str, Any]]] = None
    buttons: Optional[List[Dict[str, Any]]] = None
    carousel: Optional[List[Dict[str, Any]]] = None
    form: Optional[Dict[str, Any]] = None


@dataclass
class AIAnalysis:
    """AI analysis results for messages"""
    intent: Optional[Dict[str, Any]] = None
    entities: Optional[List[Dict[str, Any]]] = None
    sentiment: Optional[Dict[str, Any]] = None
    topics: Optional[List[str]] = None
    keywords: Optional[List[str]] = None
    toxicity: Optional[Dict[str, Any]] = None
    quality: Optional[Dict[str, Any]] = None


@dataclass
class GenerationMetadata:
    """Metadata for generated messages"""
    model_provider: Optional[str] = None
    model_name: Optional[str] = None
    model_version: Optional[str] = None
    generation_config: Optional[Dict[str, Any]] = None
    tokens_used: Optional[Dict[str, int]] = None
    cost_cents: Optional[float] = None
    generation_time_ms: Optional[int] = None
    fallback_used: Optional[bool] = None
    template_used: Optional[str] = None


@dataclass
class MessageDocument:
    """Message document model matching MongoDB schema"""
    message_id: str
    conversation_id: str
    tenant_id: str
    user_id: str
    sequence_number: int
    direction: MessageDirection
    timestamp: datetime
    channel: str
    content: MessageContent
    channel_metadata: Optional[Dict[str, Any]] = None
    ai_analysis: Optional[AIAnalysis] = None
    generation_metadata: Optional[GenerationMetadata] = None
    processing: Optional[Dict[str, Any]] = None
    quality_assurance: Optional[Dict[str, Any]] = None
    moderation: Optional[Dict[str, Any]] = None
    privacy: Optional[Dict[str, Any]] = None
    id: Optional[str] = None  # MongoDB ObjectId

    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.utcnow()
        if self.channel_metadata is None:
            self.channel_metadata = {}
        if self.processing is None:
            self.processing = {}

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for MongoDB storage"""
        data = {}

        # Handle dataclass conversion
        for key, value in asdict(self).items():
            if key == 'id' and value is None:
                continue
            if key == 'content' and isinstance(value, dict):
                # Convert content dataclass
                data[key] = value
            elif key in ['ai_analysis', 'generation_metadata'] and value:
                # Convert nested dataclasses
                data[key] = value
            else:
                data[key] = value

        # Convert enums to strings
        if 'direction' in data and hasattr(data['direction'], 'value'):
            data['direction'] = data['direction'].value
        if 'content' in data and 'type' in data['content'] and hasattr(data['content']['type'], 'value'):
            data['content']['type'] = data['content']['type'].value

        return data

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "MessageDocument":
        """Create from MongoDB document"""
        # Handle MongoDB ObjectId
        if '_id' in data:
            data['id'] = str(data['_id'])
            data.pop('_id', None)

        # Convert string enums back to enum objects
        if 'direction' in data and isinstance(data['direction'], str):
            data['direction'] = MessageDirection(data['direction'])

        # Handle content structure
        if 'content' in data and isinstance(data['content'], dict):
            content_data = data['content'].copy()
            if 'type' in content_data and isinstance(content_data['type'], str):
                content_data['type'] = MessageType(content_data['type'])

            # Convert media if present
            if 'media' in content_data and content_data['media']:
                content_data['media'] = MediaContent(**content_data['media'])

            data['content'] = MessageContent(**content_data)

        # Handle AI analysis
        if 'ai_analysis' in data and data['ai_analysis']:
            data['ai_analysis'] = AIAnalysis(**data['ai_analysis'])

        # Handle generation metadata
        if 'generation_metadata' in data and data['generation_metadata']:
            data['generation_metadata'] = GenerationMetadata(**data['generation_metadata'])

        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


# Type aliases
TenantId = str
UserId = str
ConversationId = str
MessageId = str


class MessageRepository(BaseRepository[MessageDocument]):
    """
    Repository for message documents in MongoDB

    Provides comprehensive message management with:
    - Multi-tenant isolation
    - Conversation-based organization
    - Sequence management
    - Content analysis support
    - Performance optimization
    """

    def __init__(self, database: AsyncIOMotorDatabase):
        """
        Initialize message repository

        Args:
            database: MongoDB database instance
        """
        super().__init__()
        self.database = database
        self.collection: AsyncIOMotorCollection = database.messages
        self.logger = structlog.get_logger("MessageRepository")

    async def create(self, message: MessageDocument) -> MessageDocument:
        """
        Create a new message document

        Args:
            message: Message to create

        Returns:
            Created message with assigned ID

        Raises:
            DuplicateEntityError: If message_id already exists
            RepositoryError: If creation fails
        """
        try:
            async with self._timed_operation("create_message"):
                # Validate message data
                self._validate_message(message)

                # Auto-assign sequence number if not provided
                if message.sequence_number == 0:
                    message.sequence_number = await self._get_next_sequence_number(
                        message.conversation_id
                    )

                # Prepare document for insertion
                document = message.to_dict()
                document['created_at'] = datetime.utcnow()
                document['updated_at'] = datetime.utcnow()

                # Insert document
                result = await self.collection.insert_one(document)
                message.id = str(result.inserted_id)

                self._log_operation(
                    "create_message",
                    message_id=message.message_id,
                    conversation_id=message.conversation_id,
                    tenant_id=message.tenant_id,
                    direction=message.direction.value,
                    type=message.content.type.value
                )

                return message

        except DuplicateKeyError as e:
            raise DuplicateEntityError(
                entity_type="Message",
                conflicting_fields={"message_id": message.message_id}
            )
        except Exception as e:
            self._log_error("create_message", e, message_id=message.message_id)
            raise RepositoryError(f"Failed to create message: {e}", original_error=e)

    async def get_by_id(self, message_id: MessageId) -> Optional[MessageDocument]:
        """
        Get message by message_id

        Args:
            message_id: Unique message identifier

        Returns:
            Message document if found, None otherwise
        """
        try:
            async with self._timed_operation("get_message"):
                document = await self.collection.find_one(
                    {"message_id": message_id}
                )

                if document:
                    message = MessageDocument.from_dict(document)
                    self._log_operation("get_message", message_id=message_id, found=True)
                    return message

                self._log_operation("get_message", message_id=message_id, found=False)
                return None

        except Exception as e:
            self._log_error("get_message", e, message_id=message_id)
            raise RepositoryError(f"Failed to get message: {e}", original_error=e)

    async def update(self, message: MessageDocument) -> MessageDocument:
        """
        Update existing message

        Args:
            message: Message with updated data

        Returns:
            Updated message document

        Raises:
            EntityNotFoundError: If message doesn't exist
            RepositoryError: If update fails
        """
        try:
            async with self._timed_operation("update_message"):
                # Validate message data
                self._validate_message(message)

                # Prepare update document
                document = message.to_dict()
                document.pop("_id", None)  # Remove ID from update document
                document['updated_at'] = datetime.utcnow()

                # Update document
                result = await self.collection.find_one_and_update(
                    {"message_id": message.message_id},
                    {"$set": document},
                    return_document=ReturnDocument.AFTER
                )

                if result is None:
                    raise EntityNotFoundError("Message", message.message_id)

                updated_message = MessageDocument.from_dict(result)

                self._log_operation(
                    "update_message",
                    message_id=message.message_id,
                    conversation_id=message.conversation_id
                )

                return updated_message

        except EntityNotFoundError:
            raise
        except Exception as e:
            self._log_error("update_message", e, message_id=message.message_id)
            raise RepositoryError(f"Failed to update message: {e}", original_error=e)

    async def delete(self, message_id: MessageId) -> bool:
        """
        Delete message by ID (soft delete by setting flag)

        Args:
            message_id: Message ID to delete

        Returns:
            True if deleted, False if not found
        """
        try:
            async with self._timed_operation("delete_message"):
                # Soft delete by adding deletion metadata
                result = await self.collection.update_one(
                    {"message_id": message_id},
                    {
                        "$set": {
                            "deleted": True,
                            "deleted_at": datetime.utcnow(),
                            "updated_at": datetime.utcnow()
                        }
                    }
                )

                success = result.modified_count > 0
                self._log_operation(
                    "delete_message",
                    message_id=message_id,
                    deleted=success
                )

                return success

        except Exception as e:
            self._log_error("delete_message", e, message_id=message_id)
            raise RepositoryError(f"Failed to delete message: {e}", original_error=e)

    async def list(
            self,
            filters: Optional[Dict[str, Any]] = None,
            pagination: Optional[Pagination] = None
    ) -> PaginatedResult[MessageDocument]:
        """
        List messages with filters and pagination

        Args:
            filters: Query filters
            pagination: Pagination parameters

        Returns:
            Paginated result set
        """
        try:
            async with self._timed_operation("list_messages"):
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
                messages = [
                    MessageDocument.from_dict(doc) for doc in documents
                ]

                self._log_operation(
                    "list_messages",
                    filters=filters,
                    total=total,
                    returned=len(messages)
                )

                return PaginatedResult.create(messages, total, pagination)

        except Exception as e:
            self._log_error("list_messages", e, filters=filters)
            raise RepositoryError(f"Failed to list messages: {e}", original_error=e)

    async def exists(self, message_id: MessageId) -> bool:
        """Check if message exists"""
        try:
            count = await self.collection.count_documents(
                {"message_id": message_id}, limit=1
            )
            return count > 0
        except Exception as e:
            self._log_error("check_message_exists", e, message_id=message_id)
            return False

    async def count(self, filters: Optional[Dict[str, Any]] = None) -> int:
        """Count messages matching filters"""
        try:
            query = self._build_query(filters or {})
            return await self.collection.count_documents(query)
        except Exception as e:
            self._log_error("count_messages", e, filters=filters)
            raise RepositoryError(f"Failed to count messages: {e}", original_error=e)

    # Message-specific methods

    async def get_conversation_messages(
            self,
            conversation_id: ConversationId,
            tenant_id: Optional[TenantId] = None,
            pagination: Optional[Pagination] = None,
            include_system: bool = False
    ) -> PaginatedResult[MessageDocument]:
        """
        Get all messages in a conversation

        Args:
            conversation_id: Conversation identifier
            tenant_id: Tenant identifier for isolation
            pagination: Pagination parameters
            include_system: Include system messages

        Returns:
            Paginated messages ordered by sequence number
        """
        filters = {"conversation_id": conversation_id}

        if tenant_id:
            filters["tenant_id"] = tenant_id

        if not include_system:
            filters["content.type"] = {"$ne": MessageType.SYSTEM.value}

        # Default sort by sequence number for conversation messages
        if pagination is None:
            pagination = Pagination(sort_by="sequence_number", sort_order="asc")
        elif pagination.sort_by is None:
            pagination.sort_by = "sequence_number"
            pagination.sort_order = "asc"

        return await self.list(filters=filters, pagination=pagination)

    async def get_latest_messages(
            self,
            conversation_id: ConversationId,
            limit: int = 10,
            tenant_id: Optional[TenantId] = None
    ) -> List[MessageDocument]:
        """
        Get latest messages in a conversation

        Args:
            conversation_id: Conversation identifier
            limit: Number of messages to return
            tenant_id: Tenant identifier for isolation

        Returns:
            Latest messages ordered by sequence number descending
        """
        try:
            query = {"conversation_id": conversation_id}
            if tenant_id:
                query["tenant_id"] = tenant_id

            # Exclude deleted messages
            query["deleted"] = {"$ne": True}

            cursor = self.collection.find(query)
            cursor = cursor.sort("sequence_number", DESCENDING)
            cursor = cursor.limit(limit)

            documents = await cursor.to_list(length=limit)
            messages = [MessageDocument.from_dict(doc) for doc in documents]

            # Reverse to get chronological order
            messages.reverse()

            self._log_operation(
                "get_latest_messages",
                conversation_id=conversation_id,
                count=len(messages)
            )

            return messages

        except Exception as e:
            self._log_error(
                "get_latest_messages",
                e,
                conversation_id=conversation_id
            )
            raise RepositoryError(f"Failed to get latest messages: {e}", original_error=e)

    async def get_messages_by_direction(
            self,
            conversation_id: ConversationId,
            direction: MessageDirection,
            tenant_id: Optional[TenantId] = None,
            pagination: Optional[Pagination] = None
    ) -> PaginatedResult[MessageDocument]:
        """
        Get messages by direction (inbound/outbound)

        Args:
            conversation_id: Conversation identifier
            direction: Message direction
            tenant_id: Tenant identifier
            pagination: Pagination parameters

        Returns:
            Paginated messages for the specified direction
        """
        filters = {
            "conversation_id": conversation_id,
            "direction": direction.value
        }

        if tenant_id:
            filters["tenant_id"] = tenant_id

        return await self.list(filters=filters, pagination=pagination)

    async def search_messages(
            self,
            tenant_id: TenantId,
            search_query: str,
            conversation_id: Optional[ConversationId] = None,
            message_type: Optional[MessageType] = None,
            date_from: Optional[datetime] = None,
            date_to: Optional[datetime] = None,
            pagination: Optional[Pagination] = None
    ) -> PaginatedResult[MessageDocument]:
        """
        Search messages with text search and filters

        Args:
            tenant_id: Tenant identifier
            search_query: Text to search for
            conversation_id: Optional conversation filter
            message_type: Optional message type filter
            date_from: Optional start date
            date_to: Optional end date
            pagination: Pagination parameters

        Returns:
            Paginated search results
        """
        filters = {
            "tenant_id": tenant_id,
            "$text": {"$search": search_query}
        }

        if conversation_id:
            filters["conversation_id"] = conversation_id

        if message_type:
            filters["content.type"] = message_type.value

        if date_from or date_to:
            date_filter = {}
            if date_from:
                date_filter["$gte"] = date_from
            if date_to:
                date_filter["$lte"] = date_to
            filters["timestamp"] = date_filter

        return await self.list(filters=filters, pagination=pagination)

    async def update_ai_analysis(
            self,
            message_id: MessageId,
            ai_analysis: AIAnalysis
    ) -> bool:
        """
        Update AI analysis for a message

        Args:
            message_id: Message identifier
            ai_analysis: AI analysis results

        Returns:
            True if updated successfully
        """
        try:
            result = await self.collection.update_one(
                {"message_id": message_id},
                {
                    "$set": {
                        "ai_analysis": asdict(ai_analysis),
                        "updated_at": datetime.utcnow()
                    }
                }
            )

            success = result.modified_count > 0
            self._log_operation(
                "update_ai_analysis",
                message_id=message_id,
                success=success
            )

            return success

        except Exception as e:
            self._log_error("update_ai_analysis", e, message_id=message_id)
            raise RepositoryError(f"Failed to update AI analysis: {e}", original_error=e)

    async def get_messages_by_intent(
            self,
            tenant_id: TenantId,
            intent: str,
            confidence_threshold: float = 0.7,
            pagination: Optional[Pagination] = None
    ) -> PaginatedResult[MessageDocument]:
        """
        Get messages by detected intent

        Args:
            tenant_id: Tenant identifier
            intent: Intent name to filter by
            confidence_threshold: Minimum confidence score
            pagination: Pagination parameters

        Returns:
            Paginated messages with the specified intent
        """
        filters = {
            "tenant_id": tenant_id,
            "ai_analysis.intent.detected_intent": intent,
            "ai_analysis.intent.confidence": {"$gte": confidence_threshold}
        }

        return await self.list(filters=filters, pagination=pagination)

    # Private helper methods

    async def _get_next_sequence_number(self, conversation_id: ConversationId) -> int:
        """Get the next sequence number for a conversation"""
        try:
            # Find the highest sequence number in the conversation
            pipeline = [
                {"$match": {"conversation_id": conversation_id}},
                {"$group": {"_id": None, "max_seq": {"$max": "$sequence_number"}}}
            ]

            result = await self.collection.aggregate(pipeline).to_list(length=1)

            if result and result[0]["max_seq"] is not None:
                return result[0]["max_seq"] + 1
            else:
                return 1

        except Exception as e:
            self.logger.warning(
                "Failed to get next sequence number, defaulting to 1",
                conversation_id=conversation_id,
                error=str(e)
            )
            return 1

    def _validate_message(self, message: MessageDocument) -> None:
        """Validate message data before database operations"""
        if not message.message_id:
            raise ValidationError("message_id is required", field="message_id")

        if not message.conversation_id:
            raise ValidationError("conversation_id is required", field="conversation_id")

        if not message.tenant_id:
            raise ValidationError("tenant_id is required", field="tenant_id")

        if not message.user_id:
            raise ValidationError("user_id is required", field="user_id")

        if not message.content:
            raise ValidationError("content is required", field="content")

        # Validate content based on type
        if message.content.type == MessageType.TEXT and not message.content.text:
            raise ValidationError("text content required for text messages", field="content.text")

        if message.content.type in [MessageType.IMAGE, MessageType.AUDIO, MessageType.VIDEO, MessageType.FILE]:
            if not message.content.media:
                raise ValidationError(f"media content required for {message.content.type} messages",
                                      field="content.media")

    def _build_query(self, filters: Dict[str, Any]) -> Dict[str, Any]:
        """Build MongoDB query from filters"""
        query = {}

        # Direct field mappings
        direct_fields = [
            "tenant_id", "user_id", "conversation_id", "message_id",
            "direction", "channel", "sequence_number"
        ]
        for field in direct_fields:
            if field in filters:
                query[field] = filters[field]

        # Message type filter
        if "message_type" in filters:
            query["content.type"] = filters["message_type"]

        # Date range filters
        if "start_date" in filters or "end_date" in filters:
            date_query = {}
            if "start_date" in filters:
                date_query["$gte"] = filters["start_date"]
            if "end_date" in filters:
                date_query["$lte"] = filters["end_date"]
            query["timestamp"] = date_query

        # Text search
        if "search" in filters:
            query["$text"] = {"$search": filters["search"]}

        # Content text search
        if "content_search" in filters:
            query["content.text"] = {"$regex": filters["content_search"], "$options": "i"}

        # AI analysis filters
        if "intent" in filters:
            query["ai_analysis.intent.detected_intent"] = filters["intent"]

        if "sentiment" in filters:
            query["ai_analysis.sentiment.label"] = filters["sentiment"]

        # Exclude deleted messages by default
        if "include_deleted" not in filters or not filters["include_deleted"]:
            query["deleted"] = {"$ne": True}

        return query


# Dependency injection function
async def get_message_repository() -> MessageRepository:
    """
    Get message repository instance for dependency injection

    Returns:
        MessageRepository instance
    """
    from .conversation_repository import get_mongodb_database
    database = await get_mongodb_database()
    return MessageRepository(database)