"""
Message Validation Models
Pydantic models for request/response validation in chat endpoints.
"""

from datetime import datetime
from typing import Optional, Dict, Any, List
from pydantic import BaseModel, Field, validator
from uuid import UUID, uuid4

from src.models.types import (
    MessageContent, ChannelType, ChannelMetadata, ProcessingHints,
    TenantId, UserId, ConversationId, MessageId
)


class SendMessageRequest(BaseModel):
    """Request model for sending chat messages"""

    # Message identification
    message_id: MessageId = Field(default_factory=lambda: str(uuid4()))
    conversation_id: Optional[ConversationId] = None
    user_id: UserId
    session_id: Optional[str] = None
    tenant_id: Optional[TenantId] = None  # Set by middleware

    # Channel information
    channel: ChannelType
    channel_metadata: Optional[ChannelMetadata] = None

    # Message content
    content: MessageContent

    # Processing hints
    processing_hints: Optional[ProcessingHints] = None

    # Timestamp
    timestamp: datetime = Field(default_factory=datetime.utcnow)

    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }
        schema_extra = {
            "example": {
                "message_id": "msg_123e4567-e89b-12d3-a456-426614174000",
                "user_id": "user_12345",
                "channel": "web",
                "content": {
                    "type": "text",
                    "text": "Hello, I need help with my order",
                    "language": "en"
                },
                "channel_metadata": {
                    "platform_user_id": "web_user_123",
                    "additional_data": {
                        "user_agent": "Mozilla/5.0...",
                        "ip_address": "192.168.1.1"
                    }
                },
                "processing_hints": {
                    "priority": "normal",
                    "expected_response_type": "text",
                    "bypass_automation": False
                }
            }
        }

    @validator('user_id')
    def validate_user_id(cls, v):
        if not v or len(v.strip()) == 0:
            raise ValueError('User ID cannot be empty')
        if len(v) > 255:
            raise ValueError('User ID too long (max 255 characters)')
        return v.strip()

    @validator('content')
    def validate_content(cls, v):
        if not v:
            raise ValueError('Message content is required')

        # Additional content validation based on type
        if v.type.value == "text" and not v.text:
            raise ValueError('Text content is required for text messages')

        if v.type.value in ["image", "video", "audio", "file"] and not v.media:
            raise ValueError(f'Media content is required for {v.type} messages')

        if v.type.value == "location" and not v.location:
            raise ValueError('Location content is required for location messages')

        return v


class ConversationContext(BaseModel):
    """Conversation context in response"""
    current_intent: Optional[str] = None
    entities: Dict[str, Any] = Field(default_factory=dict)
    slots: Dict[str, Any] = Field(default_factory=dict)
    conversation_stage: Optional[str] = None
    next_expected_inputs: List[str] = Field(default_factory=list)


class ProcessingMetadata(BaseModel):
    """Processing metadata in response"""
    processing_time_ms: int
    model_used: Optional[str] = None
    model_provider: Optional[str] = None
    cost_cents: Optional[float] = None
    fallback_applied: bool = False
    confidence_score: Optional[float] = None


class MessageResponse(BaseModel):
    """Response model for processed messages"""

    # Message identification
    message_id: MessageId
    conversation_id: ConversationId

    # Bot response
    response: MessageContent

    # Conversation state
    conversation_state: ConversationContext

    # Processing metadata
    processing_metadata: ProcessingMetadata

    # Response metadata
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    response_id: str = Field(default_factory=lambda: str(uuid4()))

    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }
        schema_extra = {
            "example": {
                "message_id": "msg_123e4567-e89b-12d3-a456-426614174000",
                "conversation_id": "conv_123e4567-e89b-12d3-a456-426614174001",
                "response": {
                    "type": "text",
                    "text": "I'd be happy to help you with your order. Could you please provide your order number?",
                    "language": "en",
                    "quick_replies": [
                        {
                            "title": "I have my order number",
                            "payload": "provide_order_number"
                        },
                        {
                            "title": "I don't have it",
                            "payload": "no_order_number"
                        }
                    ]
                },
                "conversation_state": {
                    "current_intent": "order_inquiry",
                    "entities": {
                        "inquiry_type": "order_status"
                    },
                    "conversation_stage": "information_gathering",
                    "next_expected_inputs": ["order_number"]
                },
                "processing_metadata": {
                    "processing_time_ms": 287,
                    "model_used": "gpt-4-turbo",
                    "model_provider": "openai",
                    "cost_cents": 1.25,
                    "confidence_score": 0.92
                }
            }
        }


class ConversationSummary(BaseModel):
    """Summary information for conversation list"""
    conversation_id: ConversationId
    user_id: UserId
    channel: ChannelType
    status: str
    started_at: datetime
    last_activity_at: datetime
    message_count: int
    primary_intent: Optional[str] = None

    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }


class ConversationHistoryResponse(BaseModel):
    """Response model for conversation history"""
    conversation_id: ConversationId
    summary: ConversationSummary
    messages: List[Dict[str, Any]]

    # Pagination
    page: int
    page_size: int
    total_messages: int
    has_next: bool
    has_previous: bool

    class Config:
        schema_extra = {
            "example": {
                "conversation_id": "conv_123e4567-e89b-12d3-a456-426614174001",
                "summary": {
                    "conversation_id": "conv_123e4567-e89b-12d3-a456-426614174001",
                    "user_id": "user_12345",
                    "channel": "web",
                    "status": "active",
                    "started_at": "2025-05-30T10:00:00Z",
                    "last_activity_at": "2025-05-30T10:05:00Z",
                    "message_count": 4,
                    "primary_intent": "order_inquiry"
                },
                "messages": [
                    {
                        "message_id": "msg_1",
                        "direction": "inbound",
                        "timestamp": "2025-05-30T10:00:00Z",
                        "content": {
                            "type": "text",
                            "text": "Hello, I need help with my order"
                        }
                    },
                    {
                        "message_id": "msg_2",
                        "direction": "outbound",
                        "timestamp": "2025-05-30T10:00:01Z",
                        "content": {
                            "type": "text",
                            "text": "I'd be happy to help! Could you provide your order number?"
                        }
                    }
                ],
                "page": 1,
                "page_size": 20,
                "total_messages": 4,
                "has_next": False,
                "has_previous": False
            }
        }


class BulkMessageRequest(BaseModel):
    """Request model for bulk message operations"""
    messages: List[SendMessageRequest] = Field(..., max_items=100)
    batch_id: Optional[str] = Field(default_factory=lambda: str(uuid4()))

    @validator('messages')
    def validate_messages_limit(cls, v):
        if len(v) == 0:
            raise ValueError('At least one message is required')
        if len(v) > 100:
            raise ValueError('Maximum 100 messages per batch')
        return v


class BulkMessageResponse(BaseModel):
    """Response model for bulk message operations"""
    batch_id: str
    total_messages: int
    successful_messages: int
    failed_messages: int
    results: List[Dict[str, Any]]
    processing_time_ms: int

    class Config:
        schema_extra = {
            "example": {
                "batch_id": "batch_123e4567-e89b-12d3-a456-426614174000",
                "total_messages": 3,
                "successful_messages": 2,
                "failed_messages": 1,
                "results": [
                    {
                        "message_id": "msg_1",
                        "status": "success",
                        "response": "Message processed successfully"
                    },
                    {
                        "message_id": "msg_2",
                        "status": "failed",
                        "error": "Invalid content format"
                    }
                ],
                "processing_time_ms": 1250
            }
        }


class MessageStatusUpdate(BaseModel):
    """Update message status (read, delivered, etc.)"""
    message_id: MessageId
    status: str = Field(..., regex=r'^(sent|delivered|read|failed)$')
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    metadata: Optional[Dict[str, Any]] = Field(default_factory=dict)

    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }


class WebhookEvent(BaseModel):
    """Webhook event from external platforms"""
    event_type: str
    platform: str
    event_id: str
    timestamp: datetime
    data: Dict[str, Any]
    signature: Optional[str] = None

    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }
        schema_extra = {
            "example": {
                "event_type": "message_received",
                "platform": "whatsapp",
                "event_id": "evt_123456789",
                "timestamp": "2025-05-30T10:00:00Z",
                "data": {
                    "from": "+1234567890",
                    "message": {
                        "type": "text",
                        "text": "Hello, I need help"
                    }
                },
                "signature": "sha256=abc123..."
            }
        }


class MessageFeedback(BaseModel):
    """User feedback on a message"""
    message_id: MessageId
    feedback_type: str = Field(..., regex=r'^(thumbs_up|thumbs_down|helpful|not_helpful|accurate|inaccurate)$')
    rating: Optional[int] = Field(None, ge=1, le=5)
    comment: Optional[str] = Field(None, max_length=1000)
    timestamp: datetime = Field(default_factory=datetime.utcnow)

    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }


class MessageAnalytics(BaseModel):
    """Analytics data for messages"""
    message_id: MessageId
    conversation_id: ConversationId
    user_engagement_score: Optional[float] = Field(None, ge=0, le=1)
    response_relevance: Optional[float] = Field(None, ge=0, le=1)
    intent_confidence: Optional[float] = Field(None, ge=0, le=1)
    sentiment_score: Optional[float] = Field(None, ge=-1, le=1)
    processing_time_ms: int
    cost_cents: Optional[float] = None

    class Config:
        schema_extra = {
            "example": {
                "message_id": "msg_123e4567-e89b-12d3-a456-426614174000",
                "conversation_id": "conv_123e4567-e89b-12d3-a456-426614174001",
                "user_engagement_score": 0.85,
                "response_relevance": 0.92,
                "intent_confidence": 0.88,
                "sentiment_score": 0.3,
                "processing_time_ms": 287,
                "cost_cents": 1.25
            }
        }