# src/models/schemas/response_schemas.py
"""
Pydantic schemas for API response validation.
Defines all response models used by the Chat Service API endpoints.
"""

from datetime import datetime
from typing import Dict, Any, List, Optional, Union
from pydantic import BaseModel, Field

from src.models.base_model import BaseResponseModel
from src.models.types import (
    ChannelType, MessageType, ConversationStatus, DeliveryStatus,
    Priority, PaginatedResponse, APIResponse, MessageContent
)


# ============================================================================
# MESSAGE RESPONSE SCHEMAS
# ============================================================================

class MessageResponse(BaseResponseModel):
    """Response schema for message operations."""

    message_id: str
    conversation_id: str
    sequence_number: int
    direction: str  # inbound or outbound
    timestamp: datetime

    # Message content
    content: MessageContent

    # Processing metadata
    processing_metadata: Dict[str, Any] = Field(default_factory=dict)

    # AI analysis results
    ai_analysis: Dict[str, Any] = Field(default_factory=dict)

    # Delivery information
    delivery_status: Optional[DeliveryStatus] = None
    delivery_timestamp: Optional[datetime] = None

    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }
        schema_extra = {
            "example": {
                "message_id": "msg_123",
                "conversation_id": "conv_456",
                "sequence_number": 1,
                "direction": "outbound",
                "timestamp": "2025-05-30T10:00:00Z",
                "content": {
                    "type": "text",
                    "text": "Hello! How can I help you today?",
                    "language": "en"
                },
                "processing_metadata": {
                    "model_used": "gpt-4-turbo",
                    "processing_time_ms": 287,
                    "cost_cents": 1.25
                },
                "ai_analysis": {
                    "confidence": 0.95,
                    "intent": "greeting"
                }
            }
        }


class SendMessageResponse(BaseResponseModel):
    """Response schema for sending a message."""

    # Input message info
    user_message: MessageResponse

    # Bot response info
    bot_response: MessageResponse

    # Conversation state
    conversation_state: Dict[str, Any] = Field(default_factory=dict)

    # Processing summary
    processing_summary: Dict[str, Any] = Field(default_factory=dict)

    class Config:
        schema_extra = {
            "example": {
                "user_message": {
                    "message_id": "msg_user_123",
                    "direction": "inbound",
                    "content": {"type": "text", "text": "Hello"}
                },
                "bot_response": {
                    "message_id": "msg_bot_456",
                    "direction": "outbound",
                    "content": {"type": "text", "text": "Hi! How can I help?"}
                },
                "conversation_state": {
                    "current_state": "greeting",
                    "turn_count": 1
                },
                "processing_summary": {
                    "total_time_ms": 287,
                    "model_used": "gpt-4-turbo"
                }
            }
        }


class MessageListResponse(BaseResponseModel):
    """Response schema for listing messages."""

    messages: List[MessageResponse]
    conversation_id: str
    total_messages: int

    # Pagination info
    page: int
    page_size: int
    has_next: bool
    has_previous: bool

    class Config:
        schema_extra = {
            "example": {
                "messages": [],
                "conversation_id": "conv_123",
                "total_messages": 50,
                "page": 1,
                "page_size": 20,
                "has_next": true,
                "has_previous": false
            }
        }


# ============================================================================
# CONVERSATION RESPONSE SCHEMAS
# ============================================================================

class ConversationSummary(BaseResponseModel):
    """Summary information for a conversation."""

    conversation_id: str
    user_id: str
    channel: ChannelType
    status: ConversationStatus

    # Timing
    started_at: datetime
    last_activity_at: datetime
    completed_at: Optional[datetime] = None
    duration_seconds: Optional[int] = None

    # Metrics
    message_count: int
    user_messages: int
    bot_messages: int
    user_satisfaction: Optional[float] = None

    # Context
    current_intent: Optional[str] = None
    primary_topics: List[str] = Field(default_factory=list)
    tags: List[str] = Field(default_factory=list)

    class Config:
        schema_extra = {
            "example": {
                "conversation_id": "conv_123",
                "user_id": "user_456",
                "channel": "web",
                "status": "completed",
                "started_at": "2025-05-30T10:00:00Z",
                "last_activity_at": "2025-05-30T10:15:00Z",
                "duration_seconds": 900,
                "message_count": 12,
                "user_messages": 6,
                "bot_messages": 6,
                "user_satisfaction": 4.2,
                "current_intent": "order_completed",
                "primary_topics": ["order_status", "shipping"],
                "tags": ["resolved", "positive_feedback"]
            }
        }


class ConversationResponse(BaseResponseModel):
    """Detailed conversation response."""

    # Basic info
    conversation_id: str
    user_id: str
    session_id: Optional[str] = None
    channel: ChannelType
    status: ConversationStatus

    # Timing
    started_at: datetime
    last_activity_at: datetime
    completed_at: Optional[datetime] = None

    # Flow information
    flow_id: Optional[str] = None
    current_state: Optional[str] = None

    # Context
    context: Dict[str, Any] = Field(default_factory=dict)

    # Metrics
    metrics: Dict[str, Any] = Field(default_factory=dict)

    # Business context
    business_context: Dict[str, Any] = Field(default_factory=dict)

    # Recent messages (limited)
    recent_messages: List[MessageResponse] = Field(default_factory=list)

    class Config:
        schema_extra = {
            "example": {
                "conversation_id": "conv_123",
                "user_id": "user_456",
                "channel": "web",
                "status": "active",
                "started_at": "2025-05-30T10:00:00Z",
                "last_activity_at": "2025-05-30T10:05:00Z",
                "flow_id": "customer_support",
                "current_state": "collecting_info",
                "context": {
                    "current_intent": "order_inquiry",
                    "slots": {"order_number": "ORD123456"}
                },
                "metrics": {
                    "message_count": 4,
                    "response_time_avg_ms": 250
                }
            }
        }


class ConversationListResponse(BaseResponseModel):
    """Response schema for listing conversations."""

    conversations: List[ConversationSummary]

    # Pagination
    total_conversations: int
    page: int
    page_size: int
    has_next: bool
    has_previous: bool

    # Filters applied
    filters_applied: Dict[str, Any] = Field(default_factory=dict)

    class Config:
        schema_extra = {
            "example": {
                "conversations": [],
                "total_conversations": 150,
                "page": 1,
                "page_size": 20,
                "has_next": true,
                "has_previous": false,
                "filters_applied": {
                    "status": "active",
                    "channel": "web"
                }
            }
        }


# ============================================================================
# SESSION RESPONSE SCHEMAS
# ============================================================================

class SessionResponse(BaseResponseModel):
    """Response schema for session information."""

    session_id: str
    user_id: str
    status: str

    # Timing
    started_at: datetime
    last_activity_at: datetime
    expires_at: datetime

    # Channel info
    primary_channel: ChannelType
    channels_used: List[ChannelType]

    # Activity metrics
    total_conversations: int
    total_messages: int
    duration_seconds: int

    # Device and location (privacy-compliant)
    device_type: Optional[str] = None
    country: Optional[str] = None

    # Feature flags
    active_features: List[str] = Field(default_factory=list)

    class Config:
        schema_extra = {
            "example": {
                "session_id": "sess_123",
                "user_id": "user_456",
                "status": "active",
                "started_at": "2025-05-30T09:00:00Z",
                "last_activity_at": "2025-05-30T10:00:00Z",
                "expires_at": "2025-05-30T18:00:00Z",
                "primary_channel": "web",
                "channels_used": ["web"],
                "total_conversations": 2,
                "total_messages": 15,
                "duration_seconds": 3600,
                "device_type": "desktop",
                "country": "US",
                "active_features": ["enhanced_ui", "voice_input"]
            }
        }


# ============================================================================
# ANALYTICS RESPONSE SCHEMAS
# ============================================================================

class MetricDataPoint(BaseResponseModel):
    """Single metric data point."""

    timestamp: datetime
    value: Union[int, float]
    metadata: Dict[str, Any] = Field(default_factory=dict)


class MetricSeries(BaseResponseModel):
    """Time series for a single metric."""

    metric_name: str
    data_points: List[MetricDataPoint]
    aggregation_type: str  # sum, avg, count, etc.


class AnalyticsResponse(BaseResponseModel):
    """Response schema for analytics queries."""

    # Query info
    start_date: datetime
    end_date: datetime
    granularity: str

    # Metrics data
    metrics: List[MetricSeries]

    # Summary statistics
    summary: Dict[str, Any] = Field(default_factory=dict)

    # Query metadata
    query_metadata: Dict[str, Any] = Field(default_factory=dict)

    class Config:
        schema_extra = {
            "example": {
                "start_date": "2025-05-01T00:00:00Z",
                "end_date": "2025-05-31T23:59:59Z",
                "granularity": "daily",
                "metrics": [
                    {
                        "metric_name": "conversations_started",
                        "data_points": [
                            {
                                "timestamp": "2025-05-01T00:00:00Z",
                                "value": 150
                            }
                        ],
                        "aggregation_type": "count"
                    }
                ],
                "summary": {
                    "total_conversations": 4650,
                    "avg_daily_conversations": 150
                }
            }
        }


# ============================================================================
# HEALTH CHECK RESPONSE SCHEMAS
# ============================================================================

class ServiceHealth(BaseResponseModel):
    """Health status for a single service."""

    service: str
    healthy: bool
    response_time_ms: Optional[float] = None
    error: Optional[str] = None
    details: Dict[str, Any] = Field(default_factory=dict)
    timestamp: datetime


class HealthCheckResponse(BaseResponseModel):
    """Response schema for health checks."""

    status: str  # healthy, unhealthy, degraded
    timestamp: datetime
    response_time_ms: float

    # Individual service health
    services: Dict[str, ServiceHealth]

    # Overall system info
    system_info: Dict[str, Any] = Field(default_factory=dict)

    class Config:
        schema_extra = {
            "example": {
                "status": "healthy",
                "timestamp": "2025-05-30T10:00:00Z",
                "response_time_ms": 45.2,
                "services": {
                    "mongodb": {
                        "service": "mongodb",
                        "healthy": true,
                        "response_time_ms": 12.5,
                        "timestamp": "2025-05-30T10:00:00Z"
                    },
                    "redis": {
                        "service": "redis",
                        "healthy": true,
                        "response_time_ms": 8.1,
                        "timestamp": "2025-05-30T10:00:00Z"
                    }
                }
            }
        }


# ============================================================================
# ERROR RESPONSE SCHEMAS
# ============================================================================

class ErrorResponse(BaseResponseModel):
    """Standard error response schema."""

    status: str = "error"
    error: Dict[str, Any]
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    request_id: Optional[str] = None

    class Config:
        schema_extra = {
            "example": {
                "status": "error",
                "error": {
                    "code": "VALIDATION_ERROR",
                    "message": "Request validation failed",
                    "details": {
                        "field": "user_id",
                        "reason": "User ID cannot be empty"
                    }
                },
                "timestamp": "2025-05-30T10:00:00Z",
                "request_id": "req_123"
            }
        }


class ValidationErrorResponse(BaseResponseModel):
    """Validation error response with field details."""

    status: str = "error"
    error: Dict[str, Any]
    validation_errors: List[Dict[str, Any]] = Field(default_factory=list)
    timestamp: datetime = Field(default_factory=datetime.utcnow)

    class Config:
        schema_extra = {
            "example": {
                "status": "error",
                "error": {
                    "code": "VALIDATION_ERROR",
                    "message": "Multiple validation errors"
                },
                "validation_errors": [
                    {
                        "field": "user_id",
                        "message": "User ID is required",
                        "type": "missing"
                    },
                    {
                        "field": "content.text",
                        "message": "Text content is required for text messages",
                        "type": "validation"
                    }
                ],
                "timestamp": "2025-05-30T10:00:00Z"
            }
        }


# ============================================================================
# SUCCESS RESPONSE SCHEMAS
# ============================================================================

class SuccessResponse(BaseResponseModel):
    """Generic success response."""

    status: str = "success"
    message: str
    data: Optional[Dict[str, Any]] = None
    timestamp: datetime = Field(default_factory=datetime.utcnow)

    class Config:
        schema_extra = {
            "example": {
                "status": "success",
                "message": "Operation completed successfully",
                "data": {"processed_items": 5},
                "timestamp": "2025-05-30T10:00:00Z"
            }
        }


class CreatedResponse(BaseResponseModel):
    """Response for successful creation operations."""

    status: str = "success"
    message: str
    created_id: str
    created_at: datetime = Field(default_factory=datetime.utcnow)
    data: Optional[Dict[str, Any]] = None

    class Config:
        schema_extra = {
            "example": {
                "status": "success",
                "message": "Conversation created successfully",
                "created_id": "conv_123",
                "created_at": "2025-05-30T10:00:00Z",
                "data": {"initial_state": "greeting"}
            }
        }


class UpdatedResponse(BaseResponseModel):
    """Response for successful update operations."""

    status: str = "success"
    message: str
    updated_id: str
    updated_at: datetime = Field(default_factory=datetime.utcnow)
    changes_applied: int = 0

    class Config:
        schema_extra = {
            "example": {
                "status": "success",
                "message": "Session updated successfully",
                "updated_id": "sess_123",
                "updated_at": "2025-05-30T10:00:00Z",
                "changes_applied": 3
            }
        }


class DeletedResponse(BaseResponseModel):
    """Response for successful deletion operations."""

    status: str = "success"
    message: str
    deleted_id: str
    deleted_at: datetime = Field(default_factory=datetime.utcnow)

    class Config:
        schema_extra = {
            "example": {
                "status": "success",
                "message": "Session deleted successfully",
                "deleted_id": "sess_123",
                "deleted_at": "2025-05-30T10:00:00Z"
            }
        }


# ============================================================================
# ADMIN RESPONSE SCHEMAS
# ============================================================================

class BulkOperationResponse(BaseResponseModel):
    """Response schema for bulk operations."""

    operation: str
    status: str

    # Results
    total_processed: int
    successful: int
    failed: int
    skipped: int

    # Timing
    started_at: datetime
    completed_at: datetime
    duration_seconds: float

    # Details
    errors: List[Dict[str, Any]] = Field(default_factory=list)
    summary: Dict[str, Any] = Field(default_factory=dict)

    class Config:
        schema_extra = {
            "example": {
                "operation": "export",
                "status": "completed",
                "total_processed": 1000,
                "successful": 995,
                "failed": 5,
                "skipped": 0,
                "started_at": "2025-05-30T10:00:00Z",
                "completed_at": "2025-05-30T10:05:00Z",
                "duration_seconds": 300.5,
                "errors": [],
                "summary": {"exported_file": "conversations_export.csv"}
            }
        }


# ============================================================================
# GENERIC WRAPPER RESPONSES
# ============================================================================

# Generic paginated response
class PaginatedConversationResponse(PaginatedResponse[ConversationSummary]):
    """Paginated conversation response."""
    pass


class PaginatedMessageResponse(PaginatedResponse[MessageResponse]):
    """Paginated message response."""
    pass


# Generic API responses
class MessageAPIResponse(APIResponse[MessageResponse]):
    """API response wrapping message data."""
    pass


class ConversationAPIResponse(APIResponse[ConversationResponse]):
    """API response wrapping conversation data."""
    pass


class SessionAPIResponse(APIResponse[SessionResponse]):
    """API response wrapping session data."""
    pass


class AnalyticsAPIResponse(APIResponse[AnalyticsResponse]):
    """API response wrapping analytics data."""
    pass


# Export all response schemas
__all__ = [
    # Message responses
    "MessageResponse",
    "SendMessageResponse",
    "MessageListResponse",

    # Conversation responses
    "ConversationSummary",
    "ConversationResponse",
    "ConversationListResponse",

    # Session responses
    "SessionResponse",

    # Analytics responses
    "MetricDataPoint",
    "MetricSeries",
    "AnalyticsResponse",

    # Health check responses
    "ServiceHealth",
    "HealthCheckResponse",

    # Error responses
    "ErrorResponse",
    "ValidationErrorResponse",

    # Success responses
    "SuccessResponse",
    "CreatedResponse",
    "UpdatedResponse",
    "DeletedResponse",

    # Admin responses
    "BulkOperationResponse",

    # Generic wrapper responses
    "PaginatedConversationResponse",
    "PaginatedMessageResponse",
    "MessageAPIResponse",
    "ConversationAPIResponse",
    "SessionAPIResponse",
    "AnalyticsAPIResponse"
]