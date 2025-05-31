# src/models/schemas/request_schemas.py
"""
Pydantic schemas for API request validation.
Defines all request models used by the Chat Service API endpoints.
"""

from datetime import datetime
from typing import Dict, Any, List, Optional
from pydantic import BaseModel, Field, validator
from uuid import uuid4

from src.models.base_model import BaseRequestModel
from src.models.types import (
    ChannelType, MessageType, Priority, TenantId, UserId,
    ConversationId, MessageId, SessionId, MessageContent,
    ChannelMetadata, ProcessingHints
)


# ============================================================================
# MESSAGE SCHEMAS
# ============================================================================

class SendMessageRequest(BaseRequestModel):
    """Request schema for sending a message."""

    message_id: MessageId = Field(default_factory=lambda: str(uuid4()))
    conversation_id: Optional[ConversationId] = None
    user_id: UserId = Field(..., min_length=1, max_length=255)
    session_id: Optional[SessionId] = None

    # Channel information
    channel: ChannelType
    timestamp: datetime = Field(default_factory=datetime.utcnow)

    # Message content
    content: MessageContent

    # Channel-specific metadata
    channel_metadata: Optional[ChannelMetadata] = Field(default_factory=ChannelMetadata)

    # Processing hints
    processing_hints: Optional[ProcessingHints] = Field(default_factory=ProcessingHints)

    # Request context
    request_context: Dict[str, Any] = Field(default_factory=dict)

    @validator('user_id')
    def validate_user_id(cls, v):
        if not v or not v.strip():
            raise ValueError('User ID cannot be empty')
        return v.strip()

    @validator('request_context')
    def validate_request_context(cls, v):
        # Limit context size to prevent abuse
        if len(str(v)) > 10000:  # 10KB limit
            raise ValueError('Request context too large')
        return v

    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }
        schema_extra = {
            "example": {
                "user_id": "user_123",
                "channel": "web",
                "content": {
                    "type": "text",
                    "text": "Hello, I need help with my order",
                    "language": "en"
                },
                "channel_metadata": {
                    "platform_user_id": "web_user_456"
                },
                "processing_hints": {
                    "priority": "normal",
                    "expected_response_type": "text"
                }
            }
        }


class MessageFeedbackRequest(BaseRequestModel):
    """Request schema for message feedback."""

    message_id: MessageId = Field(..., description="Message ID to provide feedback for")
    rating: int = Field(..., ge=1, le=5, description="Rating from 1-5")
    feedback_text: Optional[str] = Field(None, max_length=1000)
    feedback_type: str = Field(
        default="general",
        regex=r'^(general|accuracy|relevance|helpfulness|tone|speed)$'
    )
    user_id: UserId = Field(..., min_length=1)

    class Config:
        schema_extra = {
            "example": {
                "message_id": "msg_123",
                "rating": 4,
                "feedback_text": "Helpful response but could be more detailed",
                "feedback_type": "helpfulness",
                "user_id": "user_123"
            }
        }


# ============================================================================
# CONVERSATION SCHEMAS
# ============================================================================

class CreateConversationRequest(BaseRequestModel):
    """Request schema for creating a new conversation."""

    conversation_id: Optional[ConversationId] = Field(default_factory=lambda: str(uuid4()))
    user_id: UserId = Field(..., min_length=1, max_length=255)
    session_id: Optional[SessionId] = None
    channel: ChannelType

    # Flow configuration
    flow_id: Optional[str] = Field(None, max_length=100)
    initial_state: Optional[str] = Field(None, max_length=100)

    # User context
    user_context: Dict[str, Any] = Field(default_factory=dict)

    # Business context
    category: Optional[str] = Field(None, max_length=100)
    priority: Priority = Field(default=Priority.NORMAL)
    tags: List[str] = Field(default_factory=list, max_items=10)

    @validator('tags')
    def validate_tags(cls, v):
        # Validate tag format
        for tag in v:
            if not tag or len(tag) > 50:
                raise ValueError('Tags must be 1-50 characters')
        return v

    class Config:
        schema_extra = {
            "example": {
                "user_id": "user_123",
                "channel": "web",
                "flow_id": "customer_support_flow",
                "category": "support",
                "priority": "normal",
                "tags": ["new_customer", "product_inquiry"]
            }
        }


class ConversationFilterRequest(BaseRequestModel):
    """Request schema for filtering conversations."""

    # Pagination
    page: int = Field(default=1, ge=1)
    page_size: int = Field(default=20, ge=1, le=100)

    # Filters
    user_id: Optional[UserId] = None
    channel: Optional[ChannelType] = None
    status: Optional[str] = Field(None, regex=r'^(active|completed|abandoned|escalated|error)$')

    # Date filters
    start_date: Optional[datetime] = None
    end_date: Optional[datetime] = None

    # Search
    search_query: Optional[str] = Field(None, max_length=200)

    # Sorting
    sort_by: str = Field(default="started_at", regex=r'^(started_at|last_activity_at|status|user_id)$')
    sort_order: str = Field(default="desc", regex=r'^(asc|desc)$')

    @validator('end_date')
    def validate_date_range(cls, v, values):
        if v and values.get('start_date') and v < values['start_date']:
            raise ValueError('End date must be after start date')
        return v

    class Config:
        schema_extra = {
            "example": {
                "page": 1,
                "page_size": 20,
                "status": "active",
                "channel": "web",
                "sort_by": "started_at",
                "sort_order": "desc"
            }
        }


# ============================================================================
# SESSION SCHEMAS
# ============================================================================

class CreateSessionRequest(BaseRequestModel):
    """Request schema for creating a new session."""

    session_id: Optional[SessionId] = Field(default_factory=lambda: str(uuid4()))
    user_id: UserId = Field(..., min_length=1, max_length=255)
    channel: ChannelType

    # Device information
    user_agent: Optional[str] = Field(None, max_length=500)
    device_type: Optional[str] = Field(None, regex=r'^(mobile|desktop|tablet|voice)$')

    # Location information (privacy-compliant)
    country: Optional[str] = Field(None, max_length=2)
    timezone: Optional[str] = Field(None, max_length=50)

    # Entry context
    referrer_url: Optional[str] = Field(None, max_length=500)
    entry_point: Optional[str] = Field(None, max_length=200)
    utm_parameters: Dict[str, str] = Field(default_factory=dict)

    # Preferences
    language: str = Field(default="en", max_length=5)
    preferences: Dict[str, Any] = Field(default_factory=dict)

    @validator('utm_parameters')
    def validate_utm_parameters(cls, v):
        # Limit UTM parameters
        allowed_keys = ['utm_source', 'utm_medium', 'utm_campaign', 'utm_term', 'utm_content']
        for key in v.keys():
            if key not in allowed_keys:
                raise ValueError(f'Invalid UTM parameter: {key}')
        return v

    class Config:
        schema_extra = {
            "example": {
                "user_id": "user_123",
                "channel": "web",
                "device_type": "desktop",
                "country": "US",
                "timezone": "America/New_York",
                "language": "en",
                "utm_parameters": {
                    "utm_source": "google",
                    "utm_medium": "cpc"
                }
            }
        }


class UpdateSessionRequest(BaseRequestModel):
    """Request schema for updating session information."""

    # Activity tracking
    last_activity: Optional[datetime] = Field(default_factory=datetime.utcnow)

    # Context updates
    context_updates: Dict[str, Any] = Field(default_factory=dict)
    preference_updates: Dict[str, Any] = Field(default_factory=dict)

    # Feature flags
    feature_updates: Dict[str, bool] = Field(default_factory=dict)

    # Metrics
    page_views: Optional[int] = Field(None, ge=0)
    clicks: Optional[int] = Field(None, ge=0)

    class Config:
        schema_extra = {
            "example": {
                "context_updates": {
                    "current_page": "/products",
                    "cart_items": 3
                },
                "preference_updates": {
                    "theme": "dark"
                },
                "page_views": 5,
                "clicks": 12
            }
        }


# ============================================================================
# ANALYTICS SCHEMAS
# ============================================================================

class AnalyticsQueryRequest(BaseRequestModel):
    """Request schema for analytics queries."""

    # Time range
    start_date: datetime
    end_date: datetime

    # Granularity
    granularity: str = Field(default="daily", regex=r'^(hourly|daily|weekly|monthly)$')

    # Filters
    tenant_id: Optional[TenantId] = None
    channel: Optional[ChannelType] = None

    # Metrics to include
    metrics: List[str] = Field(
        default=["conversations_started", "messages_sent", "completion_rate"],
        max_items=20
    )

    # Grouping
    group_by: List[str] = Field(default_factory=list, max_items=5)

    @validator('end_date')
    def validate_date_range(cls, v, values):
        if v and values.get('start_date'):
            if v < values['start_date']:
                raise ValueError('End date must be after start date')
            # Limit to 1 year
            delta = v - values['start_date']
            if delta.days > 365:
                raise ValueError('Date range cannot exceed 1 year')
        return v

    @validator('metrics')
    def validate_metrics(cls, v):
        allowed_metrics = [
            'conversations_started', 'conversations_completed', 'messages_sent',
            'messages_received', 'completion_rate', 'response_time_avg',
            'user_satisfaction', 'escalation_rate', 'cost_total'
        ]
        for metric in v:
            if metric not in allowed_metrics:
                raise ValueError(f'Invalid metric: {metric}')
        return v

    class Config:
        schema_extra = {
            "example": {
                "start_date": "2025-05-01T00:00:00Z",
                "end_date": "2025-05-31T23:59:59Z",
                "granularity": "daily",
                "channel": "web",
                "metrics": ["conversations_started", "completion_rate"],
                "group_by": ["channel"]
            }
        }


# ============================================================================
# HEALTH CHECK SCHEMAS
# ============================================================================

class HealthCheckRequest(BaseRequestModel):
    """Request schema for health check endpoints."""

    include_details: bool = Field(default=False)
    timeout_seconds: float = Field(default=10.0, ge=1.0, le=30.0)
    services: Optional[List[str]] = Field(None, max_items=10)

    @validator('services')
    def validate_services(cls, v):
        if v:
            allowed_services = ['mongodb', 'redis', 'external_api', 'message_queue']
            for service in v:
                if service not in allowed_services:
                    raise ValueError(f'Invalid service: {service}')
        return v

    class Config:
        schema_extra = {
            "example": {
                "include_details": True,
                "timeout_seconds": 5.0,
                "services": ["mongodb", "redis"]
            }
        }


# ============================================================================
# ADMIN SCHEMAS
# ============================================================================

class BulkOperationRequest(BaseRequestModel):
    """Request schema for bulk operations."""

    operation: str = Field(..., regex=r'^(export|delete|update|migrate)$')
    filters: Dict[str, Any] = Field(default_factory=dict)
    parameters: Dict[str, Any] = Field(default_factory=dict)

    # Safety limits
    max_items: int = Field(default=1000, ge=1, le=10000)
    dry_run: bool = Field(default=True)

    @validator('filters')
    def validate_filters(cls, v):
        # Ensure required filters for safety
        required_filters = ['tenant_id']
        for required in required_filters:
            if required not in v:
                raise ValueError(f'Required filter missing: {required}')
        return v

    class Config:
        schema_extra = {
            "example": {
                "operation": "export",
                "filters": {
                    "tenant_id": "tenant_123",
                    "date_range": {
                        "start": "2025-05-01",
                        "end": "2025-05-31"
                    }
                },
                "max_items": 5000,
                "dry_run": False
            }
        }


class ConfigUpdateRequest(BaseRequestModel):
    """Request schema for configuration updates."""

    config_type: str = Field(..., regex=r'^(flow|integration|model|rate_limit)$')
    config_id: str = Field(..., min_length=1, max_length=100)
    updates: Dict[str, Any] = Field(..., min_items=1)

    # Versioning
    version: Optional[str] = None
    comment: Optional[str] = Field(None, max_length=500)

    @validator('updates')
    def validate_updates(cls, v):
        if len(str(v)) > 50000:  # 50KB limit
            raise ValueError('Update payload too large')
        return v

    class Config:
        schema_extra = {
            "example": {
                "config_type": "flow",
                "config_id": "customer_support_v2",
                "updates": {
                    "initial_state": "enhanced_greeting",
                    "timeout_seconds": 300
                },
                "version": "2.1",
                "comment": "Updated greeting flow with better context"
            }
        }


# ============================================================================
# UTILITY SCHEMAS
# ============================================================================

class PaginationRequest(BaseRequestModel):
    """Reusable pagination schema."""

    page: int = Field(default=1, ge=1, description="Page number")
    page_size: int = Field(default=20, ge=1, le=100, description="Items per page")

    def get_offset(self) -> int:
        """Calculate offset for database queries."""
        return (self.page - 1) * self.page_size


class SortingRequest(BaseRequestModel):
    """Reusable sorting schema."""

    sort_by: str = Field(default="created_at", max_length=50)
    sort_order: str = Field(default="desc", regex=r'^(asc|desc)$')


class DateRangeRequest(BaseRequestModel):
    """Reusable date range schema."""

    start_date: Optional[datetime] = None
    end_date: Optional[datetime] = None

    @validator('end_date')
    def validate_date_range(cls, v, values):
        if v and values.get('start_date') and v < values['start_date']:
            raise ValueError('End date must be after start date')
        return v


# Export all request schemas
__all__ = [
    # Message schemas
    "SendMessageRequest",
    "MessageFeedbackRequest",

    # Conversation schemas
    "CreateConversationRequest",
    "ConversationFilterRequest",

    # Session schemas
    "CreateSessionRequest",
    "UpdateSessionRequest",

    # Analytics schemas
    "AnalyticsQueryRequest",

    # Health check schemas
    "HealthCheckRequest",

    # Admin schemas
    "BulkOperationRequest",
    "ConfigUpdateRequest",

    # Utility schemas
    "PaginationRequest",
    "SortingRequest",
    "DateRangeRequest"
]