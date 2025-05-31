"""
Conversation Validation Models
Pydantic models for conversation-related request/response validation.
"""

from datetime import datetime
from typing import Optional, Dict, Any, List
from pydantic import BaseModel, Field, validator
from uuid import uuid4

from src.models.types import (
    ChannelType, ConversationId, UserId, TenantId
)


class ConversationCreateRequest(BaseModel):
    """Request model for creating new conversations"""

    conversation_id: Optional[ConversationId] = Field(default_factory=lambda: str(uuid4()))
    user_id: UserId
    channel: ChannelType
    session_id: Optional[str] = None
    initial_message: Optional[str] = None
    user_context: Optional[Dict[str, Any]] = Field(default_factory=dict)
    metadata: Optional[Dict[str, Any]] = Field(default_factory=dict)

    class Config:
        schema_extra = {
            "example": {
                "user_id": "user_12345",
                "channel": "web",
                "session_id": "session_abc123",
                "initial_message": "Hello, I need help",
                "user_context": {
                    "page_url": "https://example.com/support",
                    "user_agent": "Mozilla/5.0...",
                    "referrer": "https://google.com"
                },
                "metadata": {
                    "department": "support",
                    "priority": "normal"
                }
            }
        }

    @validator('user_id')
    def validate_user_id(cls, v):
        if not v or len(v.strip()) == 0:
            raise ValueError('User ID cannot be empty')
        return v.strip()


class ConversationUpdateRequest(BaseModel):
    """Request model for updating conversation details"""

    status: Optional[str] = Field(None, regex=r'^(active|completed|abandoned|escalated|error)$')
    tags: Optional[List[str]] = Field(None, max_items=10)
    priority: Optional[str] = Field(None, regex=r'^(low|normal|high|urgent|critical)$')
    assigned_agent: Optional[str] = None
    department: Optional[str] = None
    category: Optional[str] = None
    notes: Optional[str] = Field(None, max_length=2000)
    metadata: Optional[Dict[str, Any]] = Field(default_factory=dict)

    class Config:
        schema_extra = {
            "example": {
                "status": "escalated",
                "tags": ["billing", "urgent"],
                "priority": "high",
                "assigned_agent": "agent_456",
                "department": "billing",
                "category": "payment_issue",
                "notes": "Customer needs immediate assistance with payment processing"
            }
        }


class ConversationCloseRequest(BaseModel):
    """Request model for closing conversations"""

    reason: str = Field(..., max_length=500)
    resolution_type: str = Field(..., regex=r'^(resolved|escalated|abandoned|timeout)$')
    satisfaction_rating: Optional[int] = Field(None, ge=1, le=5)
    agent_notes: Optional[str] = Field(None, max_length=2000)
    follow_up_required: bool = Field(default=False)
    follow_up_date: Optional[datetime] = None

    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }
        schema_extra = {
            "example": {
                "reason": "Customer issue resolved successfully",
                "resolution_type": "resolved",
                "satisfaction_rating": 5,
                "agent_notes": "Provided refund for damaged product",
                "follow_up_required": False
            }
        }

    @validator('follow_up_date')
    def validate_follow_up_date(cls, v, values):
        if values.get('follow_up_required') and not v:
            raise ValueError('Follow-up date required when follow_up_required is True')
        if not values.get('follow_up_required') and v:
            raise ValueError('Follow-up date should not be set when follow_up_required is False')
        if v and v <= datetime.utcnow():
            raise ValueError('Follow-up date must be in the future')
        return v


class ConversationResponse(BaseModel):
    """Response model for conversation details"""

    conversation_id: ConversationId
    user_id: UserId
    tenant_id: TenantId
    channel: ChannelType

    # Status and lifecycle
    status: str
    created_at: datetime
    updated_at: datetime
    last_activity_at: datetime
    closed_at: Optional[datetime] = None

    # Content and context
    message_count: int
    user_messages: int
    bot_messages: int
    primary_intent: Optional[str] = None
    current_state: Optional[str] = None

    # Business context
    tags: List[str] = Field(default_factory=list)
    priority: str = Field(default="normal")
    department: Optional[str] = None
    category: Optional[str] = None
    assigned_agent: Optional[str] = None

    # Quality metrics
    satisfaction_rating: Optional[int] = None
    resolution_type: Optional[str] = None
    escalation_count: int = Field(default=0)
    avg_response_time_ms: Optional[int] = None

    # Metadata
    metadata: Dict[str, Any] = Field(default_factory=dict)
    notes: Optional[str] = None

    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }
        schema_extra = {
            "example": {
                "conversation_id": "conv_123e4567-e89b-12d3-a456-426614174001",
                "user_id": "user_12345",
                "tenant_id": "tenant_abc123",
                "channel": "web",
                "status": "active",
                "created_at": "2025-05-30T10:00:00Z",
                "updated_at": "2025-05-30T10:15:00Z",
                "last_activity_at": "2025-05-30T10:15:00Z",
                "message_count": 8,
                "user_messages": 4,
                "bot_messages": 4,
                "primary_intent": "order_inquiry",
                "current_state": "collecting_order_details",
                "tags": ["order", "support"],
                "priority": "normal",
                "department": "customer_service",
                "avg_response_time_ms": 1200
            }
        }


class ConversationListFilters(BaseModel):
    """Filters for listing conversations"""

    status: Optional[str] = Field(None, regex=r'^(active|completed|abandoned|escalated|error)$')
    channel: Optional[ChannelType] = None
    user_id: Optional[str] = None
    assigned_agent: Optional[str] = None
    department: Optional[str] = None
    category: Optional[str] = None
    priority: Optional[str] = Field(None, regex=r'^(low|normal|high|urgent|critical)$')
    tags: Optional[List[str]] = Field(None, max_items=5)

    # Date filters
    created_after: Optional[datetime] = None
    created_before: Optional[datetime] = None
    updated_after: Optional[datetime] = None
    updated_before: Optional[datetime] = None

    # Search
    search_query: Optional[str] = Field(None, max_length=200)

    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }


class ConversationListRequest(BaseModel):
    """Request model for listing conversations with pagination"""

    # Pagination
    page: int = Field(default=1, ge=1)
    page_size: int = Field(default=20, ge=1, le=100)

    # Sorting
    sort_by: str = Field(default="last_activity_at",
                         regex=r'^(created_at|updated_at|last_activity_at|status|priority)$')
    sort_order: str = Field(default="desc", regex=r'^(asc|desc)$')

    # Filters
    filters: Optional[ConversationListFilters] = None

    class Config:
        schema_extra = {
            "example": {
                "page": 1,
                "page_size": 25,
                "sort_by": "last_activity_at",
                "sort_order": "desc",
                "filters": {
                    "status": "active",
                    "channel": "web",
                    "priority": "high",
                    "created_after": "2025-05-30T00:00:00Z"
                }
            }
        }


class ConversationListResponse(BaseModel):
    """Response model for conversation list"""

    conversations: List[ConversationResponse]

    # Pagination metadata
    page: int
    page_size: int
    total_conversations: int
    total_pages: int
    has_next: bool
    has_previous: bool

    # Summary statistics
    summary: Optional[Dict[str, Any]] = Field(default_factory=dict)

    class Config:
        schema_extra = {
            "example": {
                "conversations": [
                    {
                        "conversation_id": "conv_1",
                        "user_id": "user_123",
                        "status": "active",
                        "channel": "web",
                        "created_at": "2025-05-30T10:00:00Z"
                    }
                ],
                "page": 1,
                "page_size": 20,
                "total_conversations": 150,
                "total_pages": 8,
                "has_next": True,
                "has_previous": False,
                "summary": {
                    "active_conversations": 45,
                    "completed_today": 12,
                    "avg_response_time_ms": 1200
                }
            }
        }


class ConversationExportRequest(BaseModel):
    """Request model for exporting conversation data"""

    format: str = Field(default="json", regex=r'^(json|csv|txt|pdf)$')
    include_messages: bool = Field(default=True)
    include_metadata: bool = Field(default=True)
    include_analytics: bool = Field(default=False)
    date_range: Optional[Dict[str, datetime]] = None

    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }
        schema_extra = {
            "example": {
                "format": "json",
                "include_messages": True,
                "include_metadata": True,
                "include_analytics": False,
                "date_range": {
                    "start": "2025-05-01T00:00:00Z",
                    "end": "2025-05-31T23:59:59Z"
                }
            }
        }


class ConversationAnalyticsRequest(BaseModel):
    """Request model for conversation analytics"""

    metric_types: List[str] = Field(
        default=["message_count", "response_time", "satisfaction"],
        description="Types of metrics to include"
    )
    time_range: Dict[str, datetime]
    granularity: str = Field(default="day", regex=r'^(hour|day|week|month)$')
    group_by: Optional[List[str]] = Field(
        None,
        max_items=3,
        description="Fields to group metrics by"
    )

    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }
        schema_extra = {
            "example": {
                "metric_types": ["message_count", "response_time", "satisfaction"],
                "time_range": {
                    "start": "2025-05-01T00:00:00Z",
                    "end": "2025-05-31T23:59:59Z"
                },
                "granularity": "day",
                "group_by": ["channel", "department"]
            }
        }

    @validator('time_range')
    def validate_time_range(cls, v):
        if 'start' not in v or 'end' not in v:
            raise ValueError('Both start and end dates are required')
        if v['start'] >= v['end']:
            raise ValueError('Start date must be before end date')
        return v


class ConversationAnalyticsResponse(BaseModel):
    """Response model for conversation analytics"""

    metrics: List[Dict[str, Any]]
    summary: Dict[str, Any]
    time_range: Dict[str, datetime]
    granularity: str
    generated_at: datetime = Field(default_factory=datetime.utcnow)

    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }
        schema_extra = {
            "example": {
                "metrics": [
                    {
                        "date": "2025-05-30",
                        "channel": "web",
                        "message_count": 145,
                        "avg_response_time_ms": 1200,
                        "satisfaction_score": 4.2
                    }
                ],
                "summary": {
                    "total_conversations": 1250,
                    "avg_satisfaction": 4.1,
                    "completion_rate": 0.89
                },
                "time_range": {
                    "start": "2025-05-01T00:00:00Z",
                    "end": "2025-05-31T23:59:59Z"
                },
                "granularity": "day",
                "generated_at": "2025-05-31T10:00:00Z"
            }
        }


class ConversationTransferRequest(BaseModel):
    """Request model for transferring conversations"""

    target_agent: Optional[str] = None
    target_department: Optional[str] = None
    target_queue: Optional[str] = None
    transfer_reason: str = Field(..., max_length=500)
    transfer_notes: Optional[str] = Field(None, max_length=1000)
    priority_change: Optional[str] = Field(None, regex=r'^(low|normal|high|urgent|critical)$')

    class Config:
        schema_extra = {
            "example": {
                "target_department": "billing",
                "transfer_reason": "Customer needs help with payment processing",
                "transfer_notes": "Customer unable to process payment, may need refund",
                "priority_change": "high"
            }
        }

    @validator('target_agent', 'target_department', 'target_queue')
    def validate_transfer_target(cls, v, values, field):
        # At least one target must be specified
        targets = [
            values.get('target_agent'),
            values.get('target_department'),
            values.get('target_queue'),
            v
        ]
        if not any(targets):
            raise ValueError('At least one transfer target must be specified')
        return v