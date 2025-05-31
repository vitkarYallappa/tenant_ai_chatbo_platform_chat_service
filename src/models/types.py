"""
Common type definitions, enums, and type aliases used across the application.
Centralized type definitions to ensure consistency.
"""

from enum import Enum
from typing import TypedDict, Optional, List, Dict, Any, Union
from datetime import datetime
from pydantic import BaseModel, Field, validator
from uuid import UUID, uuid4


# ============================================================================
# ENUMS FOR TYPE SAFETY
# ============================================================================

class ChannelType(str, Enum):
    """Define supported communication channels"""
    WEB = "web"
    WHATSAPP = "whatsapp"
    MESSENGER = "messenger"
    SLACK = "slack"
    TEAMS = "teams"
    SMS = "sms"
    VOICE = "voice"


class MessageType(str, Enum):
    """Define supported message content types"""
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


class ConversationStatus(str, Enum):
    """Define conversation lifecycle states"""
    ACTIVE = "active"
    COMPLETED = "completed"
    ABANDONED = "abandoned"
    ESCALATED = "escalated"
    ERROR = "error"


class DeliveryStatus(str, Enum):
    """Define message delivery states"""
    SENT = "sent"
    DELIVERED = "delivered"
    READ = "read"
    FAILED = "failed"


class ProcessingStage(str, Enum):
    """Define message processing stages"""
    RECEIVED = "received"
    VALIDATED = "validated"
    NORMALIZED = "normalized"
    PROCESSED = "processed"
    RESPONDED = "responded"
    DELIVERED = "delivered"


class UserRole(str, Enum):
    """Define user roles within tenant"""
    OWNER = "owner"
    ADMIN = "admin"
    DEVELOPER = "developer"
    MANAGER = "manager"
    MEMBER = "member"
    VIEWER = "viewer"


class IntentType(str, Enum):
    """Common intent types"""
    GREETING = "greeting"
    ORDER_INQUIRY = "order_inquiry"
    TECHNICAL_SUPPORT = "technical_support"
    BILLING_QUESTION = "billing_question"
    GENERAL_INFO = "general_info"
    ESCALATION = "escalation"
    GOODBYE = "goodbye"
    UNKNOWN = "unknown"


class SentimentLabel(str, Enum):
    """Sentiment analysis labels"""
    POSITIVE = "positive"
    NEGATIVE = "negative"
    NEUTRAL = "neutral"


class Priority(str, Enum):
    """Priority levels for messages and tasks"""
    LOW = "low"
    NORMAL = "normal"
    HIGH = "high"
    URGENT = "urgent"
    CRITICAL = "critical"


# ============================================================================
# TYPE ALIASES FOR CLARITY
# ============================================================================

TenantId = str
UserId = str
ConversationId = str
MessageId = str
SessionId = str
FlowId = str
IntegrationId = str
ApiKeyId = str


# ============================================================================
# BASE CONTENT MODELS
# ============================================================================

class MediaContent(BaseModel):
    """Media content structure for images, videos, audio, files"""
    url: str = Field(..., regex=r'^https?://')
    type: str = Field(..., description="MIME type")
    size_bytes: int = Field(..., ge=0, le=52428800)  # Max 50MB
    alt_text: Optional[str] = Field(None, max_length=500)
    thumbnail_url: Optional[str] = None

    # Audio/Video specific
    duration_ms: Optional[int] = Field(None, ge=0)

    # Image specific
    dimensions: Optional[Dict[str, int]] = None  # width, height

    # File specific
    filename: Optional[str] = Field(None, max_length=255)
    file_extension: Optional[str] = Field(None, max_length=10)

    @validator('type')
    def validate_mime_type(cls, v):
        allowed_types = [
            # Images
            'image/jpeg', 'image/png', 'image/gif', 'image/webp',
            # Videos
            'video/mp4', 'video/quicktime', 'video/webm',
            # Audio
            'audio/mpeg', 'audio/wav', 'audio/ogg', 'audio/m4a',
            # Documents
            'application/pdf', 'text/plain', 'text/csv',
            'application/vnd.openxmlformats-officedocument.wordprocessingml.document',
            'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
        ]
        if v not in allowed_types:
            raise ValueError(f'Unsupported media type: {v}')
        return v


class LocationContent(BaseModel):
    """Location data structure"""
    latitude: float = Field(..., ge=-90, le=90)
    longitude: float = Field(..., ge=-180, le=180)
    accuracy_meters: Optional[int] = Field(None, ge=0)
    address: Optional[str] = Field(None, max_length=500)
    place_name: Optional[str] = Field(None, max_length=200)
    place_id: Optional[str] = Field(None, max_length=100)


class QuickReply(BaseModel):
    """Quick reply button structure"""
    title: str = Field(..., max_length=20)
    payload: str = Field(..., max_length=1000)
    content_type: str = Field(default="text")
    image_url: Optional[str] = None


class Button(BaseModel):
    """Interactive button structure"""
    type: str = Field(..., regex=r'^(postback|url|phone|share|login)$')
    title: str = Field(..., max_length=20)
    payload: Optional[str] = Field(None, max_length=1000)
    url: Optional[str] = None
    webview_height_ratio: Optional[str] = Field(None, regex=r'^(compact|tall|full)$')


class CarouselItem(BaseModel):
    """Individual item in a carousel"""
    title: str = Field(..., max_length=80)
    subtitle: Optional[str] = Field(None, max_length=80)
    image_url: Optional[str] = None
    default_action: Optional[Dict[str, Any]] = None
    buttons: Optional[List[Button]] = Field(None, max_items=3)


class FormField(BaseModel):
    """Form field definition"""
    field_id: str = Field(..., max_length=50)
    field_type: str = Field(..., regex=r'^(text|email|phone|number|date|select|textarea)$')
    label: str = Field(..., max_length=100)
    required: bool = Field(default=False)
    placeholder: Optional[str] = Field(None, max_length=100)
    options: Optional[List[str]] = None  # For select fields
    validation_regex: Optional[str] = None
    min_length: Optional[int] = None
    max_length: Optional[int] = None


class MessageContent(BaseModel):
    """Main message content structure"""
    type: MessageType
    text: Optional[str] = Field(None, max_length=4096)
    language: Optional[str] = Field(default="en", regex=r'^[a-z]{2}$')

    # Rich content
    media: Optional[MediaContent] = None
    location: Optional[LocationContent] = None

    # Interactive elements
    quick_replies: Optional[List[QuickReply]] = Field(None, max_items=13)
    buttons: Optional[List[Button]] = Field(None, max_items=3)
    carousel: Optional[List[CarouselItem]] = Field(None, max_items=10)

    # Form content
    form_fields: Optional[List[FormField]] = None
    form_data: Optional[Dict[str, Any]] = None

    # System message specifics
    system_event: Optional[str] = None

    @validator('text')
    def text_required_for_text_type(cls, v, values):
        if values.get('type') == MessageType.TEXT and not v:
            raise ValueError('Text content required for text messages')
        return v

    @validator('media')
    def media_required_for_media_types(cls, v, values):
        media_types = [MessageType.IMAGE, MessageType.FILE, MessageType.AUDIO, MessageType.VIDEO]
        if values.get('type') in media_types and not v:
            raise ValueError(f'Media content required for {values.get("type")} messages')
        return v

    @validator('location')
    def location_required_for_location_type(cls, v, values):
        if values.get('type') == MessageType.LOCATION and not v:
            raise ValueError('Location content required for location messages')
        return v


class ChannelMetadata(BaseModel):
    """Channel-specific metadata"""
    platform_message_id: Optional[str] = None
    platform_user_id: Optional[str] = None
    thread_id: Optional[str] = None
    workspace_id: Optional[str] = None
    bot_id: Optional[str] = None
    team_id: Optional[str] = None

    # Delivery tracking
    delivery_status: Optional[DeliveryStatus] = None
    delivery_timestamp: Optional[datetime] = None
    read_timestamp: Optional[datetime] = None

    # Additional platform-specific data
    additional_data: Optional[Dict[str, Any]] = Field(default_factory=dict)


class ProcessingHints(BaseModel):
    """Hints for message processing"""
    priority: Priority = Field(default=Priority.NORMAL)
    expected_response_type: Optional[MessageType] = None
    bypass_automation: bool = Field(default=False)
    require_human_review: bool = Field(default=False)
    processing_timeout_ms: Optional[int] = Field(None, ge=1000, le=30000)
    cost_limit_cents: Optional[int] = Field(None, ge=0)


# ============================================================================
# AI AND ANALYSIS MODELS
# ============================================================================

class IntentResult(BaseModel):
    """Intent detection result"""
    detected_intent: str
    confidence: float = Field(..., ge=0, le=1)
    alternatives: List[Dict[str, Any]] = Field(default_factory=list)


class EntityResult(BaseModel):
    """Named entity extraction result"""
    entity_type: str
    entity_value: str
    start_pos: int
    end_pos: int
    confidence: float = Field(..., ge=0, le=1)
    resolution: Optional[Dict[str, Any]] = None
    source: str = Field(default="user_input")


class SentimentResult(BaseModel):
    """Sentiment analysis result"""
    label: SentimentLabel
    score: float = Field(..., ge=-1, le=1)
    confidence: float = Field(..., ge=0, le=1)
    emotions: Optional[Dict[str, float]] = None


class ToxicityResult(BaseModel):
    """Content toxicity analysis result"""
    is_toxic: bool
    toxicity_score: float = Field(..., ge=0, le=1)
    categories: List[str] = Field(default_factory=list)


# ============================================================================
# BUSINESS CONTEXT MODELS
# ============================================================================

class UserInfo(BaseModel):
    """User information for context"""
    first_seen: datetime
    return_visitor: bool = Field(default=False)
    language: str = Field(default="en")
    timezone: Optional[str] = None

    device_info: Optional[Dict[str, str]] = None
    location_info: Optional[Dict[str, str]] = None

    # Privacy-compliant data
    preferences: Dict[str, Any] = Field(default_factory=dict)
    custom_attributes: Dict[str, Any] = Field(default_factory=dict)


class BusinessContext(BaseModel):
    """Business-specific context"""
    department: Optional[str] = None
    category: Optional[str] = None
    subcategory: Optional[str] = None
    priority: Priority = Field(default=Priority.NORMAL)
    tags: List[str] = Field(default_factory=list)

    # Resolution tracking
    resolution_type: Optional[str] = None
    outcome: Optional[str] = None
    value_generated: Optional[float] = None
    cost_incurred: Optional[float] = None


# ============================================================================
# GENERIC RESPONSE TYPES
# ============================================================================

from typing import TypeVar, Generic

T = TypeVar('T')


class PaginatedResponse(BaseModel, Generic[T]):
    """Generic paginated response structure"""
    items: List[T]
    total: int
    page: int
    page_size: int
    has_next: bool
    has_previous: bool


class APIResponse(BaseModel, Generic[T]):
    """Generic API response wrapper"""
    status: str = Field(default="success")
    data: Optional[T] = None
    error: Optional[Dict[str, Any]] = None
    meta: Dict[str, Any] = Field(default_factory=dict)


class ErrorDetail(BaseModel):
    """Error detail structure"""
    code: str
    message: str
    field: Optional[str] = None
    details: Optional[Dict[str, Any]] = None


# ============================================================================
# VALIDATION HELPERS
# ============================================================================

class PhoneNumber(BaseModel):
    """Phone number validation model"""
    phone: str

    @validator('phone')
    def validate_e164(cls, v):
        import re
        pattern = r'^\+[1-9]\d{1,14}$'
        if not re.match(pattern, v):
            raise ValueError('Invalid E.164 phone number format')
        return v


class EmailAddress(BaseModel):
    """Email address validation model"""
    email: str

    @validator('email')
    def validate_email(cls, v):
        import re
        pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
        if not re.match(pattern, v.lower()):
            raise ValueError('Invalid email format')
        return v.lower()


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def generate_message_id() -> MessageId:
    """Generate a unique message ID"""
    return str(uuid4())


def generate_conversation_id() -> ConversationId:
    """Generate a unique conversation ID"""
    return str(uuid4())


def generate_session_id() -> SessionId:
    """Generate a unique session ID"""
    return str(uuid4())


def is_valid_uuid(uuid_string: str) -> bool:
    """Check if string is a valid UUID"""
    try:
        UUID(uuid_string)
        return True
    except ValueError:
        return False