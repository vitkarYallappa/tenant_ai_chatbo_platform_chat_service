"""
Unified Type Definitions and Enumerations
=========================================

Comprehensive type definitions, enumerations, and validation utilities used across the chat service.
Combines and extends functionality from both old and new type systems for maximum compatibility
and feature completeness.

Features:
- Complete enumeration types with utility methods
- Type aliases with validation
- Pydantic models for complex data structures
- Validation utilities and type guards
- Serialization support
- Factory functions for ID generation
- Generic response types
"""
import uuid
from enum import Enum, IntEnum
from typing import TypeVar, NewType, Union, List, Dict, Any, Optional, Generic
from datetime import datetime
from pydantic import BaseModel, Field, validator
from uuid import UUID, uuid4
import re

# ============================================================================
# GENERIC TYPE VARIABLES
# ============================================================================

T = TypeVar('T')
K = TypeVar('K')
V = TypeVar('V')

# ============================================================================
# TYPE ALIASES FOR BETTER CODE READABILITY
# ============================================================================

TenantId = NewType('TenantId', str)
UserId = NewType('UserId', str)
ConversationId = NewType('ConversationId', str)
MessageId = NewType('MessageId', str)
SessionId = NewType('SessionId', str)
FlowId = NewType('FlowId', str)
IntegrationId = NewType('IntegrationId', str)
ApiKeyId = NewType('ApiKeyId', str)

# Timestamp type for consistent datetime handling
Timestamp = NewType('Timestamp', datetime)

# ============================================================================
# ENUMERATIONS FOR TYPE SAFETY
# ============================================================================

class ChannelType(str, Enum):
    """Communication channel types"""
    WEB = "web"
    WHATSAPP = "whatsapp"
    MESSENGER = "messenger"
    SLACK = "slack"
    TEAMS = "teams"
    TELEGRAM = "telegram"
    VOICE = "voice"
    SMS = "sms"
    EMAIL = "email"
    API = "api"
    WIDGET = "widget"

    @classmethod
    def get_supported_channels(cls) -> List[str]:
        """Get list of supported channel types"""
        return [channel.value for channel in cls]

    @classmethod
    def is_valid_channel(cls, channel: str) -> bool:
        """Check if channel type is valid"""
        return channel in cls.get_supported_channels()

    @classmethod
    def get_messaging_channels(cls) -> List[str]:
        """Get channels that support messaging"""
        return [cls.WEB.value, cls.WHATSAPP.value, cls.MESSENGER.value,
                cls.SLACK.value, cls.TEAMS.value, cls.TELEGRAM.value, cls.SMS.value]

    @classmethod
    def get_voice_channels(cls) -> List[str]:
        """Get channels that support voice"""
        return [cls.VOICE.value, cls.WHATSAPP.value, cls.TELEGRAM.value]


class MessageType(str, Enum):
    """Message content types"""
    TEXT = "text"
    IMAGE = "image"
    FILE = "file"
    AUDIO = "audio"
    VIDEO = "video"
    LOCATION = "location"
    CONTACT = "contact"
    QUICK_REPLY = "quick_reply"
    CAROUSEL = "carousel"
    FORM = "form"
    SYSTEM = "system"
    TYPING = "typing"
    READ_RECEIPT = "read_receipt"

    @classmethod
    def get_media_types(cls) -> List[str]:
        """Get message types that contain media"""
        return [cls.IMAGE.value, cls.FILE.value, cls.AUDIO.value, cls.VIDEO.value]

    @classmethod
    def get_interactive_types(cls) -> List[str]:
        """Get interactive message types"""
        return [cls.QUICK_REPLY.value, cls.CAROUSEL.value, cls.FORM.value]

    @classmethod
    def get_system_types(cls) -> List[str]:
        """Get system message types"""
        return [cls.SYSTEM.value, cls.TYPING.value, cls.READ_RECEIPT.value]


class MessageDirection(str, Enum):
    """Message direction in conversation"""
    INBOUND = "inbound"
    OUTBOUND = "outbound"
    INTERNAL = "internal"  # System-generated messages


class ConversationStatus(str, Enum):
    """Conversation lifecycle status"""
    ACTIVE = "active"
    COMPLETED = "completed"
    ABANDONED = "abandoned"
    ESCALATED = "escalated"
    PAUSED = "paused"
    ERROR = "error"
    ARCHIVED = "archived"

    @classmethod
    def get_active_statuses(cls) -> List[str]:
        """Get statuses considered as active"""
        return [cls.ACTIVE.value, cls.PAUSED.value, cls.ESCALATED.value]

    @classmethod
    def get_terminal_statuses(cls) -> List[str]:
        """Get final statuses"""
        return [cls.COMPLETED.value, cls.ABANDONED.value, cls.ARCHIVED.value]

    @classmethod
    def can_transition_to(cls, from_status: str, to_status: str) -> bool:
        """Check if status transition is valid"""
        valid_transitions = {
            cls.ACTIVE.value: [cls.COMPLETED.value, cls.ABANDONED.value, cls.ESCALATED.value, cls.PAUSED.value, cls.ERROR.value],
            cls.PAUSED.value: [cls.ACTIVE.value, cls.COMPLETED.value, cls.ABANDONED.value],
            cls.ESCALATED.value: [cls.ACTIVE.value, cls.COMPLETED.value, cls.ABANDONED.value],
            cls.ERROR.value: [cls.ACTIVE.value, cls.ABANDONED.value],
            cls.COMPLETED.value: [cls.ARCHIVED.value],
            cls.ABANDONED.value: [cls.ARCHIVED.value],
            cls.ARCHIVED.value: []
        }
        return to_status in valid_transitions.get(from_status, [])


class SessionStatus(str, Enum):
    """User session status"""
    ACTIVE = "active"
    EXPIRED = "expired"
    TERMINATED = "terminated"
    SUSPENDED = "suspended"


class DeliveryStatus(str, Enum):
    """Message delivery status"""
    SENT = "sent"
    DELIVERED = "delivered"
    READ = "read"
    FAILED = "failed"
    PENDING = "pending"
    REJECTED = "rejected"

    @classmethod
    def get_successful_statuses(cls) -> List[str]:
        """Get statuses that indicate successful delivery"""
        return [cls.SENT.value, cls.DELIVERED.value, cls.READ.value]

    @classmethod
    def get_failed_statuses(cls) -> List[str]:
        """Get statuses that indicate delivery failure"""
        return [cls.FAILED.value, cls.REJECTED.value]


class ProcessingStage(str, Enum):
    """Message processing stages"""
    RECEIVED = "received"
    VALIDATED = "validated"
    NORMALIZED = "normalized"
    PROCESSED = "processed"
    RESPONDED = "responded"
    DELIVERED = "delivered"
    FAILED = "failed"


class Priority(IntEnum):
    """Priority levels"""
    LOW = 1
    NORMAL = 2
    HIGH = 3
    URGENT = 4
    CRITICAL = 5

    @classmethod
    def from_string(cls, priority_str: str) -> "Priority":
        """Convert string to Priority enum"""
        priority_map = {
            "low": cls.LOW,
            "normal": cls.NORMAL,
            "high": cls.HIGH,
            "urgent": cls.URGENT,
            "critical": cls.CRITICAL
        }
        return priority_map.get(priority_str.lower(), cls.NORMAL)

    def to_string(self) -> str:
        """Convert Priority enum to string"""
        return self.name.lower()


class UserRole(str, Enum):
    """User role types"""
    OWNER = "owner"
    ADMIN = "admin"
    DEVELOPER = "developer"
    MANAGER = "manager"
    MEMBER = "member"
    VIEWER = "viewer"
    GUEST = "guest"

    @classmethod
    def get_permission_hierarchy(cls) -> Dict[str, int]:
        """Get role permission hierarchy (higher number = more permissions)"""
        return {
            cls.GUEST.value: 0,
            cls.VIEWER.value: 1,
            cls.MEMBER.value: 2,
            cls.MANAGER.value: 3,
            cls.DEVELOPER.value: 4,
            cls.ADMIN.value: 5,
            cls.OWNER.value: 6
        }

    def has_permission_level(self, required_role: "UserRole") -> bool:
        """Check if this role has at least the required permission level"""
        hierarchy = self.get_permission_hierarchy()
        current_level = hierarchy.get(self.value, 0)
        required_level = hierarchy.get(required_role.value, 0)
        return current_level >= required_level

    @classmethod
    def get_roles_with_permission(cls, min_role: "UserRole") -> List[str]:
        """Get all roles that have at least the specified permission level"""
        hierarchy = cls.get_permission_hierarchy()
        min_level = hierarchy.get(min_role.value, 0)
        return [role for role, level in hierarchy.items() if level >= min_level]


class TenantPlan(str, Enum):
    """Tenant subscription plans"""
    FREE = "free"
    STARTER = "starter"
    PROFESSIONAL = "professional"
    ENTERPRISE = "enterprise"
    CUSTOM = "custom"

    @classmethod
    def get_plan_limits(cls, plan: str) -> Dict[str, int]:
        """Get limits for each plan"""
        limits = {
            cls.FREE.value: {"messages_per_month": 1000, "integrations": 2, "users": 3},
            cls.STARTER.value: {"messages_per_month": 10000, "integrations": 5, "users": 10},
            cls.PROFESSIONAL.value: {"messages_per_month": 100000, "integrations": 20, "users": 50},
            cls.ENTERPRISE.value: {"messages_per_month": -1, "integrations": -1, "users": -1},
            cls.CUSTOM.value: {"messages_per_month": -1, "integrations": -1, "users": -1}
        }
        return limits.get(plan, limits[cls.FREE.value])


class IntegrationType(str, Enum):
    """Integration types for external services"""
    REST_API = "rest_api"
    GRAPHQL = "graphql"
    WEBHOOK = "webhook"
    DATABASE = "database"
    OAUTH2 = "oauth2"
    CUSTOM = "custom"
    WEBSOCKET = "websocket"


class ModelProvider(str, Enum):
    """AI model providers"""
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    GOOGLE = "google"
    AZURE = "azure"
    HUGGINGFACE = "huggingface"
    CUSTOM = "custom"
    COHERE = "cohere"
    MISTRAL = "mistral"


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
    COMPLAINT = "complaint"
    COMPLIMENT = "compliment"
    BOOKING = "booking"
    CANCELLATION = "cancellation"


class SentimentLabel(str, Enum):
    """Sentiment analysis labels"""
    POSITIVE = "positive"
    NEGATIVE = "negative"
    NEUTRAL = "neutral"
    MIXED = "mixed"

    @classmethod
    def from_score(cls, score: float) -> "SentimentLabel":
        """Convert numerical score to sentiment label"""
        if score > 0.1:
            return cls.POSITIVE
        elif score < -0.1:
            return cls.NEGATIVE
        else:
            return cls.NEUTRAL


class IntentConfidence(str, Enum):
    """Intent confidence levels"""
    HIGH = "high"      # > 0.8
    MEDIUM = "medium"  # 0.5 - 0.8
    LOW = "low"        # < 0.5

    @classmethod
    def from_score(cls, score: float) -> "IntentConfidence":
        """Convert numerical score to confidence level"""
        if score > 0.8:
            return cls.HIGH
        elif score >= 0.5:
            return cls.MEDIUM
        else:
            return cls.LOW


class LanguageCode(str, Enum):
    """Supported language codes (ISO 639-1)"""
    EN = "en"  # English
    ES = "es"  # Spanish
    FR = "fr"  # French
    DE = "de"  # German
    IT = "it"  # Italian
    PT = "pt"  # Portuguese
    RU = "ru"  # Russian
    ZH = "zh"  # Chinese
    JA = "ja"  # Japanese
    KO = "ko"  # Korean
    AR = "ar"  # Arabic
    HI = "hi"  # Hindi
    NL = "nl"  # Dutch
    SV = "sv"  # Swedish
    NO = "no"  # Norwegian

    @classmethod
    def get_supported_languages(cls) -> List[str]:
        """Get list of supported language codes"""
        return [lang.value for lang in cls]

    @classmethod
    def get_language_name(cls, code: str) -> str:
        """Get human-readable language name"""
        names = {
            cls.EN.value: "English",
            cls.ES.value: "Spanish",
            cls.FR.value: "French",
            cls.DE.value: "German",
            cls.IT.value: "Italian",
            cls.PT.value: "Portuguese",
            cls.RU.value: "Russian",
            cls.ZH.value: "Chinese",
            cls.JA.value: "Japanese",
            cls.KO.value: "Korean",
            cls.AR.value: "Arabic",
            cls.HI.value: "Hindi",
            cls.NL.value: "Dutch",
            cls.SV.value: "Swedish",
            cls.NO.value: "Norwegian"
        }
        return names.get(code, "Unknown")


class ErrorCode(str, Enum):
    """Standardized error codes"""
    # Validation errors
    VALIDATION_ERROR = "VALIDATION_ERROR"
    INVALID_INPUT = "INVALID_INPUT"
    MISSING_REQUIRED_FIELD = "MISSING_REQUIRED_FIELD"

    # Authentication/Authorization errors
    AUTHENTICATION_FAILED = "AUTHENTICATION_FAILED"
    AUTHORIZATION_FAILED = "AUTHORIZATION_FAILED"
    INVALID_TOKEN = "INVALID_TOKEN"
    TOKEN_EXPIRED = "TOKEN_EXPIRED"

    # Resource errors
    RESOURCE_NOT_FOUND = "RESOURCE_NOT_FOUND"
    RESOURCE_CONFLICT = "RESOURCE_CONFLICT"
    RESOURCE_LOCKED = "RESOURCE_LOCKED"

    # Rate limiting
    RATE_LIMIT_EXCEEDED = "RATE_LIMIT_EXCEEDED"
    QUOTA_EXCEEDED = "QUOTA_EXCEEDED"

    # Service errors
    INTERNAL_ERROR = "INTERNAL_ERROR"
    SERVICE_UNAVAILABLE = "SERVICE_UNAVAILABLE"
    TIMEOUT_ERROR = "TIMEOUT_ERROR"
    EXTERNAL_SERVICE_ERROR = "EXTERNAL_SERVICE_ERROR"

    # Database errors
    DATABASE_ERROR = "DATABASE_ERROR"
    CONNECTION_ERROR = "CONNECTION_ERROR"
    TRANSACTION_ERROR = "TRANSACTION_ERROR"

    # Channel-specific errors
    CHANNEL_ERROR = "CHANNEL_ERROR"
    MESSAGE_TOO_LARGE = "MESSAGE_TOO_LARGE"
    UNSUPPORTED_MEDIA_TYPE = "UNSUPPORTED_MEDIA_TYPE"
    DELIVERY_FAILED = "DELIVERY_FAILED"

# ============================================================================
# PYDANTIC MODELS FOR COMPLEX DATA STRUCTURES
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
            'image/jpeg', 'image/png', 'image/gif', 'image/webp', 'image/svg+xml',
            # Videos
            'video/mp4', 'video/quicktime', 'video/webm', 'video/avi',
            # Audio
            'audio/mpeg', 'audio/wav', 'audio/ogg', 'audio/m4a', 'audio/flac',
            # Documents
            'application/pdf', 'text/plain', 'text/csv', 'application/json',
            'application/vnd.openxmlformats-officedocument.wordprocessingml.document',
            'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
            'application/vnd.openxmlformats-officedocument.presentationml.presentation'
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
    country: Optional[str] = Field(None, max_length=100)
    city: Optional[str] = Field(None, max_length=100)
    postal_code: Optional[str] = Field(None, max_length=20)


class ContactContent(BaseModel):
    """Contact information structure"""
    name: str = Field(..., max_length=200)
    phone_number: Optional[str] = Field(None, regex=r'^\+[1-9]\d{1,14}$')
    email: Optional[str] = None
    organization: Optional[str] = Field(None, max_length=200)
    formatted_address: Optional[str] = Field(None, max_length=500)

    @validator('email')
    def validate_email(cls, v):
        if v:
            pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
            if not re.match(pattern, v.lower()):
                raise ValueError('Invalid email format')
            return v.lower()
        return v


class QuickReply(BaseModel):
    """Quick reply button structure"""
    title: str = Field(..., max_length=20)
    payload: str = Field(..., max_length=1000)
    content_type: str = Field(default="text")
    image_url: Optional[str] = None


class Button(BaseModel):
    """Interactive button structure"""
    type: str = Field(..., regex=r'^(postback|url|phone|share|login|call)$')
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
    field_type: str = Field(..., regex=r'^(text|email|phone|number|date|select|textarea|checkbox|radio)$')
    label: str = Field(..., max_length=100)
    required: bool = Field(default=False)
    placeholder: Optional[str] = Field(None, max_length=100)
    options: Optional[List[str]] = None  # For select/radio fields
    validation_regex: Optional[str] = None
    min_length: Optional[int] = None
    max_length: Optional[int] = None
    min_value: Optional[float] = None
    max_value: Optional[float] = None


class MessageContent(BaseModel):
    """Main message content structure"""
    type: MessageType
    text: Optional[str] = Field(None, max_length=4096)
    language: Optional[str] = Field(default="en", regex=r'^[a-z]{2}$')

    # Rich content
    media: Optional[MediaContent] = None
    location: Optional[LocationContent] = None
    contact: Optional[ContactContent] = None

    # Interactive elements
    quick_replies: Optional[List[QuickReply]] = Field(None, max_items=13)
    buttons: Optional[List[Button]] = Field(None, max_items=3)
    carousel: Optional[List[CarouselItem]] = Field(None, max_items=10)

    # Form content
    form_fields: Optional[List[FormField]] = None
    form_data: Optional[Dict[str, Any]] = None

    # System message specifics
    system_event: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = Field(default_factory=dict)

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

    @validator('contact')
    def contact_required_for_contact_type(cls, v, values):
        if values.get('type') == MessageType.CONTACT and not v:
            raise ValueError('Contact content required for contact messages')
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
    delivery_attempts: int = Field(default=0)

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
    tags: List[str] = Field(default_factory=list)


class IntentResult(BaseModel):
    """Intent detection result"""
    detected_intent: str
    confidence: float = Field(..., ge=0, le=1)
    alternatives: List[Dict[str, Any]] = Field(default_factory=list)
    intent_type: Optional[IntentType] = None
    confidence_level: Optional[IntentConfidence] = None

    @validator('confidence_level', always=True)
    def set_confidence_level(cls, v, values):
        confidence = values.get('confidence', 0)
        return IntentConfidence.from_score(confidence)


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

    @validator('label', always=True)
    def set_label_from_score(cls, v, values):
        score = values.get('score', 0)
        return SentimentLabel.from_score(score)


class ToxicityResult(BaseModel):
    """Content toxicity analysis result"""
    is_toxic: bool
    toxicity_score: float = Field(..., ge=0, le=1)
    categories: List[str] = Field(default_factory=list)
    severity: Optional[str] = None


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
    interaction_count: int = Field(default=0)
    last_interaction: Optional[datetime] = None


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
    satisfaction_score: Optional[float] = Field(None, ge=1, le=5)


# ============================================================================
# VALIDATION UTILITIES
# ============================================================================

class ValidationUtils:
    """Utility functions for type validation"""

    @staticmethod
    def is_valid_uuid(value: str) -> bool:
        """Check if string is a valid UUID"""
        try:
            UUID(value)
            return True
        except (ValueError, TypeError):
            return False

    @staticmethod
    def is_valid_email(email: str) -> bool:
        """Check if string is a valid email address"""
        pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
        return re.match(pattern, email) is not None

    @staticmethod
    def is_valid_phone(phone: str) -> bool:
        """Check if string is a valid phone number (E.164 format)"""
        pattern = r'^\+[1-9]\d{1,14}$'
        return re.match(pattern, phone) is not None

    @staticmethod
    def is_valid_url(url: str) -> bool:
        """Check if string is a valid URL"""
        pattern = r'^https?://[^\s/$.?#].[^\s]*$'
        return re.match(pattern, url) is not None

    @staticmethod
    def validate_tenant_id(tenant_id: str) -> bool:
        """Validate tenant ID format"""
        return ValidationUtils.is_valid_uuid(tenant_id)

    @staticmethod
    def validate_user_id(user_id: str) -> bool:
        """Validate user ID format (can be UUID or alphanumeric)"""
        if ValidationUtils.is_valid_uuid(user_id):
            return True
        # Allow alphanumeric with underscores and hyphens
        pattern = r'^[a-zA-Z0-9_-]+$'
        return re.match(pattern, user_id) is not None and len(user_id) <= 255

    @staticmethod
    def validate_message_content(content: str, message_type: MessageType) -> bool:
        """Validate message content based on type"""
        if message_type == MessageType.TEXT:
            return isinstance(content, str) and len(content.strip()) > 0 and len(content) <= 4096
        # Add other validation rules as needed
        return True

    @staticmethod
    def validate_language_code(code: str) -> bool:
        """Validate language code"""
        return code in LanguageCode.get_supported_languages()


# ============================================================================
# SERIALIZATION UTILITIES
# ============================================================================
# Serialization utilities
class SerializationUtils:
    """Utility functions for serialization"""

    @staticmethod
    def enum_to_dict(enum_value: Enum) -> Dict[str, Any]:
        """Convert enum to dictionary"""
        return {
            "value": enum_value.value,
            "name": enum_value.name,
            "type": enum_value.__class__.__name__
        }

    @staticmethod
    def serialize_datetime(dt: datetime) -> str:
        """Serialize datetime to ISO format"""
        return dt.isoformat() if dt else None

    @staticmethod
    def deserialize_datetime(dt_str: str) -> Optional[datetime]:
        """Deserialize datetime from ISO format"""
        try:
            return datetime.fromisoformat(dt_str) if dt_str else None
        except (ValueError, TypeError):
            return None

# Type guards for runtime type checking
def is_tenant_id(value: Any) -> bool:
    """Type guard for TenantId"""
    return isinstance(value, str) and ValidationUtils.validate_tenant_id(value)

def is_user_id(value: Any) -> bool:
    """Type guard for UserId"""
    return isinstance(value, str) and ValidationUtils.validate_user_id(value)

def is_message_type(value: Any) -> bool:
    """Type guard for MessageType"""
    return isinstance(value, str) and value in MessageType.get_supported_channels()

def is_channel_type(value: Any) -> bool:
    """Type guard for ChannelType"""
    return isinstance(value, str) and ChannelType.is_valid_channel(value)

# Factory functions for creating IDs
def generate_conversation_id() -> ConversationId:
    """Generate a new conversation ID"""
    return ConversationId(str(uuid.uuid4()))

def generate_message_id() -> MessageId:
    """Generate a new message ID"""
    return MessageId(str(uuid.uuid4()))

def generate_session_id() -> SessionId:
    """Generate a new session ID"""
    return SessionId(str(uuid.uuid4()))

def generate_tenant_id() -> TenantId:
    """Generate a new tenant ID"""
    return TenantId(str(uuid.uuid4()))

def generate_user_id() -> UserId:
    """Generate a new user ID"""
    return UserId(str(uuid.uuid4()))

# Common data structures
class StatusInfo:
    """Common status information structure"""

    def __init__(self, status: str, message: str = "", details: Dict[str, Any] = None):
        self.status = status
        self.message = message
        self.details = details or {}
        self.timestamp = datetime.utcnow()

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "status": self.status,
            "message": self.message,
            "details": self.details,
            "timestamp": self.timestamp.isoformat()
        }

class PaginationParams:
    """Pagination parameters with validation"""

    def __init__(self, page: int = 1, page_size: int = 20, sort_by: Optional[str] = None, sort_order: str = "desc"):
        self.page = max(1, page)
        self.page_size = min(max(1, page_size), 100)  # Limit to 100 items per page
        self.sort_by = sort_by
        self.sort_order = sort_order if sort_order in ["asc", "desc"] else "desc"

    @property
    def offset(self) -> int:
        """Calculate offset for database queries"""
        return (self.page - 1) * self.page_size

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "page": self.page,
            "page_size": self.page_size,
            "sort_by": self.sort_by,
            "sort_order": self.sort_order,
            "offset": self.offset
        }

# Export all public types and utilities
__all__ = [
    # Type aliases
    'TenantId', 'UserId', 'ConversationId', 'MessageId', 'SessionId',
    'FlowId', 'IntegrationId', 'ApiKeyId', 'Timestamp',

    # Enumerations
    'ChannelType', 'MessageType', 'MessageDirection', 'ConversationStatus',
    'SessionStatus', 'DeliveryStatus', 'Priority', 'UserRole', 'TenantPlan',
    'IntegrationType', 'ModelProvider', 'SentimentLabel', 'IntentConfidence',
    'LanguageCode', 'ErrorCode','MediaContent','MessageContent','ContactContent',

    # Validation utilities
    'ValidationUtils', 'SerializationUtils',

    # Type guards
    'is_tenant_id', 'is_user_id', 'is_message_type', 'is_channel_type',

    # Factory functions
    'generate_conversation_id', 'generate_message_id', 'generate_session_id',
    'generate_tenant_id', 'generate_user_id',

    # Common structures
    'StatusInfo', 'PaginationParams',
]