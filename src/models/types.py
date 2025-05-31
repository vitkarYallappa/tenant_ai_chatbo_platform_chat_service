"""
Common Types and Enumerations
============================

Centralized type definitions and enumerations used across the chat service
for consistency and type safety.

Features:
- Common enumeration types
- Type aliases for clarity
- Validation utilities
- Serialization support
"""

from enum import Enum, IntEnum
from typing import TypeVar, NewType, Union, List, Dict, Any, Optional
from datetime import datetime
import re
import uuid

# Generic type variables
T = TypeVar('T')
K = TypeVar('K')
V = TypeVar('V')

# Type aliases for better code readability
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

class TenantPlan(str, Enum):
    """Tenant subscription plans"""
    FREE = "free"
    STARTER = "starter"
    PROFESSIONAL = "professional"
    ENTERPRISE = "enterprise"
    CUSTOM = "custom"

class IntegrationType(str, Enum):
    """Integration types for external services"""
    REST_API = "rest_api"
    GRAPHQL = "graphql"
    WEBHOOK = "webhook"
    DATABASE = "database"
    OAUTH2 = "oauth2"
    CUSTOM = "custom"

class ModelProvider(str, Enum):
    """AI model providers"""
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    GOOGLE = "google"
    AZURE = "azure"
    HUGGINGFACE = "huggingface"
    CUSTOM = "custom"

class SentimentLabel(str, Enum):
    """Sentiment analysis labels"""
    POSITIVE = "positive"
    NEGATIVE = "negative"
    NEUTRAL = "neutral"
    MIXED = "mixed"

class IntentConfidence(str, Enum):
    """Intent confidence levels"""
    HIGH = "high"      # > 0.8
    MEDIUM = "medium"  # 0.5 - 0.8
    LOW = "low"        # < 0.5

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

    @classmethod
    def get_supported_languages(cls) -> List[str]:
        """Get list of supported language codes"""
        return [lang.value for lang in cls]

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

# Validation utilities
class ValidationUtils:
    """Utility functions for type validation"""

    @staticmethod
    def is_valid_uuid(value: str) -> bool:
        """Check if string is a valid UUID"""
        try:
            uuid.UUID(value)
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
    'LanguageCode', 'ErrorCode',

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