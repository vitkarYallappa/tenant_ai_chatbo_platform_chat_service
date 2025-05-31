"""
Application constants and enumerations.

This module defines all constant values, enumerations, and
configuration defaults used throughout the Chat Service.
"""

from enum import Enum
from typing import Dict, List, Tuple

# Service Information
SERVICE_NAME = "chat-service"
API_VERSION = "v2"
SERVICE_VERSION = "2.0.0"
SERVICE_DESCRIPTION = "Multi-tenant AI chatbot platform - Chat Service"

# API Configuration
API_PREFIX = f"/api/{API_VERSION}"
DEFAULT_PAGE_SIZE = 20
MAX_PAGE_SIZE = 100
MIN_PAGE_SIZE = 1

# Timeout Configuration (in seconds)
DEFAULT_TIMEOUT_MS = 30000
SHORT_TIMEOUT_MS = 5000
LONG_TIMEOUT_MS = 60000
STARTUP_TIMEOUT = 60
SHUTDOWN_TIMEOUT = 30

# Health Check Configuration
HEALTH_CHECK_INTERVAL = 30  # seconds
HEALTH_CHECK_TIMEOUT = 5    # seconds
MAX_CONSECUTIVE_FAILURES = 3

# Cache TTL Configuration (in seconds)
CACHE_TTL = {
    "session": 3600,        # 1 hour
    "user_profile": 900,    # 15 minutes
    "config": 300,          # 5 minutes
    "response": 1800,       # 30 minutes
    "rate_limit": 60,       # 1 minute
    "health_check": 10,     # 10 seconds
}

# Rate Limiting Configuration
RATE_LIMIT_WINDOWS = {
    "second": 1,
    "minute": 60,
    "hour": 3600,
    "day": 86400,
}

# Message Size Limits (in bytes)
MAX_MESSAGE_SIZE = 4096      # 4KB for text messages
MAX_MEDIA_SIZE = 52428800    # 50MB for media files
MAX_FILE_SIZE = 104857600    # 100MB for document files
MAX_AUDIO_DURATION = 600     # 10 minutes in seconds
MAX_VIDEO_DURATION = 1800    # 30 minutes in seconds

# Supported Channel Types
class ChannelType(str, Enum):
    """Supported communication channels."""
    WEB = "web"
    WHATSAPP = "whatsapp"
    MESSENGER = "messenger"
    SLACK = "slack"
    TEAMS = "teams"
    VOICE = "voice"
    SMS = "sms"


# Message Types
class MessageType(str, Enum):
    """Types of messages that can be processed."""
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


# Conversation Status Types
class ConversationStatus(str, Enum):
    """Conversation lifecycle statuses."""
    ACTIVE = "active"
    COMPLETED = "completed"
    ABANDONED = "abandoned"
    ESCALATED = "escalated"
    ERROR = "error"
    PAUSED = "paused"


# Message Direction
class MessageDirection(str, Enum):
    """Direction of message flow."""
    INBOUND = "inbound"
    OUTBOUND = "outbound"


# Processing Priority Levels
class Priority(str, Enum):
    """Message processing priority levels."""
    LOW = "low"
    NORMAL = "normal"
    HIGH = "high"
    URGENT = "urgent"
    CRITICAL = "critical"


# User Role Types
class UserRole(str, Enum):
    """User role types for authorization."""
    OWNER = "owner"
    ADMIN = "admin"
    DEVELOPER = "developer"
    MANAGER = "manager"
    MEMBER = "member"
    VIEWER = "viewer"


# Delivery Status Types
class DeliveryStatus(str, Enum):
    """Message delivery status tracking."""
    PENDING = "pending"
    SENT = "sent"
    DELIVERED = "delivered"
    READ = "read"
    FAILED = "failed"


# Error Categories
class ErrorCategory(str, Enum):
    """Error categorization for monitoring."""
    VALIDATION = "validation"
    AUTHENTICATION = "authentication"
    AUTHORIZATION = "authorization"
    NOT_FOUND = "not_found"
    RATE_LIMIT = "rate_limit"
    INTERNAL = "internal"
    EXTERNAL = "external"
    TIMEOUT = "timeout"
    NETWORK = "network"


# HTTP Status Code Mappings
HTTP_STATUS_CODES = {
    ErrorCategory.VALIDATION: 400,
    ErrorCategory.AUTHENTICATION: 401,
    ErrorCategory.AUTHORIZATION: 403,
    ErrorCategory.NOT_FOUND: 404,
    ErrorCategory.RATE_LIMIT: 429,
    ErrorCategory.INTERNAL: 500,
    ErrorCategory.EXTERNAL: 502,
    ErrorCategory.TIMEOUT: 504,
    ErrorCategory.NETWORK: 503,
}

# Supported Languages (ISO 639-1 codes)
SUPPORTED_LANGUAGES = {
    "en": "English",
    "es": "Spanish", 
    "fr": "French",
    "de": "German",
    "it": "Italian",
    "pt": "Portuguese",
    "zh": "Chinese",
    "ja": "Japanese",
    "ko": "Korean",
    "ar": "Arabic",
    "hi": "Hindi",
    "ru": "Russian",
}

# Supported MIME Types for Media
SUPPORTED_IMAGE_TYPES = {
    "image/jpeg",
    "image/jpg", 
    "image/png",
    "image/gif",
    "image/webp",
}

SUPPORTED_AUDIO_TYPES = {
    "audio/mpeg",
    "audio/mp3",
    "audio/wav",
    "audio/ogg",
    "audio/m4a",
    "audio/aac",
}

SUPPORTED_VIDEO_TYPES = {
    "video/mp4",
    "video/mpeg",
    "video/quicktime",
    "video/webm",
    "video/x-msvideo",  # .avi
}

SUPPORTED_DOCUMENT_TYPES = {
    "application/pdf",
    "application/msword",
    "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    "application/vnd.ms-excel", 
    "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    "application/vnd.ms-powerpoint",
    "application/vnd.openxmlformats-officedocument.presentationml.presentation",
    "text/plain",
    "text/csv",
}

# All supported MIME types
SUPPORTED_MEDIA_TYPES = (
    SUPPORTED_IMAGE_TYPES | 
    SUPPORTED_AUDIO_TYPES | 
    SUPPORTED_VIDEO_TYPES | 
    SUPPORTED_DOCUMENT_TYPES
)

# Database Collection Names
MONGODB_COLLECTIONS = {
    "conversations": "conversations",
    "messages": "messages", 
    "sessions": "sessions",
    "attachments": "attachments",
    "analytics": "analytics_events",
}

# Redis Key Patterns
REDIS_KEY_PATTERNS = {
    "session": "session:{tenant_id}:{session_id}",
    "conversation": "conversation:{tenant_id}:{conversation_id}",
    "rate_limit": "rate_limit:{tenant_id}:{api_key}:{window}",
    "cache": "cache:{tenant_id}:{key}",
    "lock": "lock:{resource_type}:{resource_id}",
    "health": "health:{service_name}",
    "metrics": "metrics:{tenant_id}:{metric_type}:{time_window}",
}

# Kafka Topic Names
KAFKA_TOPICS = {
    "message_received": "message.received.v1",
    "message_sent": "message.sent.v1",
    "conversation_started": "conversation.started.v1",
    "conversation_ended": "conversation.ended.v1",
    "user_action": "user.action.v1",
    "system_event": "system.event.v1",
    "analytics_event": "analytics.event.v1",
    "error_event": "error.event.v1",
}

# Webhook Event Types
WEBHOOK_EVENTS = {
    "message.received",
    "message.sent",
    "conversation.started",
    "conversation.ended",
    "delivery.status.updated",
    "user.joined",
    "user.left",
    "agent.assigned",
    "agent.unassigned",
    "escalation.created",
    "escalation.resolved",
}

# Default Configuration Values
DEFAULT_CONFIG = {
    "pagination": {
        "default_page_size": DEFAULT_PAGE_SIZE,
        "max_page_size": MAX_PAGE_SIZE,
    },
    "timeouts": {
        "request_timeout_ms": DEFAULT_TIMEOUT_MS,
        "short_timeout_ms": SHORT_TIMEOUT_MS,
        "long_timeout_ms": LONG_TIMEOUT_MS,
    },
    "limits": {
        "max_message_size": MAX_MESSAGE_SIZE,
        "max_media_size": MAX_MEDIA_SIZE,
        "max_file_size": MAX_FILE_SIZE,
        "max_audio_duration": MAX_AUDIO_DURATION,
        "max_video_duration": MAX_VIDEO_DURATION,
    },
    "cache_ttl": CACHE_TTL,
    "rate_limits": {
        "default_per_minute": 1000,
        "burst_multiplier": 2.0,
        "windows": RATE_LIMIT_WINDOWS,
    },
}

# Regex Patterns for Validation
VALIDATION_PATTERNS = {
    "email": r"^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$",
    "phone": r"^\+[1-9]\d{1,14}$",  # E.164 format
    "uuid": r"^[0-9a-f]{8}-[0-9a-f]{4}-[1-5][0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$",
    "slug": r"^[a-z0-9]+(?:-[a-z0-9]+)*$",
    "api_key": r"^cb_[a-z]{4}_[a-f0-9]{32}$",
    "tenant_subdomain": r"^[a-z0-9][a-z0-9-]{1,61}[a-z0-9]$",
    "username": r"^[a-zA-Z0-9_-]{3,30}$",
}

# Error Messages
ERROR_MESSAGES = {
    "INVALID_REQUEST": "Invalid request format or parameters",
    "UNAUTHORIZED": "Authentication required",
    "FORBIDDEN": "Insufficient permissions",
    "NOT_FOUND": "Resource not found",
    "RATE_LIMITED": "Rate limit exceeded",
    "VALIDATION_FAILED": "Request validation failed",
    "INTERNAL_ERROR": "Internal server error",
    "SERVICE_UNAVAILABLE": "Service temporarily unavailable",
    "TIMEOUT": "Request timeout",
    "INVALID_TENANT": "Invalid or inactive tenant",
    "INVALID_MESSAGE_TYPE": "Unsupported message type",
    "MESSAGE_TOO_LARGE": "Message exceeds size limit",
    "CHANNEL_NOT_SUPPORTED": "Channel not supported",
    "CONVERSATION_NOT_FOUND": "Conversation not found",
    "USER_NOT_FOUND": "User not found",
    "SESSION_EXPIRED": "Session has expired",
    "INVALID_MEDIA_TYPE": "Unsupported media type",
    "MEDIA_UPLOAD_FAILED": "Media upload failed",
    "EXTERNAL_SERVICE_ERROR": "External service error",
    "DATABASE_ERROR": "Database operation failed",
    "CACHE_ERROR": "Cache operation failed",
}

# Success Messages
SUCCESS_MESSAGES = {
    "MESSAGE_SENT": "Message sent successfully",
    "CONVERSATION_CREATED": "Conversation created successfully",
    "SESSION_CREATED": "Session created successfully",
    "USER_CREATED": "User created successfully",
    "CONFIG_UPDATED": "Configuration updated successfully",
    "MEDIA_UPLOADED": "Media uploaded successfully",
    "WEBHOOK_DELIVERED": "Webhook delivered successfully",
}

# Feature Flags (default values)
FEATURE_FLAGS = {
    "enable_rate_limiting": True,
    "enable_caching": True,
    "enable_metrics": True,
    "enable_tracing": False,
    "enable_webhooks": True,
    "enable_media_upload": True,
    "enable_file_upload": True,
    "enable_voice_messages": True,
    "enable_video_messages": True,
    "enable_location_sharing": True,
    "enable_quick_replies": True,
    "enable_carousels": True,
    "enable_forms": True,
    "enable_analytics": True,
    "enable_a_b_testing": False,
    "enable_sentiment_analysis": False,
    "enable_auto_translation": False,
    "enable_spam_detection": True,
    "enable_content_moderation": True,
}

# Channel-specific Configuration
CHANNEL_CONFIG = {
    ChannelType.WEB: {
        "supports_media": True,
        "supports_location": True,
        "supports_quick_replies": True,
        "supports_carousels": True,
        "supports_forms": True,
        "max_message_length": 4096,
        "typing_indicator": True,
        "read_receipts": True,
    },
    ChannelType.WHATSAPP: {
        "supports_media": True,
        "supports_location": True,
        "supports_quick_replies": True,
        "supports_carousels": False,
        "supports_forms": False,
        "max_message_length": 4096,
        "typing_indicator": True,
        "read_receipts": True,
        "max_buttons": 3,
        "max_quick_replies": 10,
    },
    ChannelType.MESSENGER: {
        "supports_media": True,
        "supports_location": True,
        "supports_quick_replies": True,
        "supports_carousels": True,
        "supports_forms": False,
        "max_message_length": 2000,
        "typing_indicator": True,
        "read_receipts": True,
        "max_buttons": 3,
        "max_quick_replies": 13,
    },
    ChannelType.SLACK: {
        "supports_media": True,
        "supports_location": False,
        "supports_quick_replies": False,
        "supports_carousels": False,
        "supports_forms": True,
        "max_message_length": 4000,
        "typing_indicator": False,
        "read_receipts": False,
        "supports_threading": True,
    },
    ChannelType.TEAMS: {
        "supports_media": True,
        "supports_location": False,
        "supports_quick_replies": False,
        "supports_carousels": True,
        "supports_forms": True,
        "max_message_length": 4000,
        "typing_indicator": False,
        "read_receipts": False,
        "supports_threading": True,
    },
    ChannelType.SMS: {
        "supports_media": False,
        "supports_location": False,
        "supports_quick_replies": False,
        "supports_carousels": False,
        "supports_forms": False,
        "max_message_length": 160,
        "typing_indicator": False,
        "read_receipts": False,
    },
    ChannelType.VOICE: {
        "supports_media": True,
        "supports_location": False,
        "supports_quick_replies": False,
        "supports_carousels": False,
        "supports_forms": False,
        "max_message_length": 0,  # No text length limit for voice
        "typing_indicator": False,
        "read_receipts": False,
        "supports_speech_to_text": True,
        "supports_text_to_speech": True,
    },
}

# Monitoring and Alerting Thresholds
MONITORING_THRESHOLDS = {
    "response_time_ms": {
        "warning": 1000,
        "critical": 3000,
    },
    "error_rate_percent": {
        "warning": 5.0,
        "critical": 10.0,
    },
    "memory_usage_percent": {
        "warning": 80.0,
        "critical": 95.0,
    },
    "cpu_usage_percent": {
        "warning": 80.0,
        "critical": 95.0,
    },
    "disk_usage_percent": {
        "warning": 85.0,
        "critical": 95.0,
    },
    "connection_pool_usage_percent": {
        "warning": 80.0,
        "critical": 95.0,
    },
    "queue_depth": {
        "warning": 1000,
        "critical": 5000,
    },
}

# Export collections for easy iteration
SUPPORTED_CHANNELS = list(ChannelType.__members__.values())
MESSAGE_TYPES = list(MessageType.__members__.values())
CONVERSATION_STATUSES = list(ConversationStatus.__members__.values())
USER_ROLES = list(UserRole.__members__.values())
DELIVERY_STATUSES = list(DeliveryStatus.__members__.values())
ERROR_CATEGORIES = list(ErrorCategory.__members__.values())
PRIORITIES = list(Priority.__members__.values())

# Version compatibility
API_COMPATIBILITY = {
    "v1": {
        "deprecated": True,
        "sunset_date": "2025-12-31",
        "migration_guide": "/docs/migration/v1-to-v2",
    },
    "v2": {
        "current": True,
        "stable": True,
        "introduced": "2025-01-01",
    },
}