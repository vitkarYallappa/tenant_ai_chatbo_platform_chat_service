# src/models/__init__.py
"""
Models package for the Chat Service.
Provides data models, types, and schemas for the application.
"""

from src.models.types import (
    # Enums
    ChannelType,
    MessageType,
    ConversationStatus,
    DeliveryStatus,
    ProcessingStage,
    UserRole,
    IntentType,
    SentimentLabel,
    Priority,

    # Type aliases
    TenantId,
    UserId,
    ConversationId,
    MessageId,
    SessionId,
    FlowId,
    IntegrationId,
    ApiKeyId,

    # Content models
    MediaContent,
    LocationContent,
    QuickReply,
    Button,
    CarouselItem,
    FormField,
    MessageContent,
    ChannelMetadata,
    ProcessingHints,

    # AI models
    IntentResult,
    EntityResult,
    SentimentResult,
    ToxicityResult,

    # Business models
    UserInfo,
    BusinessContext,

    # Response models
    PaginatedResponse,
    APIResponse,
    ErrorDetail,

    # Validation helpers
    PhoneNumber,
    EmailAddress,

    # Utility functions
    generate_message_id,
    generate_conversation_id,
    generate_session_id,
    is_valid_uuid
)

from src.models.base_model import (
    BaseMongoModel,
    BaseRedisModel,
    BaseRequestModel,
    BaseResponseModel,
    TimestampMixin,
    AuditMixin,
    SoftDeleteMixin,
    ValidationHelpers
)

# MongoDB models
from src.models.mongo.conversation_model import (
    ConversationDocument,
    ConversationMetrics,
    ConversationContext,
    ConversationSummary,
    StateTransition,
    AIMetadata,
    ComplianceInfo
)

from src.models.mongo.message_model import (
    MessageDocument,
    ProcessingMetadata,
    ProcessingStageInfo,
    AIAnalysis,
    GenerationMetadata,
    QualityAssurance,
    ModerationInfo,
    PrivacyInfo
)

from src.models.mongo.session_model import (
    SessionDocument,
    SessionMetrics,
    ConversationRef,
    SessionPreferences,
    SessionContext,
    SecurityInfo,
    DeviceInfo,
    LocationInfo
)

# Redis models
from src.models.redis.session_cache import (
    SessionCache,
    ConversationState,
    ActiveConversations,
    SessionCacheManager
)

from src.models.redis.rate_limit_cache import (
    RateLimitWindow,
    TokenBucket,
    RateLimitTier,
    ConcurrentRequestTracker,
    RateLimitViolation,
    DEFAULT_RATE_LIMIT_TIERS
)

# Version information
__version__ = "1.0.0"

# Export all models and utilities
__all__ = [
    # Core types
    "ChannelType",
    "MessageType",
    "ConversationStatus",
    "DeliveryStatus",
    "ProcessingStage",
    "UserRole",
    "IntentType",
    "SentimentLabel",
    "Priority",

    # Type aliases
    "TenantId",
    "UserId",
    "ConversationId",
    "MessageId",
    "SessionId",
    "FlowId",
    "IntegrationId",
    "ApiKeyId",

    # Content models
    "MediaContent",
    "LocationContent",
    "QuickReply",
    "Button",
    "CarouselItem",
    "FormField",
    "MessageContent",
    "ChannelMetadata",
    "ProcessingHints",

    # AI models
    "IntentResult",
    "EntityResult",
    "SentimentResult",
    "ToxicityResult",

    # Business models
    "UserInfo",
    "BusinessContext",

    # Response models
    "PaginatedResponse",
    "APIResponse",
    "ErrorDetail",

    # Validation helpers
    "PhoneNumber",
    "EmailAddress",

    # Base models
    "BaseMongoModel",
    "BaseRedisModel",
    "BaseRequestModel",
    "BaseResponseModel",
    "TimestampMixin",
    "AuditMixin",
    "SoftDeleteMixin",
    "ValidationHelpers",

    # MongoDB models
    "ConversationDocument",
    "ConversationMetrics",
    "ConversationContext",
    "ConversationSummary",
    "StateTransition",
    "AIMetadata",
    "ComplianceInfo",
    "MessageDocument",
    "ProcessingMetadata",
    "ProcessingStageInfo",
    "AIAnalysis",
    "GenerationMetadata",
    "QualityAssurance",
    "ModerationInfo",
    "PrivacyInfo",
    "SessionDocument",
    "SessionMetrics",
    "ConversationRef",
    "SessionPreferences",
    "SessionContext",
    "SecurityInfo",
    "DeviceInfo",
    "LocationInfo",

    # Redis models
    "SessionCache",
    "ConversationState",
    "ActiveConversations",
    "SessionCacheManager",
    "RateLimitWindow",
    "TokenBucket",
    "RateLimitTier",
    "ConcurrentRequestTracker",
    "RateLimitViolation",
    "DEFAULT_RATE_LIMIT_TIERS",

    # Utility functions
    "generate_message_id",
    "generate_conversation_id",
    "generate_session_id",
    "is_valid_uuid"
]