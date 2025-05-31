# src/models/schemas/__init__.py
"""
Schemas package for the Chat Service.
Provides request/response validation schemas and specialized validators.
"""

# Request schemas
from src.models.schemas.request_schemas import (
    # Message schemas
    SendMessageRequest,
    MessageFeedbackRequest,

    # Conversation schemas
    CreateConversationRequest,
    ConversationFilterRequest,

    # Session schemas
    CreateSessionRequest,
    UpdateSessionRequest,

    # Analytics schemas
    AnalyticsQueryRequest,

    # Health check schemas
    HealthCheckRequest,

    # Admin schemas
    BulkOperationRequest,
    ConfigUpdateRequest,

    # Utility schemas
    PaginationRequest,
    SortingRequest,
    DateRangeRequest
)

# Response schemas
from src.models.schemas.response_schemas import (
    # Message responses
    MessageResponse,
    SendMessageResponse,
    MessageListResponse,

    # Conversation responses
    ConversationSummary,
    ConversationResponse,
    ConversationListResponse,

    # Session responses
    SessionResponse,

    # Analytics responses
    MetricDataPoint,
    MetricSeries,
    AnalyticsResponse,

    # Health check responses
    ServiceHealth,
    HealthCheckResponse,

    # Error responses
    ErrorResponse,
    ValidationErrorResponse,

    # Success responses
    SuccessResponse,
    CreatedResponse,
    UpdatedResponse,
    DeletedResponse,

    # Admin responses
    BulkOperationResponse,

    # Generic wrapper responses
    PaginatedConversationResponse,
    PaginatedMessageResponse,
    MessageAPIResponse,
    ConversationAPIResponse,
    SessionAPIResponse,
    AnalyticsAPIResponse
)

# Validation schemas
from src.models.schemas.validation_schemas import (
    # Result classes
    ValidationResult,
    FieldValidationResult,

    # Business rule validation
    MessageContentValidation,
    RateLimitValidation,

    # Security validation
    SecurityValidation,

    # Data integrity validation
    ConversationIntegrityValidation,
    SessionIntegrityValidation,

    # Custom validators
    CustomValidator,
    RegexValidator,
    LengthValidator,
    RangeValidator,

    # Rule sets
    ValidationRuleSet,
    USER_ID_RULES,
    EMAIL_RULES,
    PHONE_RULES
)

# Version information
__version__ = "1.0.0"

# Export all schemas
__all__ = [
    # Request schemas
    "SendMessageRequest",
    "MessageFeedbackRequest",
    "CreateConversationRequest",
    "ConversationFilterRequest",
    "CreateSessionRequest",
    "UpdateSessionRequest",
    "AnalyticsQueryRequest",
    "HealthCheckRequest",
    "BulkOperationRequest",
    "ConfigUpdateRequest",
    "PaginationRequest",
    "SortingRequest",
    "DateRangeRequest",

    # Response schemas
    "MessageResponse",
    "SendMessageResponse",
    "MessageListResponse",
    "ConversationSummary",
    "ConversationResponse",
    "ConversationListResponse",
    "SessionResponse",
    "MetricDataPoint",
    "MetricSeries",
    "AnalyticsResponse",
    "ServiceHealth",
    "HealthCheckResponse",
    "ErrorResponse",
    "ValidationErrorResponse",
    "SuccessResponse",
    "CreatedResponse",
    "UpdatedResponse",
    "DeletedResponse",
    "BulkOperationResponse",
    "PaginatedConversationResponse",
    "PaginatedMessageResponse",
    "MessageAPIResponse",
    "ConversationAPIResponse",
    "SessionAPIResponse",
    "AnalyticsAPIResponse",

    # Validation schemas and utilities
    "ValidationResult",
    "FieldValidationResult",
    "MessageContentValidation",
    "RateLimitValidation",
    "SecurityValidation",
    "ConversationIntegrityValidation",
    "SessionIntegrityValidation",
    "CustomValidator",
    "RegexValidator",
    "LengthValidator",
    "RangeValidator",
    "ValidationRuleSet",
    "USER_ID_RULES",
    "EMAIL_RULES",
    "PHONE_RULES"
]