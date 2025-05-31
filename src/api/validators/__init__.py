"""
API Validators Package
Provides validation models for all API endpoints in the Chat Service.
"""

from .message_validators import (
    SendMessageRequest,
    ConversationContext,
    ProcessingMetadata,
    MessageResponse,
    ConversationSummary,
    ConversationHistoryResponse,
    BulkMessageRequest,
    BulkMessageResponse,
    MessageStatusUpdate,
    WebhookEvent,
    MessageFeedback,
    MessageAnalytics
)

from .conversation_validators import (
    ConversationCreateRequest,
    ConversationUpdateRequest,
    ConversationCloseRequest,
    ConversationResponse,
    ConversationListFilters,
    ConversationListRequest,
    ConversationListResponse,
    ConversationExportRequest,
    ConversationAnalyticsRequest,
    ConversationAnalyticsResponse,
    ConversationTransferRequest
)

from .common_validators import (
    PhoneNumberValidator,
    EmailValidator,
    URLValidator,
    PaginationParams,
    DateRangeFilter,
    SearchParams,
    MetadataValidator,
    TagsValidator,
    TimeRangeValidator,
    LanguageValidator,
    CurrencyValidator,
    FileValidator,
    ConfigurationValidator,
    BulkOperationValidator
)

__all__ = [
    # Message Validators
    "SendMessageRequest",
    "ConversationContext",
    "ProcessingMetadata",
    "MessageResponse",
    "ConversationSummary",
    "ConversationHistoryResponse",
    "BulkMessageRequest",
    "BulkMessageResponse",
    "MessageStatusUpdate",
    "WebhookEvent",
    "MessageFeedback",
    "MessageAnalytics",

    # Conversation Validators
    "ConversationCreateRequest",
    "ConversationUpdateRequest",
    "ConversationCloseRequest",
    "ConversationResponse",
    "ConversationListFilters",
    "ConversationListRequest",
    "ConversationListResponse",
    "ConversationExportRequest",
    "ConversationAnalyticsRequest",
    "ConversationAnalyticsResponse",
    "ConversationTransferRequest",

    # Common Validators
    "PhoneNumberValidator",
    "EmailValidator",
    "URLValidator",
    "PaginationParams",
    "DateRangeFilter",
    "SearchParams",
    "MetadataValidator",
    "TagsValidator",
    "TimeRangeValidator",
    "LanguageValidator",
    "CurrencyValidator",
    "FileValidator",
    "ConfigurationValidator",
    "BulkOperationValidator"
]