"""
Data Models Package
==================

Centralized data models for the chat service including MongoDB documents,
Redis cache structures, and common type definitions.

Features:
- MongoDB document models with full schema support
- Redis cache models with serialization
- Common type definitions and enumerations
- Validation utilities
- Model factories and utilities
"""

# Import common types and enumerations
from .types import (
    # Type aliases
    TenantId, UserId, ConversationId, MessageId, SessionId,
    FlowId, IntegrationId, ApiKeyId, Timestamp,

    # Enumerations
    ChannelType, MessageType, MessageDirection, ConversationStatus,
    SessionStatus, DeliveryStatus, Priority, UserRole, TenantPlan,
    IntegrationType, ModelProvider, SentimentLabel, IntentConfidence,
    LanguageCode, ErrorCode,

    # Validation utilities
    ValidationUtils, SerializationUtils,

    # Type guards
    is_tenant_id, is_user_id, is_message_type, is_channel_type,

    # Factory functions
    generate_conversation_id, generate_message_id, generate_session_id,
    generate_tenant_id, generate_user_id,

    # Common structures
    StatusInfo, PaginationParams,
)

# Import MongoDB models
from .mongo.conversation_model import (
    # Main document model
    ConversationDocument,

    # Component models
    ChannelMetadata as ConversationChannelMetadata,
    ConversationContext,
    UserInfo,
    ConversationMetrics,
    AIMetadata as ConversationAIMetadata,
    BusinessContext,
    ComplianceData,
    IntegrationUsage,
    ConversationSummary,
    ABTestingInfo,
    StateHistory,

    # Utility functions
    create_conversation,
)

from .mongo.message_model import (
    # Main document model
    MessageDocument,

    # Content models
    MessageContent,
    MediaContent,
    LocationContent,
    QuickReply,
    Button,
    CarouselItem,
    FormData,

    # Analysis models
    AIAnalysis as MessageAIAnalysis,
    IntentAnalysis,
    EntityExtraction,
    SentimentAnalysis,
    ToxicityAnalysis,
    QualityAnalysis,

    # Generation models
    GenerationMetadata,
    GenerationConfig,
    TokenUsage,

    # Processing models
    ChannelMetadata as MessageChannelMetadata,
    ProcessingMetadata,
    ProcessingStage,
    QualityAssurance,
    ModerationData,
    PrivacyData,

    # Utility functions
    create_message,
    create_text_message,
)

# Import Redis models
from .redis.session_cache import (
    # Main session model
    SessionData,

    # Component models
    SessionStatus,
    SessionContext,
    DeviceInfo,
    LocationInfo,
    UserPreferences,
    SecurityMetadata,

    # Utility functions
    create_session,
    parse_user_agent,
)

# Model registry for dynamic model access
_model_registry = {
    # MongoDB models
    'conversation': ConversationDocument,
    'message': MessageDocument,

    # Redis models
    'session': SessionData,

    # Component models
    'conversation_context': ConversationContext,
    'message_content': MessageContent,
    'ai_analysis': MessageAIAnalysis,
    'generation_metadata': GenerationMetadata,
    'user_preferences': UserPreferences,
    'device_info': DeviceInfo,
}


def get_model_class(model_name: str):
    """
    Get model class by name

    Args:
        model_name: Name of the model

    Returns:
        Model class

    Raises:
        ValueError: If model not found
    """
    if model_name not in _model_registry:
        raise ValueError(f"Unknown model: {model_name}")

    return _model_registry[model_name]


def register_model(model_name: str, model_class: type) -> None:
    """
    Register a custom model class

    Args:
        model_name: Unique identifier for the model
        model_class: Model class
    """
    _model_registry[model_name] = model_class


def list_available_models() -> list:
    """
    List all available model names

    Returns:
        List of model names
    """
    return list(_model_registry.keys())


# Model validation utilities
class ModelValidator:
    """Utility class for model validation"""

    @staticmethod
    def validate_conversation(conversation: ConversationDocument) -> bool:
        """
        Validate conversation document

        Args:
            conversation: Conversation document to validate

        Returns:
            True if valid

        Raises:
            ValueError: If validation fails
        """
        if not conversation.conversation_id:
            raise ValueError("conversation_id is required")

        if not conversation.tenant_id:
            raise ValueError("tenant_id is required")

        if not conversation.user_id:
            raise ValueError("user_id is required")

        if not ValidationUtils.validate_tenant_id(conversation.tenant_id):
            raise ValueError("Invalid tenant_id format")

        if not ValidationUtils.validate_user_id(conversation.user_id):
            raise ValueError("Invalid user_id format")

        return True

    @staticmethod
    def validate_message(message: MessageDocument) -> bool:
        """
        Validate message document

        Args:
            message: Message document to validate

        Returns:
            True if valid

        Raises:
            ValueError: If validation fails
        """
        if not message.message_id:
            raise ValueError("message_id is required")

        if not message.conversation_id:
            raise ValueError("conversation_id is required")

        if not message.tenant_id:
            raise ValueError("tenant_id is required")

        if not message.user_id:
            raise ValueError("user_id is required")

        if not message.content:
            raise ValueError("content is required")

        # Validate content based on type
        if message.content.type == MessageType.TEXT and not message.content.text:
            raise ValueError("text content required for text messages")

        return True

    @staticmethod
    def validate_session(session: SessionData) -> bool:
        """
        Validate session data

        Args:
            session: Session data to validate

        Returns:
            True if valid

        Raises:
            ValueError: If validation fails
        """
        if not session.session_id:
            raise ValueError("session_id is required")

        if not session.tenant_id:
            raise ValueError("tenant_id is required")

        if not session.user_id:
            raise ValueError("user_id is required")

        if session.expires_at <= session.created_at:
            raise ValueError("expires_at must be after created_at")

        return True


# Model serialization utilities
class ModelSerializer:
    """Utility class for model serialization"""

    @staticmethod
    def serialize_for_api(model_instance) -> dict:
        """
        Serialize model instance for API response

        Args:
            model_instance: Model instance to serialize

        Returns:
            Dictionary representation for API
        """
        if hasattr(model_instance, 'to_dict'):
            return model_instance.to_dict()
        elif hasattr(model_instance, '__dict__'):
            return vars(model_instance)
        else:
            raise ValueError(f"Cannot serialize model of type {type(model_instance)}")

    @staticmethod
    def deserialize_from_api(model_class: type, data: dict):
        """
        Deserialize API data to model instance

        Args:
            model_class: Target model class
            data: Dictionary data from API

        Returns:
            Model instance
        """
        if hasattr(model_class, 'from_dict'):
            return model_class.from_dict(data)
        else:
            # Try to create instance directly
            return model_class(**data)


# Export all public components
__all__ = [
    # Common types and utilities
    'TenantId', 'UserId', 'ConversationId', 'MessageId', 'SessionId',
    'FlowId', 'IntegrationId', 'ApiKeyId', 'Timestamp',
    'ChannelType', 'MessageType', 'MessageDirection', 'ConversationStatus',
    'SessionStatus', 'DeliveryStatus', 'Priority', 'UserRole', 'TenantPlan',
    'IntegrationType', 'ModelProvider', 'SentimentLabel', 'IntentConfidence',
    'LanguageCode', 'ErrorCode',
    'ValidationUtils', 'SerializationUtils',
    'is_tenant_id', 'is_user_id', 'is_message_type', 'is_channel_type',
    'generate_conversation_id', 'generate_message_id', 'generate_session_id',
    'generate_tenant_id', 'generate_user_id',
    'StatusInfo', 'PaginationParams',

    # MongoDB models
    'ConversationDocument', 'MessageDocument',
    'ConversationChannelMetadata', 'ConversationContext', 'UserInfo',
    'ConversationMetrics', 'ConversationAIMetadata', 'BusinessContext',
    'ComplianceData', 'IntegrationUsage', 'ConversationSummary',
    'ABTestingInfo', 'StateHistory',
    'MessageContent', 'MediaContent', 'LocationContent',
    'QuickReply', 'Button', 'CarouselItem', 'FormData',
    'MessageAIAnalysis', 'IntentAnalysis', 'EntityExtraction',
    'SentimentAnalysis', 'ToxicityAnalysis', 'QualityAnalysis',
    'GenerationMetadata', 'GenerationConfig', 'TokenUsage',
    'MessageChannelMetadata', 'ProcessingMetadata', 'ProcessingStage',
    'QualityAssurance', 'ModerationData', 'PrivacyData',

    # Redis models
    'SessionData', 'SessionContext', 'DeviceInfo', 'LocationInfo',
    'UserPreferences', 'SecurityMetadata',

    # Factory functions
    'create_conversation', 'create_message', 'create_text_message',
    'create_session', 'parse_user_agent',

    # Registry and utilities
    'get_model_class', 'register_model', 'list_available_models',
    'ModelValidator', 'ModelSerializer',
]

# Version information
__version__ = "1.0.0"
__author__ = "Chat Service Team"
__description__ = "Data models for multi-tenant chat service"