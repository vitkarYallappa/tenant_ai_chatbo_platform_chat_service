# src/models/mongo/__init__.py
"""
MongoDB document models for the Chat Service.
Provides document structures for conversations, messages, and sessions.
"""

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

__all__ = [
    # Conversation models
    "ConversationDocument",
    "ConversationMetrics",
    "ConversationContext",
    "ConversationSummary",
    "StateTransition",
    "AIMetadata",
    "ComplianceInfo",

    # Message models
    "MessageDocument",
    "ProcessingMetadata",
    "ProcessingStageInfo",
    "AIAnalysis",
    "GenerationMetadata",
    "QualityAssurance",
    "ModerationInfo",
    "PrivacyInfo",

    # Session models
    "SessionDocument",
    "SessionMetrics",
    "ConversationRef",
    "SessionPreferences",
    "SessionContext",
    "SecurityInfo",
    "DeviceInfo",
    "LocationInfo"
]