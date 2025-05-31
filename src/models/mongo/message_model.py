"""
MongoDB Message Model
====================

MongoDB document models for message data with comprehensive
message content, AI analysis, and processing metadata support.

Features:
- Complete message document structure
- Rich media content handling
- AI analysis and processing results
- Quality assurance and moderation
- Privacy and compliance handling
"""

from datetime import datetime, timedelta
from typing import Optional, Dict, Any, List, Union
from dataclasses import dataclass, asdict, field
from enum import Enum
import structlog

from ..types import (
    TenantId, UserId, ConversationId, MessageId,
    MessageType, MessageDirection, ChannelType,
    DeliveryStatus, LanguageCode, ModelProvider
)

logger = structlog.get_logger(__name__)


@dataclass
class MediaContent:
    """Media content structure for messages"""
    url: str
    secure_url: Optional[str] = None
    type: str = "application/octet-stream"  # MIME type
    size_bytes: int = 0
    duration_ms: Optional[int] = None  # For audio/video
    dimensions: Optional[Dict[str, int]] = None  # width, height
    thumbnail_url: Optional[str] = None
    alt_text: Optional[str] = None
    caption: Optional[str] = None

    # Audio/Video specific
    transcript: Optional[str] = None
    transcript_confidence: Optional[float] = None

    # Image specific
    ocr_text: Optional[str] = None
    ocr_confidence: Optional[float] = None
    detected_objects: List[str] = field(default_factory=list)

    # File specific
    filename: Optional[str] = None
    file_extension: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "MediaContent":
        """Create from dictionary"""
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


@dataclass
class LocationContent:
    """Location content for messages"""
    latitude: float
    longitude: float
    accuracy_meters: Optional[int] = None
    address: Optional[str] = None
    place_name: Optional[str] = None
    place_id: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "LocationContent":
        """Create from dictionary"""
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


@dataclass
class QuickReply:
    """Quick reply button structure"""
    title: str
    payload: str
    content_type: str = "text"
    clicked: bool = False
    click_timestamp: Optional[datetime] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary with datetime serialization"""
        data = asdict(self)
        if data['click_timestamp']:
            data['click_timestamp'] = data['click_timestamp'].isoformat()
        return data

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "QuickReply":
        """Create from dictionary with datetime deserialization"""
        if 'click_timestamp' in data and data['click_timestamp']:
            if isinstance(data['click_timestamp'], str):
                data['click_timestamp'] = datetime.fromisoformat(data['click_timestamp'])

        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


@dataclass
class Button:
    """Button structure for interactive messages"""
    type: str  # postback, url, phone, share, login
    title: str
    payload: Optional[str] = None
    url: Optional[str] = None
    clicked: bool = False
    click_timestamp: Optional[datetime] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary with datetime serialization"""
        data = asdict(self)
        if data['click_timestamp']:
            data['click_timestamp'] = data['click_timestamp'].isoformat()
        return data

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Button":
        """Create from dictionary with datetime deserialization"""
        if 'click_timestamp' in data and data['click_timestamp']:
            if isinstance(data['click_timestamp'], str):
                data['click_timestamp'] = datetime.fromisoformat(data['click_timestamp'])

        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


@dataclass
class CarouselItem:
    """Carousel item structure"""
    title: str
    subtitle: Optional[str] = None
    image_url: Optional[str] = None
    buttons: List[Button] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        data = asdict(self)
        data['buttons'] = [button.to_dict() for button in self.buttons]
        return data

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "CarouselItem":
        """Create from dictionary"""
        if 'buttons' in data and isinstance(data['buttons'], list):
            data['buttons'] = [Button.from_dict(btn) for btn in data['buttons']]

        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


@dataclass
class FormData:
    """Form data structure"""
    form_id: str
    form_data: Dict[str, Any] = field(default_factory=dict)
    validation_status: str = "pending"  # valid, invalid, pending
    submitted_at: Optional[datetime] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary with datetime serialization"""
        data = asdict(self)
        if data['submitted_at']:
            data['submitted_at'] = data['submitted_at'].isoformat()
        return data

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "FormData":
        """Create from dictionary with datetime deserialization"""
        if 'submitted_at' in data and data['submitted_at']:
            if isinstance(data['submitted_at'], str):
                data['submitted_at'] = datetime.fromisoformat(data['submitted_at'])

        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


@dataclass
class MessageContent:
    """Comprehensive message content structure"""
    # Content type and text
    type: MessageType
    text: Optional[str] = None
    original_text: Optional[str] = None  # Before processing/translation
    translated_text: Optional[str] = None
    language: LanguageCode = LanguageCode.EN
    language_confidence: Optional[float] = None

    # Rich media content
    media: Optional[MediaContent] = None

    # Location data
    location: Optional[LocationContent] = None

    # Interactive elements
    quick_replies: List[QuickReply] = field(default_factory=list)
    buttons: List[Button] = field(default_factory=list)
    carousel: List[CarouselItem] = field(default_factory=list)

    # Form data
    form: Optional[FormData] = None

    # Context metadata
    context: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        data = {}

        # Basic fields
        data['type'] = self.type.value if hasattr(self.type, 'value') else self.type
        data['text'] = self.text
        data['original_text'] = self.original_text
        data['translated_text'] = self.translated_text
        data['language'] = self.language.value if hasattr(self.language, 'value') else self.language
        data['language_confidence'] = self.language_confidence
        data['context'] = self.context

        # Complex objects
        if self.media:
            data['media'] = self.media.to_dict()

        if self.location:
            data['location'] = self.location.to_dict()

        if self.form:
            data['form'] = self.form.to_dict()

        # Lists
        data['quick_replies'] = [qr.to_dict() for qr in self.quick_replies]
        data['buttons'] = [btn.to_dict() for btn in self.buttons]
        data['carousel'] = [item.to_dict() for item in self.carousel]

        return data

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "MessageContent":
        """Create from dictionary"""
        # Convert enums
        if 'type' in data and isinstance(data['type'], str):
            try:
                data['type'] = MessageType(data['type'])
            except ValueError:
                data['type'] = MessageType.TEXT

        if 'language' in data and isinstance(data['language'], str):
            try:
                data['language'] = LanguageCode(data['language'])
            except ValueError:
                data['language'] = LanguageCode.EN

        # Convert complex objects
        if 'media' in data and isinstance(data['media'], dict):
            data['media'] = MediaContent.from_dict(data['media'])

        if 'location' in data and isinstance(data['location'], dict):
            data['location'] = LocationContent.from_dict(data['location'])

        if 'form' in data and isinstance(data['form'], dict):
            data['form'] = FormData.from_dict(data['form'])

        # Convert lists
        if 'quick_replies' in data and isinstance(data['quick_replies'], list):
            data['quick_replies'] = [QuickReply.from_dict(qr) for qr in data['quick_replies']]

        if 'buttons' in data and isinstance(data['buttons'], list):
            data['buttons'] = [Button.from_dict(btn) for btn in data['buttons']]

        if 'carousel' in data and isinstance(data['carousel'], list):
            data['carousel'] = [CarouselItem.from_dict(item) for item in data['carousel']]

        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


@dataclass
class IntentAnalysis:
    """Intent detection results"""
    detected_intent: str
    confidence: float
    alternatives: List[Dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "IntentAnalysis":
        """Create from dictionary"""
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


@dataclass
class EntityExtraction:
    """Entity extraction results"""
    entity: str
    value: str
    start: int
    end: int
    confidence: float
    resolution: Dict[str, Any] = field(default_factory=dict)
    source: str = "user_input"  # user_input, context, integration

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "EntityExtraction":
        """Create from dictionary"""
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


@dataclass
class SentimentAnalysis:
    """Sentiment analysis results"""
    label: str  # positive, negative, neutral
    score: float  # -1 to 1
    confidence: float
    emotions: Dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SentimentAnalysis":
        """Create from dictionary"""
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


@dataclass
class ToxicityAnalysis:
    """Toxicity detection results"""
    is_toxic: bool
    toxicity_score: float
    categories: List[str] = field(default_factory=list)  # harassment, hate_speech, spam

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ToxicityAnalysis":
        """Create from dictionary"""
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


@dataclass
class QualityAnalysis:
    """Content quality analysis"""
    grammar_score: Optional[float] = None
    readability_score: Optional[float] = None
    completeness_score: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "QualityAnalysis":
        """Create from dictionary"""
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


@dataclass
class AIAnalysis:
    """Comprehensive AI analysis results"""
    intent: Optional[IntentAnalysis] = None
    entities: List[EntityExtraction] = field(default_factory=list)
    sentiment: Optional[SentimentAnalysis] = None
    topics: List[str] = field(default_factory=list)
    keywords: List[str] = field(default_factory=list)
    categories: List[str] = field(default_factory=list)
    toxicity: Optional[ToxicityAnalysis] = None
    quality: Optional[QualityAnalysis] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        data = {}

        if self.intent:
            data['intent'] = self.intent.to_dict()

        data['entities'] = [entity.to_dict() for entity in self.entities]

        if self.sentiment:
            data['sentiment'] = self.sentiment.to_dict()

        data['topics'] = self.topics
        data['keywords'] = self.keywords
        data['categories'] = self.categories

        if self.toxicity:
            data['toxicity'] = self.toxicity.to_dict()

        if self.quality:
            data['quality'] = self.quality.to_dict()

        return data

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "AIAnalysis":
        """Create from dictionary"""
        result_data = {}

        if 'intent' in data and data['intent']:
            result_data['intent'] = IntentAnalysis.from_dict(data['intent'])

        if 'entities' in data and isinstance(data['entities'], list):
            result_data['entities'] = [EntityExtraction.from_dict(e) for e in data['entities']]

        if 'sentiment' in data and data['sentiment']:
            result_data['sentiment'] = SentimentAnalysis.from_dict(data['sentiment'])

        if 'toxicity' in data and data['toxicity']:
            result_data['toxicity'] = ToxicityAnalysis.from_dict(data['toxicity'])

        if 'quality' in data and data['quality']:
            result_data['quality'] = QualityAnalysis.from_dict(data['quality'])

        # Simple fields
        for field in ['topics', 'keywords', 'categories']:
            if field in data:
                result_data[field] = data[field]

        return cls(**result_data)


@dataclass
class GenerationConfig:
    """Model generation configuration"""
    temperature: Optional[float] = None
    max_tokens: Optional[int] = None
    top_p: Optional[float] = None
    frequency_penalty: Optional[float] = None
    presence_penalty: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "GenerationConfig":
        """Create from dictionary"""
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


@dataclass
class TokenUsage:
    """Token usage information"""
    input: int = 0
    output: int = 0
    total: int = 0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TokenUsage":
        """Create from dictionary"""
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


@dataclass
class GenerationMetadata:
    """Generation metadata for outbound messages"""
    model_provider: Optional[str] = None
    model_name: Optional[str] = None
    model_version: Optional[str] = None
    generation_config: Optional[GenerationConfig] = None
    tokens_used: Optional[TokenUsage] = None
    cost_cents: Optional[float] = None
    generation_time_ms: Optional[int] = None
    fallback_used: Optional[bool] = None
    fallback_reason: Optional[str] = None
    template_used: Optional[str] = None
    personalization_applied: Optional[bool] = None
    a_b_variant: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        data = asdict(self)

        if self.generation_config:
            data['generation_config'] = self.generation_config.to_dict()

        if self.tokens_used:
            data['tokens_used'] = self.tokens_used.to_dict()

        return data

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "GenerationMetadata":
        """Create from dictionary"""
        result_data = dict(data)

        if 'generation_config' in data and data['generation_config']:
            result_data['generation_config'] = GenerationConfig.from_dict(data['generation_config'])

        if 'tokens_used' in data and data['tokens_used']:
            result_data['tokens_used'] = TokenUsage.from_dict(data['tokens_used'])

        return cls(**{k: v for k, v in result_data.items() if k in cls.__dataclass_fields__})


@dataclass
class ChannelMetadata:
    """Channel-specific metadata"""
    platform_message_id: Optional[str] = None
    platform_timestamp: Optional[datetime] = None
    thread_id: Optional[str] = None
    parent_message_id: Optional[str] = None
    reply_to_message_id: Optional[str] = None
    forwarded: bool = False
    forwarded_from: Optional[str] = None
    edited: bool = False
    edited_at: Optional[datetime] = None
    delivery_status: DeliveryStatus = DeliveryStatus.SENT
    delivery_timestamp: Optional[datetime] = None
    read_timestamp: Optional[datetime] = None
    delivery_attempts: int = 0
    failure_reason: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary with datetime serialization"""
        data = asdict(self)

        # Convert datetimes
        for field in ['platform_timestamp', 'edited_at', 'delivery_timestamp', 'read_timestamp']:
            if data[field]:
                data[field] = data[field].isoformat()

        # Convert enum
        if hasattr(data['delivery_status'], 'value'):
            data['delivery_status'] = data['delivery_status'].value

        return data

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ChannelMetadata":
        """Create from dictionary with datetime deserialization"""
        result_data = dict(data)

        # Convert datetimes
        for field in ['platform_timestamp', 'edited_at', 'delivery_timestamp', 'read_timestamp']:
            if field in result_data and result_data[field]:
                if isinstance(result_data[field], str):
                    result_data[field] = datetime.fromisoformat(result_data[field])

        # Convert enum
        if 'delivery_status' in result_data and isinstance(result_data['delivery_status'], str):
            try:
                result_data['delivery_status'] = DeliveryStatus(result_data['delivery_status'])
            except ValueError:
                result_data['delivery_status'] = DeliveryStatus.SENT

        return cls(**{k: v for k, v in result_data.items() if k in cls.__dataclass_fields__})


@dataclass
class ProcessingStage:
    """Individual processing stage information"""
    stage: str
    status: str  # success, error, skipped
    duration_ms: Optional[int] = None
    error_details: Optional[str] = None
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary with datetime serialization"""
        data = asdict(self)

        for field in ['started_at', 'completed_at']:
            if data[field]:
                data[field] = data[field].isoformat()

        return data

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ProcessingStage":
        """Create from dictionary with datetime deserialization"""
        result_data = dict(data)

        for field in ['started_at', 'completed_at']:
            if field in result_data and result_data[field]:
                if isinstance(result_data[field], str):
                    result_data[field] = datetime.fromisoformat(result_data[field])

        return cls(**{k: v for k, v in result_data.items() if k in cls.__dataclass_fields__})


@dataclass
class ProcessingMetadata:
    """Processing pipeline information"""
    pipeline_version: Optional[str] = None
    processing_stages: List[ProcessingStage] = field(default_factory=list)
    total_processing_time_ms: Optional[int] = None
    queue_time_ms: Optional[int] = None
    priority: str = "normal"  # low, normal, high, urgent, critical
    retry_count: int = 0
    last_retry_at: Optional[datetime] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        data = asdict(self)

        data['processing_stages'] = [stage.to_dict() for stage in self.processing_stages]

        if data['last_retry_at']:
            data['last_retry_at'] = data['last_retry_at'].isoformat()

        return data

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ProcessingMetadata":
        """Create from dictionary"""
        result_data = dict(data)

        if 'processing_stages' in result_data and isinstance(result_data['processing_stages'], list):
            result_data['processing_stages'] = [
                ProcessingStage.from_dict(stage) for stage in result_data['processing_stages']
            ]

        if 'last_retry_at' in result_data and result_data['last_retry_at']:
            if isinstance(result_data['last_retry_at'], str):
                result_data['last_retry_at'] = datetime.fromisoformat(result_data['last_retry_at'])

        return cls(**{k: v for k, v in result_data.items() if k in cls.__dataclass_fields__})


@dataclass
class QualityAssurance:
    """Quality assurance and feedback"""
    automated_quality_score: Optional[float] = None
    human_quality_rating: Optional[int] = None  # 1-5
    quality_feedback: Optional[str] = None
    reported_issues: List[str] = field(default_factory=list)  # accuracy, relevance, tone, grammar
    improvement_suggestions: Optional[str] = None
    reviewed_by: Optional[str] = None
    reviewed_at: Optional[datetime] = None
    approved: Optional[bool] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary with datetime serialization"""
        data = asdict(self)

        if data['reviewed_at']:
            data['reviewed_at'] = data['reviewed_at'].isoformat()

        return data

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "QualityAssurance":
        """Create from dictionary with datetime deserialization"""
        result_data = dict(data)

        if 'reviewed_at' in result_data and result_data['reviewed_at']:
            if isinstance(result_data['reviewed_at'], str):
                result_data['reviewed_at'] = datetime.fromisoformat(result_data['reviewed_at'])

        return cls(**{k: v for k, v in result_data.items() if k in cls.__dataclass_fields__})


@dataclass
class ModerationData:
    """Moderation and compliance information"""
    flagged: bool = False
    flags: List[str] = field(default_factory=list)  # spam, inappropriate, pii, toxic, off_topic
    auto_moderated: bool = False
    human_reviewed: bool = False
    approved: bool = False
    moderator_id: Optional[str] = None
    moderator_notes: Optional[str] = None
    moderated_at: Optional[datetime] = None
    escalated: bool = False
    escalation_reason: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary with datetime serialization"""
        data = asdict(self)

        if data['moderated_at']:
            data['moderated_at'] = data['moderated_at'].isoformat()

        return data

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ModerationData":
        """Create from dictionary with datetime deserialization"""
        result_data = dict(data)

        if 'moderated_at' in result_data and result_data['moderated_at']:
            if isinstance(result_data['moderated_at'], str):
                result_data['moderated_at'] = datetime.fromisoformat(result_data['moderated_at'])

        return cls(**{k: v for k, v in result_data.items() if k in cls.__dataclass_fields__})


@dataclass
class PrivacyData:
    """PII and privacy information"""
    contains_pii: bool = False
    pii_types: List[str] = field(default_factory=list)  # email, phone, ssn, credit_card, address
    masked_content: Optional[str] = None  # PII-masked version
    anonymization_level: str = "none"  # none, partial, full
    retention_category: str = "standard"  # standard, extended, permanent
    auto_delete_at: Optional[datetime] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary with datetime serialization"""
        data = asdict(self)

        if data['auto_delete_at']:
            data['auto_delete_at'] = data['auto_delete_at'].isoformat()

        return data

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "PrivacyData":
        """Create from dictionary with datetime deserialization"""
        result_data = dict(data)

        if 'auto_delete_at' in result_data and result_data['auto_delete_at']:
            if isinstance(result_data['auto_delete_at'], str):
                result_data['auto_delete_at'] = datetime.fromisoformat(result_data['auto_delete_at'])

        return cls(**{k: v for k, v in result_data.items() if k in cls.__dataclass_fields__})


@dataclass
class MessageDocument:
    """
    Complete MongoDB message document model

    This class represents the full message document structure
    matching the MongoDB schema specification.
    """
    # Core identifiers
    message_id: MessageId
    conversation_id: ConversationId
    tenant_id: TenantId
    user_id: UserId

    # Message metadata
    sequence_number: int = 0
    direction: MessageDirection = MessageDirection.INBOUND
    timestamp: datetime = field(default_factory=datetime.utcnow)
    channel: ChannelType = ChannelType.WEB

    # Content
    content: MessageContent = field(default_factory=lambda: MessageContent(type=MessageType.TEXT))

    # Metadata and analysis
    channel_metadata: Optional[ChannelMetadata] = None
    ai_analysis: Optional[AIAnalysis] = None
    generation_metadata: Optional[GenerationMetadata] = None
    processing: Optional[ProcessingMetadata] = None
    quality_assurance: Optional[QualityAssurance] = None
    moderation: Optional[ModerationData] = None
    privacy: Optional[PrivacyData] = None

    # MongoDB ObjectId (set by database)
    id: Optional[str] = None

    # Timestamps (managed by database)
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None

    # Soft delete flag
    deleted: bool = False
    deleted_at: Optional[datetime] = None

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert to dictionary for MongoDB storage

        Returns:
            Dictionary representation suitable for MongoDB
        """
        data = {}

        # Core fields
        data['message_id'] = self.message_id
        data['conversation_id'] = self.conversation_id
        data['tenant_id'] = self.tenant_id
        data['user_id'] = self.user_id
        data['sequence_number'] = self.sequence_number
        data['timestamp'] = self.timestamp
        data['deleted'] = self.deleted

        # Enum fields
        data['direction'] = self.direction.value if hasattr(self.direction, 'value') else self.direction
        data['channel'] = self.channel.value if hasattr(self.channel, 'value') else self.channel

        # Content
        data['content'] = self.content.to_dict()

        # Optional complex objects
        if self.channel_metadata:
            data['channel_metadata'] = self.channel_metadata.to_dict()

        if self.ai_analysis:
            data['ai_analysis'] = self.ai_analysis.to_dict()

        if self.generation_metadata:
            data['generation_metadata'] = self.generation_metadata.to_dict()

        if self.processing:
            data['processing'] = self.processing.to_dict()

        if self.quality_assurance:
            data['quality_assurance'] = self.quality_assurance.to_dict()

        if self.moderation:
            data['moderation'] = self.moderation.to_dict()

        if self.privacy:
            data['privacy'] = self.privacy.to_dict()

        # Timestamps
        if self.deleted_at:
            data['deleted_at'] = self.deleted_at

        # MongoDB fields
        if self.id:
            data['_id'] = self.id
        data['created_at'] = self.created_at
        data['updated_at'] = self.updated_at

        return data

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "MessageDocument":
        """
        Create from MongoDB document

        Args:
            data: MongoDB document data

        Returns:
            MessageDocument instance
        """
        # Handle MongoDB ObjectId
        if '_id' in data:
            data['id'] = str(data['_id'])
            data.pop('_id', None)

        # Convert enum strings back to enums
        if 'direction' in data and isinstance(data['direction'], str):
            try:
                data['direction'] = MessageDirection(data['direction'])
            except ValueError:
                data['direction'] = MessageDirection.INBOUND

        if 'channel' in data and isinstance(data['channel'], str):
            try:
                data['channel'] = ChannelType(data['channel'])
            except ValueError:
                data['channel'] = ChannelType.WEB

        # Convert content
        if 'content' in data and isinstance(data['content'], dict):
            data['content'] = MessageContent.from_dict(data['content'])

        # Convert complex objects
        complex_fields = {
            'channel_metadata': ChannelMetadata,
            'ai_analysis': AIAnalysis,
            'generation_metadata': GenerationMetadata,
            'processing': ProcessingMetadata,
            'quality_assurance': QualityAssurance,
            'moderation': ModerationData,
            'privacy': PrivacyData
        }

        for field, cls_type in complex_fields.items():
            if field in data and isinstance(data[field], dict):
                data[field] = cls_type.from_dict(data[field])

        # Create instance with only valid fields
        valid_fields = {k: v for k, v in data.items() if k in cls.__dataclass_fields__}
        return cls(**valid_fields)


# Utility functions for message management
def create_message(
        tenant_id: TenantId,
        user_id: UserId,
        conversation_id: ConversationId,
        content: MessageContent,
        direction: MessageDirection = MessageDirection.INBOUND,
        channel: ChannelType = ChannelType.WEB,
        sequence_number: int = 0
) -> MessageDocument:
    """
    Create a new message document

    Args:
        tenant_id: Tenant identifier
        user_id: User identifier
        conversation_id: Conversation identifier
        content: Message content
        direction: Message direction
        channel: Communication channel
        sequence_number: Message sequence in conversation

    Returns:
        New MessageDocument instance
    """
    from ..types import generate_message_id

    message_id = generate_message_id()

    message = MessageDocument(
        message_id=message_id,
        conversation_id=conversation_id,
        tenant_id=tenant_id,
        user_id=user_id,
        content=content,
        direction=direction,
        channel=channel,
        sequence_number=sequence_number
    )

    return message


def create_text_message(
        tenant_id: TenantId,
        user_id: UserId,
        conversation_id: ConversationId,
        text: str,
        direction: MessageDirection = MessageDirection.INBOUND,
        channel: ChannelType = ChannelType.WEB,
        language: LanguageCode = LanguageCode.EN
) -> MessageDocument:
    """
    Create a text message document

    Args:
        tenant_id: Tenant identifier
        user_id: User identifier
        conversation_id: Conversation identifier
        text: Message text
        direction: Message direction
        channel: Communication channel
        language: Message language

    Returns:
        New MessageDocument instance with text content
    """
    content = MessageContent(
        type=MessageType.TEXT,
        text=text,
        language=language
    )

    return create_message(
        tenant_id=tenant_id,
        user_id=user_id,
        conversation_id=conversation_id,
        content=content,
        direction=direction,
        channel=channel
    )