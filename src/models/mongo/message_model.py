# src/models/mongo/message_model.py
"""
MongoDB document model for message storage.
Represents individual messages within conversations with comprehensive metadata.
"""

from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional
from pydantic import BaseModel, Field, validator
from bson import ObjectId

from src.models.base_model import BaseMongoModel, TimestampMixin
from src.models.types import (
    MessageType, DeliveryStatus, ProcessingStage, Priority,
    TenantId, UserId, ConversationId, MessageId,
    MessageContent, ChannelMetadata, IntentResult,
    EntityResult, SentimentResult, ToxicityResult
)


class ProcessingStageInfo(BaseModel):
    """Information about a processing stage"""
    stage: ProcessingStage
    status: str = Field(..., regex=r'^(success|error|skipped|pending)$')
    duration_ms: Optional[int] = Field(None, ge=0)
    error_details: Optional[str] = Field(None, max_length=1000)
    started_at: datetime = Field(default_factory=datetime.utcnow)
    completed_at: Optional[datetime] = None

    def complete_stage(self, status: str = "success", error_details: Optional[str] = None) -> None:
        """Mark processing stage as complete"""
        self.status = status
        self.completed_at = datetime.utcnow()
        if error_details:
            self.error_details = error_details

        if self.started_at and self.completed_at:
            delta = self.completed_at - self.started_at
            self.duration_ms = int(delta.total_seconds() * 1000)


class ProcessingMetadata(BaseModel):
    """Message processing pipeline information"""
    pipeline_version: str = Field(default="1.0", max_length=20)
    processing_stages: List[ProcessingStageInfo] = Field(default_factory=list)
    total_processing_time_ms: Optional[int] = Field(None, ge=0)
    queue_time_ms: Optional[int] = Field(None, ge=0)
    priority: Priority = Field(default=Priority.NORMAL)
    retry_count: int = Field(default=0, ge=0)
    last_retry_at: Optional[datetime] = None

    def add_stage(self, stage: ProcessingStage) -> ProcessingStageInfo:
        """Add a new processing stage"""
        stage_info = ProcessingStageInfo(stage=stage)
        self.processing_stages.append(stage_info)
        return stage_info

    def get_current_stage(self) -> Optional[ProcessingStageInfo]:
        """Get the current (last) processing stage"""
        return self.processing_stages[-1] if self.processing_stages else None

    def calculate_total_time(self) -> None:
        """Calculate total processing time from all completed stages"""
        total_ms = sum(
            stage.duration_ms for stage in self.processing_stages
            if stage.duration_ms is not None
        )
        self.total_processing_time_ms = total_ms


class AIAnalysis(BaseModel):
    """AI processing results for the message"""
    # Intent detection
    intent: Optional[IntentResult] = None

    # Entity extraction
    entities: List[EntityResult] = Field(default_factory=list)

    # Sentiment analysis
    sentiment: Optional[SentimentResult] = None

    # Topic and keyword extraction
    topics: List[str] = Field(default_factory=list, max_items=10)
    keywords: List[str] = Field(default_factory=list, max_items=20)
    categories: List[str] = Field(default_factory=list, max_items=5)

    # Content analysis
    toxicity: Optional[ToxicityResult] = None

    # Quality metrics
    quality: Dict[str, float] = Field(default_factory=dict)

    def add_entity(self, entity_type: str, value: str, start_pos: int, end_pos: int, confidence: float) -> None:
        """Add an extracted entity"""
        entity = EntityResult(
            entity_type=entity_type,
            entity_value=value,
            start_pos=start_pos,
            end_pos=end_pos,
            confidence=confidence
        )
        self.entities.append(entity)

    def set_intent(self, intent: str, confidence: float, alternatives: List[Dict[str, Any]] = None) -> None:
        """Set the detected intent"""
        self.intent = IntentResult(
            detected_intent=intent,
            confidence=confidence,
            alternatives=alternatives or []
        )


class GenerationMetadata(BaseModel):
    """Metadata for generated (outbound) messages"""
    model_provider: Optional[str] = Field(None, max_length=50)
    model_name: Optional[str] = Field(None, max_length=100)
    model_version: Optional[str] = Field(None, max_length=50)

    generation_config: Dict[str, Any] = Field(default_factory=dict)

    tokens_used: Dict[str, int] = Field(default_factory=dict)  # input, output, total
    cost_cents: Optional[float] = Field(None, ge=0)
    generation_time_ms: Optional[int] = Field(None, ge=0)

    fallback_used: bool = Field(default=False)
    fallback_reason: Optional[str] = Field(None, max_length=200)

    template_used: Optional[str] = Field(None, max_length=100)
    personalization_applied: bool = Field(default=False)
    a_b_variant: Optional[str] = Field(None, max_length=50)

    def set_token_usage(self, input_tokens: int, output_tokens: int) -> None:
        """Set token usage information"""
        self.tokens_used = {
            "input": input_tokens,
            "output": output_tokens,
            "total": input_tokens + output_tokens
        }


class QualityAssurance(BaseModel):
    """Quality assurance and feedback information"""
    automated_quality_score: Optional[float] = Field(None, ge=0, le=1)
    human_quality_rating: Optional[int] = Field(None, ge=1, le=5)
    quality_feedback: Optional[str] = Field(None, max_length=1000)

    reported_issues: List[str] = Field(default_factory=list)  # accuracy, relevance, tone, grammar
    improvement_suggestions: Optional[str] = Field(None, max_length=500)

    reviewed_by: Optional[str] = Field(None, max_length=100)
    reviewed_at: Optional[datetime] = None
    approved: bool = Field(default=True)


class ModerationInfo(BaseModel):
    """Content moderation information"""
    flagged: bool = Field(default=False)
    flags: List[str] = Field(default_factory=list)  # spam, inappropriate, pii, toxic, off_topic
    auto_moderated: bool = Field(default=False)
    human_reviewed: bool = Field(default=False)
    approved: bool = Field(default=True)

    moderator_id: Optional[str] = Field(None, max_length=100)
    moderator_notes: Optional[str] = Field(None, max_length=500)
    moderated_at: Optional[datetime] = None

    escalated: bool = Field(default=False)
    escalation_reason: Optional[str] = Field(None, max_length=200)


class PrivacyInfo(BaseModel):
    """Privacy and PII information"""
    contains_pii: bool = Field(default=False)
    pii_types: List[str] = Field(default_factory=list)  # email, phone, ssn, credit_card, address
    masked_content: Optional[str] = Field(None, max_length=5000)  # PII-masked version

    anonymization_level: str = Field(default="none")  # none, partial, full
    retention_category: str = Field(default="standard")  # standard, extended, permanent
    auto_delete_at: Optional[datetime] = None

    def set_auto_deletion(self, days: int) -> None:
        """Set automatic deletion date"""
        self.auto_delete_at = datetime.utcnow() + timedelta(days=days)

    def should_be_deleted(self) -> bool:
        """Check if message should be auto-deleted"""
        if not self.auto_delete_at:
            return False
        return datetime.utcnow() > self.auto_delete_at


class MessageDocument(BaseMongoModel, TimestampMixin):
    """
    MongoDB document structure for messages.
    Comprehensive message storage with content, metadata, and analysis.
    """

    # Core identifiers
    message_id: MessageId = Field(..., max_length=100)
    conversation_id: ConversationId = Field(..., max_length=100)
    tenant_id: TenantId = Field(..., max_length=100)
    user_id: UserId = Field(..., max_length=255)

    # Message ordering and metadata
    sequence_number: int = Field(..., ge=1)
    direction: str = Field(..., regex=r'^(inbound|outbound)$')
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    channel: str = Field(..., max_length=50)

    # Content
    content: MessageContent

    # Original and processed content
    original_text: Optional[str] = Field(None, max_length=5000)
    translated_text: Optional[str] = Field(None, max_length=5000)
    language: str = Field(default="en", max_length=5)
    language_confidence: Optional[float] = Field(None, ge=0, le=1)

    # AI analysis results
    ai_analysis: AIAnalysis = Field(default_factory=AIAnalysis)

    # Generation metadata (for outbound messages)
    generation_metadata: Optional[GenerationMetadata] = None

    # Channel-specific metadata
    channel_metadata: Optional[ChannelMetadata] = Field(default_factory=ChannelMetadata)

    # Processing pipeline information
    processing: ProcessingMetadata = Field(default_factory=ProcessingMetadata)

    # Quality assurance and feedback
    quality_assurance: QualityAssurance = Field(default_factory=QualityAssurance)

    # Moderation and compliance
    moderation: ModerationInfo = Field(default_factory=ModerationInfo)

    # Privacy and PII
    privacy: PrivacyInfo = Field(default_factory=PrivacyInfo)

    # Parent/reply relationships
    parent_message_id: Optional[MessageId] = Field(None, max_length=100)
    reply_to_message_id: Optional[MessageId] = Field(None, max_length=100)
    thread_id: Optional[str] = Field(None, max_length=100)

    # Edit tracking
    edited: bool = Field(default=False)
    edited_at: Optional[datetime] = None
    edit_history: List[Dict[str, Any]] = Field(default_factory=list)

    class Config:
        collection_name = "messages"

    @validator('message_id', 'conversation_id', 'tenant_id', 'user_id')
    def validate_required_ids(cls, v):
        if not v or not v.strip():
            raise ValueError("ID fields cannot be empty")
        return v.strip()

    @validator('sequence_number')
    def validate_sequence_number(cls, v):
        if v < 1:
            raise ValueError("Sequence number must be positive")
        return v

    def is_inbound(self) -> bool:
        """Check if message is from user (inbound)"""
        return self.direction == "inbound"

    def is_outbound(self) -> bool:
        """Check if message is from bot (outbound)"""
        return self.direction == "outbound"

    def has_media(self) -> bool:
        """Check if message contains media content"""
        return self.content.media is not None

    def has_location(self) -> bool:
        """Check if message contains location data"""
        return self.content.location is not None

    def mark_as_edited(self, new_content: MessageContent, editor_id: Optional[str] = None) -> None:
        """Mark message as edited and store edit history"""
        # Store current content in edit history
        edit_entry = {
            "timestamp": datetime.utcnow().isoformat(),
            "editor_id": editor_id,
            "previous_content": self.content.dict()
        }
        self.edit_history.append(edit_entry)

        # Update content and mark as edited
        self.content = new_content
        self.edited = True
        self.edited_at = datetime.utcnow()
        self.update_timestamp()

    def add_processing_stage(self, stage: ProcessingStage) -> ProcessingStageInfo:
        """Add a processing stage to the message"""
        return self.processing.add_stage(stage)

    def complete_processing_stage(self, status: str = "success", error_details: Optional[str] = None) -> None:
        """Complete the current processing stage"""
        current_stage = self.processing.get_current_stage()
        if current_stage:
            current_stage.complete_stage(status, error_details)
            self.processing.calculate_total_time()

    def set_ai_intent(self, intent: str, confidence: float, alternatives: List[Dict[str, Any]] = None) -> None:
        """Set the AI-detected intent"""
        self.ai_analysis.set_intent(intent, confidence, alternatives)

    def add_ai_entity(self, entity_type: str, value: str, start_pos: int, end_pos: int, confidence: float) -> None:
        """Add an AI-extracted entity"""
        self.ai_analysis.add_entity(entity_type, value, start_pos, end_pos, confidence)

    def set_sentiment(self, label: str, score: float, confidence: float) -> None:
        """Set sentiment analysis results"""
        from src.models.types import SentimentLabel
        self.ai_analysis.sentiment = SentimentResult(
            label=SentimentLabel(label),
            score=score,
            confidence=confidence
        )

    def flag_for_moderation(self, flags: List[str], reason: Optional[str] = None) -> None:
        """Flag message for moderation"""
        self.moderation.flagged = True
        self.moderation.flags.extend(flags)
        if reason:
            self.moderation.escalation_reason = reason
        self.update_timestamp()

    def mark_pii_detected(self, pii_types: List[str], masked_content: Optional[str] = None) -> None:
        """Mark message as containing PII"""
        self.privacy.contains_pii = True
        self.privacy.pii_types.extend(pii_types)
        if masked_content:
            self.privacy.masked_content = masked_content
        self.update_timestamp()

    def set_generation_metadata(
            self,
            model_provider: str,
            model_name: str,
            cost_cents: float,
            generation_time_ms: int,
            input_tokens: int,
            output_tokens: int
    ) -> None:
        """Set generation metadata for outbound messages"""
        if not self.generation_metadata:
            self.generation_metadata = GenerationMetadata()

        self.generation_metadata.model_provider = model_provider
        self.generation_metadata.model_name = model_name
        self.generation_metadata.cost_cents = cost_cents
        self.generation_metadata.generation_time_ms = generation_time_ms
        self.generation_metadata.set_token_usage(input_tokens, output_tokens)

    def get_display_text(self) -> str:
        """Get text content for display (handles PII masking)"""
        if self.privacy.contains_pii and self.privacy.masked_content:
            return self.privacy.masked_content
        return self.content.text or ""

    def is_delivery_confirmed(self) -> bool:
        """Check if message delivery is confirmed"""
        if not self.channel_metadata:
            return False
        return self.channel_metadata.delivery_status in [DeliveryStatus.DELIVERED, DeliveryStatus.READ]

    def calculate_age_hours(self) -> float:
        """Calculate message age in hours"""
        delta = datetime.utcnow() - self.timestamp
        return delta.total_seconds() / 3600

    @classmethod
    def create_inbound(
            cls,
            message_id: MessageId,
            conversation_id: ConversationId,
            tenant_id: TenantId,
            user_id: UserId,
            sequence_number: int,
            channel: str,
            content: MessageContent,
            **kwargs
    ) -> "MessageDocument":
        """
        Create a new inbound message document.

        Args:
            message_id: Unique message identifier
            conversation_id: Parent conversation identifier
            tenant_id: Tenant identifier
            user_id: User identifier
            sequence_number: Message sequence number in conversation
            channel: Communication channel
            content: Message content
            **kwargs: Additional fields

        Returns:
            New MessageDocument instance for inbound message
        """
        return cls(
            message_id=message_id,
            conversation_id=conversation_id,
            tenant_id=tenant_id,
            user_id=user_id,
            sequence_number=sequence_number,
            direction="inbound",
            channel=channel,
            content=content,
            **kwargs
        )

    @classmethod
    def create_outbound(
            cls,
            message_id: MessageId,
            conversation_id: ConversationId,
            tenant_id: TenantId,
            user_id: UserId,
            sequence_number: int,
            channel: str,
            content: MessageContent,
            generation_metadata: Optional[GenerationMetadata] = None,
            **kwargs
    ) -> "MessageDocument":
        """
        Create a new outbound message document.

        Args:
            message_id: Unique message identifier
            conversation_id: Parent conversation identifier
            tenant_id: Tenant identifier
            user_id: User identifier
            sequence_number: Message sequence number in conversation
            channel: Communication channel
            content: Message content
            generation_metadata: Optional generation metadata
            **kwargs: Additional fields

        Returns:
            New MessageDocument instance for outbound message
        """
        return cls(
            message_id=message_id,
            conversation_id=conversation_id,
            tenant_id=tenant_id,
            user_id=user_id,
            sequence_number=sequence_number,
            direction="outbound",
            channel=channel,
            content=content,
            generation_metadata=generation_metadata,
            **kwargs
        )

    def to_summary_dict(self) -> Dict[str, Any]:
        """
        Convert to summary dictionary for API responses.

        Returns:
            Summary dictionary with key message information
        """
        return {
            "message_id": self.message_id,
            "conversation_id": self.conversation_id,
            "direction": self.direction,
            "timestamp": self.timestamp,
            "sequence_number": self.sequence_number,
            "content_type": self.content.type,
            "text": self.get_display_text()[:200] + "..." if len(
                self.get_display_text()) > 200 else self.get_display_text(),
            "has_media": self.has_media(),
            "has_location": self.has_location(),
            "language": self.language,
            "intent": self.ai_analysis.intent.detected_intent if self.ai_analysis.intent else None,
            "sentiment": self.ai_analysis.sentiment.label if self.ai_analysis.sentiment else None,
            "flagged": self.moderation.flagged,
            "contains_pii": self.privacy.contains_pii
        }