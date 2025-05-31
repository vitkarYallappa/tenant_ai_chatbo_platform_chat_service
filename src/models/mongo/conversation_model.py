# src/models/mongo/conversation_model.py
"""
MongoDB document model for conversation storage.
Represents conversation documents in MongoDB with all metadata, context, and metrics.
"""

from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional
from pydantic import BaseModel, Field, validator
from bson import ObjectId

from src.models.base_model import BaseMongoModel, TimestampMixin, AuditMixin
from src.models.types import (
    ConversationStatus, ChannelType, TenantId, UserId,
    ConversationId, SessionId, ChannelMetadata, UserInfo,
    BusinessContext, Priority, IntentType
)


class StateTransition(BaseModel):
    """Individual state transition in conversation flow"""
    state: str = Field(..., max_length=100)
    entered_at: datetime = Field(default_factory=datetime.utcnow)
    exited_at: Optional[datetime] = None
    duration_ms: Optional[int] = None

    def complete_transition(self) -> None:
        """Mark state transition as complete and calculate duration"""
        self.exited_at = datetime.utcnow()
        if self.entered_at:
            delta = self.exited_at - self.entered_at
            self.duration_ms = int(delta.total_seconds() * 1000)


class ConversationMetrics(BaseModel):
    """Conversation performance and quality metrics"""
    message_count: int = Field(default=0, ge=0)
    user_messages: int = Field(default=0, ge=0)
    bot_messages: int = Field(default=0, ge=0)

    # Response time metrics (in milliseconds)
    response_time_avg_ms: Optional[int] = Field(None, ge=0)
    response_time_max_ms: Optional[int] = Field(None, ge=0)
    response_time_min_ms: Optional[int] = Field(None, ge=0)

    # Conversation flow metrics
    intent_switches: int = Field(default=0, ge=0)
    escalation_triggers: int = Field(default=0, ge=0)

    # Quality metrics
    user_satisfaction: Optional[float] = Field(None, ge=1, le=5)
    user_feedback: Optional[str] = Field(None, max_length=1000)
    completion_rate: Optional[float] = Field(None, ge=0, le=1)
    goal_achieved: bool = Field(default=False)

    # Cost and resource metrics
    total_cost_cents: Optional[float] = Field(None, ge=0)
    total_tokens: Optional[int] = Field(None, ge=0)

    def add_message(self, is_user_message: bool, response_time_ms: Optional[int] = None) -> None:
        """Add a message to metrics and update counters"""
        self.message_count += 1

        if is_user_message:
            self.user_messages += 1
        else:
            self.bot_messages += 1

            # Update response time metrics for bot messages
            if response_time_ms is not None:
                if self.response_time_avg_ms is None:
                    self.response_time_avg_ms = response_time_ms
                    self.response_time_max_ms = response_time_ms
                    self.response_time_min_ms = response_time_ms
                else:
                    # Calculate running average
                    total_responses = self.bot_messages
                    total_time = self.response_time_avg_ms * (total_responses - 1) + response_time_ms
                    self.response_time_avg_ms = int(total_time / total_responses)

                    # Update min/max
                    self.response_time_max_ms = max(self.response_time_max_ms or 0, response_time_ms)
                    self.response_time_min_ms = min(self.response_time_min_ms or float('inf'), response_time_ms)


class ConversationContext(BaseModel):
    """Conversation context and state information"""
    # Intent tracking
    intent_history: List[str] = Field(default_factory=list, max_items=50)
    current_intent: Optional[str] = Field(None, max_length=100)
    intent_confidence: Optional[float] = Field(None, ge=0, le=1)

    # Entity and slot management
    entities: Dict[str, Any] = Field(default_factory=dict)
    slots: Dict[str, Any] = Field(default_factory=dict)

    # User and session data
    user_profile: Dict[str, Any] = Field(default_factory=dict)
    session_variables: Dict[str, Any] = Field(default_factory=dict)

    # Conversation metadata
    custom_attributes: Dict[str, Any] = Field(default_factory=dict)
    conversation_tags: List[str] = Field(default_factory=list, max_items=20)

    # Flow state
    conversation_stage: Optional[str] = Field(None, max_length=100)
    next_expected_inputs: List[str] = Field(default_factory=list, max_items=10)

    def add_intent(self, intent: str, confidence: Optional[float] = None) -> None:
        """Add new intent to history and update current intent"""
        if intent != self.current_intent:
            if self.current_intent:
                self.intent_history.append(self.current_intent)

            # Keep history limited to last 50 intents
            if len(self.intent_history) > 50:
                self.intent_history = self.intent_history[-50:]

        self.current_intent = intent
        self.intent_confidence = confidence

    def set_slot(self, slot_name: str, value: Any) -> None:
        """Set a slot value"""
        self.slots[slot_name] = value

    def get_slot(self, slot_name: str, default: Any = None) -> Any:
        """Get a slot value"""
        return self.slots.get(slot_name, default)

    def clear_slot(self, slot_name: str) -> None:
        """Clear a slot value"""
        self.slots.pop(slot_name, None)


class AIMetadata(BaseModel):
    """AI processing and model usage metadata"""
    primary_models_used: List[str] = Field(default_factory=list)
    fallback_models_used: List[str] = Field(default_factory=list)
    total_cost_cents: Optional[float] = Field(None, ge=0)
    total_tokens: Optional[int] = Field(None, ge=0)
    average_confidence: Optional[float] = Field(None, ge=0, le=1)

    quality_scores: Dict[str, float] = Field(default_factory=dict)

    def add_model_usage(self, model_name: str, cost_cents: float, tokens: int, is_fallback: bool = False) -> None:
        """Record model usage"""
        if is_fallback:
            if model_name not in self.fallback_models_used:
                self.fallback_models_used.append(model_name)
        else:
            if model_name not in self.primary_models_used:
                self.primary_models_used.append(model_name)

        # Update totals
        self.total_cost_cents = (self.total_cost_cents or 0) + cost_cents
        self.total_tokens = (self.total_tokens or 0) + tokens


class ComplianceInfo(BaseModel):
    """Compliance and privacy information"""
    pii_detected: bool = Field(default=False)
    pii_masked: bool = Field(default=False)
    pii_types: List[str] = Field(default_factory=list)

    # Data retention
    data_retention_until: Optional[datetime] = None
    anonymization_level: str = Field(default="none")  # none, partial, full

    # Regulatory compliance
    gdpr_flags: List[str] = Field(default_factory=list)
    audit_required: bool = Field(default=False)
    consent_collected: bool = Field(default=False)
    consent_details: Dict[str, Any] = Field(default_factory=dict)

    def set_retention_period(self, days: int) -> None:
        """Set data retention period"""
        self.data_retention_until = datetime.utcnow() + timedelta(days=days)

    def is_retention_expired(self) -> bool:
        """Check if retention period has expired"""
        if not self.data_retention_until:
            return False
        return datetime.utcnow() > self.data_retention_until


class ConversationSummary(BaseModel):
    """Conversation summary and analysis"""
    auto_generated_summary: Optional[str] = Field(None, max_length=2000)
    key_topics: List[str] = Field(default_factory=list, max_items=10)
    entities_mentioned: List[str] = Field(default_factory=list, max_items=20)
    action_items: List[str] = Field(default_factory=list, max_items=10)

    follow_up_required: bool = Field(default=False)
    follow_up_date: Optional[datetime] = None
    escalation_reason: Optional[str] = Field(None, max_length=500)
    human_notes: Optional[str] = Field(None, max_length=2000)


class ConversationDocument(BaseMongoModel, TimestampMixin, AuditMixin):
    """
    MongoDB document structure for conversations.
    Comprehensive conversation tracking with metadata, metrics, and context.
    """

    # Core identifiers
    conversation_id: ConversationId = Field(..., max_length=100)
    tenant_id: TenantId = Field(..., max_length=100)
    user_id: UserId = Field(..., max_length=255)
    session_id: Optional[SessionId] = Field(None, max_length=100)

    # Channel and metadata
    channel: ChannelType
    channel_metadata: Optional[ChannelMetadata] = Field(default_factory=ChannelMetadata)

    # Lifecycle tracking
    status: ConversationStatus = Field(default=ConversationStatus.ACTIVE)
    started_at: datetime = Field(default_factory=datetime.utcnow)
    last_activity_at: datetime = Field(default_factory=datetime.utcnow)
    completed_at: Optional[datetime] = None
    duration_seconds: Optional[int] = None

    # Flow and state management
    flow_id: Optional[str] = Field(None, max_length=100)
    flow_version: Optional[str] = Field(None, max_length=50)
    current_state: Optional[str] = Field(None, max_length=100)
    previous_states: List[str] = Field(default_factory=list, max_items=100)
    state_history: List[StateTransition] = Field(default_factory=list, max_items=100)

    # Context and data
    context: ConversationContext = Field(default_factory=ConversationContext)
    metrics: ConversationMetrics = Field(default_factory=ConversationMetrics)

    # User information (privacy compliant)
    user_info: Optional[UserInfo] = None

    # Business context
    business_context: BusinessContext = Field(default_factory=BusinessContext)

    # AI and processing metadata
    ai_metadata: AIMetadata = Field(default_factory=AIMetadata)

    # Compliance and privacy
    compliance: ComplianceInfo = Field(default_factory=ComplianceInfo)

    # Analysis and summary
    summary: ConversationSummary = Field(default_factory=ConversationSummary)

    # A/B testing information
    ab_testing: Dict[str, Any] = Field(default_factory=dict)

    class Config:
        collection_name = "conversations"

    @validator('conversation_id', 'tenant_id', 'user_id')
    def validate_required_ids(cls, v):
        if not v or not v.strip():
            raise ValueError("ID fields cannot be empty")
        return v.strip()

    def update_last_activity(self) -> None:
        """Update last activity timestamp"""
        self.last_activity_at = datetime.utcnow()
        self.update_timestamp()

    def complete_conversation(self, status: ConversationStatus = ConversationStatus.COMPLETED) -> None:
        """Mark conversation as completed and calculate duration"""
        self.status = status
        self.completed_at = datetime.utcnow()

        if self.started_at:
            delta = self.completed_at - self.started_at
            self.duration_seconds = int(delta.total_seconds())

        self.update_timestamp()

    def transition_to_state(self, new_state: str) -> None:
        """Transition to a new conversation state"""
        # Complete current state transition if exists
        if self.state_history and not self.state_history[-1].exited_at:
            self.state_history[-1].complete_transition()

        # Add previous state to history
        if self.current_state:
            self.previous_states.append(self.current_state)

            # Keep previous states limited
            if len(self.previous_states) > 100:
                self.previous_states = self.previous_states[-100:]

        # Set new state
        self.current_state = new_state

        # Add to state history
        transition = StateTransition(state=new_state)
        self.state_history.append(transition)

        # Keep state history limited
        if len(self.state_history) > 100:
            self.state_history = self.state_history[-100:]

        self.update_last_activity()

    def add_message(self, is_user_message: bool, response_time_ms: Optional[int] = None) -> None:
        """Add message to conversation and update metrics"""
        self.metrics.add_message(is_user_message, response_time_ms)
        self.update_last_activity()

    def set_user_satisfaction(self, score: float, feedback: Optional[str] = None) -> None:
        """Set user satisfaction score and feedback"""
        if not 1 <= score <= 5:
            raise ValueError("Satisfaction score must be between 1 and 5")

        self.metrics.user_satisfaction = score
        if feedback:
            self.metrics.user_feedback = feedback
        self.update_timestamp()

    def add_business_tag(self, tag: str) -> None:
        """Add a business tag to the conversation"""
        if tag not in self.business_context.tags:
            self.business_context.tags.append(tag)
            self.update_timestamp()

    def set_priority(self, priority: Priority) -> None:
        """Set conversation priority"""
        self.business_context.priority = priority
        self.update_timestamp()

    def is_active(self) -> bool:
        """Check if conversation is still active"""
        return self.status == ConversationStatus.ACTIVE

    def is_stale(self, hours: int = 24) -> bool:
        """Check if conversation is stale (no activity for specified hours)"""
        if not self.last_activity_at:
            return True

        threshold = datetime.utcnow() - timedelta(hours=hours)
        return self.last_activity_at < threshold

    def should_be_cleaned_up(self) -> bool:
        """Check if conversation should be cleaned up based on compliance rules"""
        if self.compliance.is_retention_expired():
            return True

        # Additional cleanup rules can be added here
        return False

    @classmethod
    def create_new(
            cls,
            conversation_id: ConversationId,
            tenant_id: TenantId,
            user_id: UserId,
            channel: ChannelType,
            session_id: Optional[SessionId] = None,
            flow_id: Optional[str] = None,
            **kwargs
    ) -> "ConversationDocument":
        """
        Create a new conversation document with required fields.

        Args:
            conversation_id: Unique conversation identifier
            tenant_id: Tenant identifier
            user_id: User identifier
            channel: Communication channel
            session_id: Optional session identifier
            flow_id: Optional conversation flow identifier
            **kwargs: Additional fields

        Returns:
            New ConversationDocument instance
        """
        return cls(
            conversation_id=conversation_id,
            tenant_id=tenant_id,
            user_id=user_id,
            channel=channel,
            session_id=session_id,
            flow_id=flow_id,
            **kwargs
        )

    def to_summary_dict(self) -> Dict[str, Any]:
        """
        Convert to summary dictionary for API responses.

        Returns:
            Summary dictionary with key conversation information
        """
        return {
            "conversation_id": self.conversation_id,
            "user_id": self.user_id,
            "channel": self.channel,
            "status": self.status,
            "started_at": self.started_at,
            "last_activity_at": self.last_activity_at,
            "completed_at": self.completed_at,
            "duration_seconds": self.duration_seconds,
            "message_count": self.metrics.message_count,
            "user_satisfaction": self.metrics.user_satisfaction,
            "current_intent": self.context.current_intent,
            "current_state": self.current_state,
            "priority": self.business_context.priority,
            "tags": self.business_context.tags
        }