"""
MongoDB Conversation Model
=========================

MongoDB document models for conversation data with comprehensive
conversation lifecycle management and business analytics support.

Features:
- Complete conversation document structure
- Business context and analytics
- Compliance and privacy handling
- A/B testing support
- Serialization utilities
"""

from datetime import datetime, timedelta
from typing import Optional, Dict, Any, List
from dataclasses import dataclass, asdict, field
from enum import Enum
import structlog

from ..types import (
    TenantId, UserId, ConversationId, SessionId, FlowId,
    ConversationStatus, ChannelType, Priority, LanguageCode
)

logger = structlog.get_logger(__name__)


@dataclass
class ChannelMetadata:
    """Channel-specific metadata for conversations"""
    platform_user_id: Optional[str] = None
    platform_channel_id: Optional[str] = None
    thread_id: Optional[str] = None
    bot_id: Optional[str] = None
    workspace_id: Optional[str] = None
    platform_message_id: Optional[str] = None
    webhook_url: Optional[str] = None
    additional_data: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ChannelMetadata":
        """Create from dictionary"""
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


@dataclass
class ConversationContext:
    """Conversation context and state information"""
    intent_history: List[str] = field(default_factory=list)
    current_intent: Optional[str] = None
    intent_confidence: Optional[float] = None
    entities: Dict[str, Any] = field(default_factory=dict)
    slots: Dict[str, Any] = field(default_factory=dict)
    user_profile: Dict[str, Any] = field(default_factory=dict)
    session_variables: Dict[str, Any] = field(default_factory=dict)
    custom_attributes: Dict[str, Any] = field(default_factory=dict)
    conversation_tags: List[str] = field(default_factory=list)
    conversation_stage: Optional[str] = None
    next_expected_input: Optional[str] = None

    def add_intent(self, intent: str, confidence: Optional[float] = None) -> None:
        """Add intent to history"""
        self.intent_history.append(intent)
        self.current_intent = intent
        if confidence is not None:
            self.intent_confidence = confidence
        # Keep only last 20 intents
        self.intent_history = self.intent_history[-20:]

    def set_slot(self, key: str, value: Any) -> None:
        """Set slot value"""
        self.slots[key] = value

    def get_slot(self, key: str, default: Any = None) -> Any:
        """Get slot value"""
        return self.slots.get(key, default)

    def set_entity(self, entity_type: str, entity_value: Any) -> None:
        """Set entity value"""
        self.entities[entity_type] = entity_value

    def get_entity(self, entity_type: str, default: Any = None) -> Any:
        """Get entity value"""
        return self.entities.get(entity_type, default)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ConversationContext":
        """Create from dictionary"""
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


@dataclass
class UserInfo:
    """User information for conversations (privacy compliant)"""
    first_seen: Optional[datetime] = None
    return_visitor: bool = False
    language: LanguageCode = LanguageCode.EN
    timezone: Optional[str] = None
    device_info: Dict[str, Any] = field(default_factory=dict)
    location: Dict[str, Any] = field(default_factory=dict)
    preferences: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary with datetime serialization"""
        data = asdict(self)
        if data['first_seen']:
            data['first_seen'] = data['first_seen'].isoformat()
        if 'language' in data and hasattr(data['language'], 'value'):
            data['language'] = data['language'].value
        return data

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "UserInfo":
        """Create from dictionary with datetime deserialization"""
        if 'first_seen' in data and data['first_seen']:
            if isinstance(data['first_seen'], str):
                data['first_seen'] = datetime.fromisoformat(data['first_seen'])

        if 'language' in data and isinstance(data['language'], str):
            try:
                data['language'] = LanguageCode(data['language'])
            except ValueError:
                data['language'] = LanguageCode.EN

        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


@dataclass
class ConversationMetrics:
    """Conversation performance and quality metrics"""
    message_count: int = 0
    user_messages: int = 0
    bot_messages: int = 0
    response_time_avg_ms: Optional[float] = None
    response_time_max_ms: Optional[float] = None
    intent_switches: int = 0
    escalation_triggers: int = 0
    user_satisfaction: Optional[Dict[str, Any]] = None
    completion_rate: Optional[float] = None
    goal_achieved: Optional[bool] = None
    resolution_time_seconds: Optional[int] = None

    def increment_user_message(self) -> None:
        """Increment user message count"""
        self.user_messages += 1
        self.message_count += 1

    def increment_bot_message(self) -> None:
        """Increment bot message count"""
        self.bot_messages += 1
        self.message_count += 1

    def update_response_time(self, response_time_ms: float) -> None:
        """Update response time metrics"""
        if self.response_time_avg_ms is None:
            self.response_time_avg_ms = response_time_ms
        else:
            # Simple moving average
            total_responses = self.bot_messages
            if total_responses > 0:
                self.response_time_avg_ms = (
                        (self.response_time_avg_ms * (total_responses - 1) + response_time_ms) / total_responses
                )

        if self.response_time_max_ms is None or response_time_ms > self.response_time_max_ms:
            self.response_time_max_ms = response_time_ms

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ConversationMetrics":
        """Create from dictionary"""
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


@dataclass
class AIMetadata:
    """AI and model metadata for conversations"""
    primary_models_used: List[str] = field(default_factory=list)
    fallback_models_used: List[str] = field(default_factory=list)
    total_cost_cents: float = 0.0
    total_tokens: int = 0
    average_confidence: Optional[float] = None
    quality_scores: Dict[str, float] = field(default_factory=dict)
    model_performance: Dict[str, Any] = field(default_factory=dict)

    def add_model_usage(self, model_name: str, cost_cents: float, tokens: int, is_fallback: bool = False) -> None:
        """Add model usage information"""
        if is_fallback:
            if model_name not in self.fallback_models_used:
                self.fallback_models_used.append(model_name)
        else:
            if model_name not in self.primary_models_used:
                self.primary_models_used.append(model_name)

        self.total_cost_cents += cost_cents
        self.total_tokens += tokens

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "AIMetadata":
        """Create from dictionary"""
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


@dataclass
class BusinessContext:
    """Business context and categorization"""
    department: Optional[str] = None  # sales, support, marketing, general
    category: Optional[str] = None
    subcategory: Optional[str] = None
    priority: Priority = Priority.NORMAL
    tags: List[str] = field(default_factory=list)
    resolution_type: Optional[str] = None  # automated, escalated, abandoned, timeout
    outcome: Optional[str] = None  # resolved, unresolved, pending, escalated
    value_generated: Optional[float] = None
    cost_incurred: Optional[float] = None
    customer_effort_score: Optional[int] = None
    business_impact: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        data = asdict(self)
        if 'priority' in data and hasattr(data['priority'], 'value'):
            data['priority'] = data['priority'].value
        return data

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "BusinessContext":
        """Create from dictionary"""
        if 'priority' in data and isinstance(data['priority'], (str, int)):
            try:
                data['priority'] = Priority(data['priority'])
            except ValueError:
                data['priority'] = Priority.NORMAL

        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


@dataclass
class ComplianceData:
    """Compliance and privacy information"""
    pii_detected: bool = False
    pii_masked: bool = False
    pii_types: List[str] = field(default_factory=list)
    data_retention_until: Optional[datetime] = None
    anonymization_level: str = "none"  # none, partial, full
    gdpr_flags: List[str] = field(default_factory=list)
    audit_required: bool = False
    consent_collected: bool = False
    consent_details: Dict[str, Any] = field(default_factory=dict)
    deletion_scheduled: bool = False

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary with datetime serialization"""
        data = asdict(self)
        if data['data_retention_until']:
            data['data_retention_until'] = data['data_retention_until'].isoformat()
        return data

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ComplianceData":
        """Create from dictionary with datetime deserialization"""
        if 'data_retention_until' in data and data['data_retention_until']:
            if isinstance(data['data_retention_until'], str):
                data['data_retention_until'] = datetime.fromisoformat(data['data_retention_until'])

        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


@dataclass
class IntegrationUsage:
    """Integration usage tracking"""
    integration_id: str
    integration_name: str
    calls_made: int = 0
    success_rate: float = 1.0
    total_cost_cents: float = 0.0
    average_response_time_ms: Optional[float] = None
    last_used: Optional[datetime] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary with datetime serialization"""
        data = asdict(self)
        if data['last_used']:
            data['last_used'] = data['last_used'].isoformat()
        return data

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "IntegrationUsage":
        """Create from dictionary with datetime deserialization"""
        if 'last_used' in data and data['last_used']:
            if isinstance(data['last_used'], str):
                data['last_used'] = datetime.fromisoformat(data['last_used'])

        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


@dataclass
class ConversationSummary:
    """Conversation summary and analysis"""
    auto_generated_summary: Optional[str] = None
    key_topics: List[str] = field(default_factory=list)
    entities_mentioned: List[str] = field(default_factory=list)
    action_items: List[str] = field(default_factory=list)
    follow_up_required: bool = False
    follow_up_date: Optional[datetime] = None
    escalation_reason: Optional[str] = None
    human_notes: Optional[str] = None
    resolution_summary: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary with datetime serialization"""
        data = asdict(self)
        if data['follow_up_date']:
            data['follow_up_date'] = data['follow_up_date'].isoformat()
        return data

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ConversationSummary":
        """Create from dictionary with datetime deserialization"""
        if 'follow_up_date' in data and data['follow_up_date']:
            if isinstance(data['follow_up_date'], str):
                data['follow_up_date'] = datetime.fromisoformat(data['follow_up_date'])

        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


@dataclass
class ABTestingInfo:
    """A/B testing information"""
    experiment_id: Optional[str] = None
    variant: Optional[str] = None
    control_group: bool = False
    experiment_start: Optional[datetime] = None
    experiment_end: Optional[datetime] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary with datetime serialization"""
        data = asdict(self)
        for field in ['experiment_start', 'experiment_end']:
            if data[field]:
                data[field] = data[field].isoformat()
        return data

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ABTestingInfo":
        """Create from dictionary with datetime deserialization"""
        for field in ['experiment_start', 'experiment_end']:
            if field in data and data[field]:
                if isinstance(data[field], str):
                    data[field] = datetime.fromisoformat(data[field])

        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


@dataclass
class StateHistory:
    """State transition history"""
    state: str
    entered_at: datetime
    exited_at: Optional[datetime] = None
    duration_ms: Optional[int] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def complete_state(self, exit_time: Optional[datetime] = None) -> None:
        """Mark state as completed"""
        self.exited_at = exit_time or datetime.utcnow()
        if self.exited_at and self.entered_at:
            delta = self.exited_at - self.entered_at
            self.duration_ms = int(delta.total_seconds() * 1000)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary with datetime serialization"""
        data = asdict(self)
        data['entered_at'] = data['entered_at'].isoformat()
        if data['exited_at']:
            data['exited_at'] = data['exited_at'].isoformat()
        return data

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "StateHistory":
        """Create from dictionary with datetime deserialization"""
        data['entered_at'] = datetime.fromisoformat(data['entered_at'])
        if data.get('exited_at'):
            data['exited_at'] = datetime.fromisoformat(data['exited_at'])

        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


@dataclass
class ConversationDocument:
    """
    Complete MongoDB conversation document model

    This class represents the full conversation document structure
    matching the MongoDB schema specification.
    """
    # Core identifiers
    conversation_id: ConversationId
    tenant_id: TenantId
    user_id: UserId
    session_id: Optional[SessionId] = None

    # Channel and context
    channel: ChannelType = ChannelType.WEB
    channel_metadata: ChannelMetadata = field(default_factory=ChannelMetadata)

    # Conversation lifecycle
    status: ConversationStatus = ConversationStatus.ACTIVE
    started_at: datetime = field(default_factory=datetime.utcnow)
    last_activity_at: datetime = field(default_factory=datetime.utcnow)
    completed_at: Optional[datetime] = None
    duration_seconds: Optional[int] = None

    # Flow and state management
    flow_id: Optional[FlowId] = None
    flow_version: Optional[str] = None
    current_state: Optional[str] = None
    previous_states: List[str] = field(default_factory=list)
    state_history: List[StateHistory] = field(default_factory=list)

    # Conversation context
    context: ConversationContext = field(default_factory=ConversationContext)

    # User information (privacy compliant)
    user_info: UserInfo = field(default_factory=UserInfo)

    # Conversation quality and metrics
    metrics: ConversationMetrics = field(default_factory=ConversationMetrics)

    # AI and model metadata
    ai_metadata: AIMetadata = field(default_factory=AIMetadata)

    # Business and operational context
    business_context: BusinessContext = field(default_factory=BusinessContext)

    # Compliance and privacy
    compliance: ComplianceData = field(default_factory=ComplianceData)

    # Integration and external system data
    integrations_used: List[IntegrationUsage] = field(default_factory=list)

    # Summary and analysis
    summary: ConversationSummary = field(default_factory=ConversationSummary)

    # A/B testing information
    ab_testing: ABTestingInfo = field(default_factory=ABTestingInfo)

    # MongoDB ObjectId (set by database)
    id: Optional[str] = None

    # Timestamps (managed by database)
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None

    def update_activity(self) -> None:
        """Update last activity timestamp"""
        self.last_activity_at = datetime.utcnow()

    def complete_conversation(self, outcome: Optional[str] = None) -> None:
        """Mark conversation as completed"""
        self.status = ConversationStatus.COMPLETED
        self.completed_at = datetime.utcnow()

        if self.started_at and self.completed_at:
            delta = self.completed_at - self.started_at
            self.duration_seconds = int(delta.total_seconds())

        if outcome:
            self.business_context.outcome = outcome

        # Complete current state if any
        if self.state_history and self.state_history[-1].exited_at is None:
            self.state_history[-1].complete_state(self.completed_at)

    def transition_to_state(self, new_state: str, metadata: Optional[Dict[str, Any]] = None) -> None:
        """Transition to a new state"""
        # Complete previous state
        if self.state_history and self.state_history[-1].exited_at is None:
            self.state_history[-1].complete_state()

        # Add previous state to history
        if self.current_state:
            self.previous_states.append(self.current_state)
            # Keep only last 50 states
            self.previous_states = self.previous_states[-50:]

        # Set new state
        self.current_state = new_state

        # Add to state history
        state_entry = StateHistory(
            state=new_state,
            entered_at=datetime.utcnow(),
            metadata=metadata or {}
        )
        self.state_history.append(state_entry)

        self.update_activity()

    def add_integration_usage(self, integration_usage: IntegrationUsage) -> None:
        """Add integration usage information"""
        # Check if integration already exists
        for i, existing in enumerate(self.integrations_used):
            if existing.integration_id == integration_usage.integration_id:
                self.integrations_used[i] = integration_usage
                return

        # Add new integration usage
        self.integrations_used.append(integration_usage)

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert to dictionary for MongoDB storage

        Returns:
            Dictionary representation suitable for MongoDB
        """
        data = {}

        # Core fields
        data['conversation_id'] = self.conversation_id
        data['tenant_id'] = self.tenant_id
        data['user_id'] = self.user_id
        data['session_id'] = self.session_id

        # Enum fields
        data['channel'] = self.channel.value if hasattr(self.channel, 'value') else self.channel
        data['status'] = self.status.value if hasattr(self.status, 'value') else self.status

        # Timestamps
        data['started_at'] = self.started_at
        data['last_activity_at'] = self.last_activity_at
        data['completed_at'] = self.completed_at
        data['duration_seconds'] = self.duration_seconds

        # Flow and state
        data['flow_id'] = self.flow_id
        data['flow_version'] = self.flow_version
        data['current_state'] = self.current_state
        data['previous_states'] = self.previous_states
        data['state_history'] = [state.to_dict() for state in self.state_history]

        # Complex objects
        data['channel_metadata'] = self.channel_metadata.to_dict()
        data['context'] = self.context.to_dict()
        data['user_info'] = self.user_info.to_dict()
        data['metrics'] = self.metrics.to_dict()
        data['ai_metadata'] = self.ai_metadata.to_dict()
        data['business_context'] = self.business_context.to_dict()
        data['compliance'] = self.compliance.to_dict()
        data['integrations_used'] = [integration.to_dict() for integration in self.integrations_used]
        data['summary'] = self.summary.to_dict()
        data['ab_testing'] = self.ab_testing.to_dict()

        # MongoDB fields
        if self.id:
            data['_id'] = self.id
        data['created_at'] = self.created_at
        data['updated_at'] = self.updated_at

        return data

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ConversationDocument":
        """
        Create from MongoDB document

        Args:
            data: MongoDB document data

        Returns:
            ConversationDocument instance
        """
        # Handle MongoDB ObjectId
        if '_id' in data:
            data['id'] = str(data['_id'])
            data.pop('_id', None)

        # Convert enum strings back to enums
        if 'channel' in data and isinstance(data['channel'], str):
            try:
                data['channel'] = ChannelType(data['channel'])
            except ValueError:
                data['channel'] = ChannelType.WEB

        if 'status' in data and isinstance(data['status'], str):
            try:
                data['status'] = ConversationStatus(data['status'])
            except ValueError:
                data['status'] = ConversationStatus.ACTIVE

        # Convert complex objects
        if 'channel_metadata' in data and isinstance(data['channel_metadata'], dict):
            data['channel_metadata'] = ChannelMetadata.from_dict(data['channel_metadata'])

        if 'context' in data and isinstance(data['context'], dict):
            data['context'] = ConversationContext.from_dict(data['context'])

        if 'user_info' in data and isinstance(data['user_info'], dict):
            data['user_info'] = UserInfo.from_dict(data['user_info'])

        if 'metrics' in data and isinstance(data['metrics'], dict):
            data['metrics'] = ConversationMetrics.from_dict(data['metrics'])

        if 'ai_metadata' in data and isinstance(data['ai_metadata'], dict):
            data['ai_metadata'] = AIMetadata.from_dict(data['ai_metadata'])

        if 'business_context' in data and isinstance(data['business_context'], dict):
            data['business_context'] = BusinessContext.from_dict(data['business_context'])

        if 'compliance' in data and isinstance(data['compliance'], dict):
            data['compliance'] = ComplianceData.from_dict(data['compliance'])

        if 'summary' in data and isinstance(data['summary'], dict):
            data['summary'] = ConversationSummary.from_dict(data['summary'])

        if 'ab_testing' in data and isinstance(data['ab_testing'], dict):
            data['ab_testing'] = ABTestingInfo.from_dict(data['ab_testing'])

        # Convert lists of complex objects
        if 'state_history' in data and isinstance(data['state_history'], list):
            data['state_history'] = [
                StateHistory.from_dict(state_data)
                for state_data in data['state_history']
            ]

        if 'integrations_used' in data and isinstance(data['integrations_used'], list):
            data['integrations_used'] = [
                IntegrationUsage.from_dict(integration_data)
                for integration_data in data['integrations_used']
            ]

        # Create instance with only valid fields
        valid_fields = {k: v for k, v in data.items() if k in cls.__dataclass_fields__}
        return cls(**valid_fields)


# Utility functions for conversation management
def create_conversation(
        tenant_id: TenantId,
        user_id: UserId,
        channel: ChannelType = ChannelType.WEB,
        session_id: Optional[SessionId] = None,
        flow_id: Optional[FlowId] = None
) -> ConversationDocument:
    """
    Create a new conversation document

    Args:
        tenant_id: Tenant identifier
        user_id: User identifier
        channel: Communication channel
        session_id: Optional session identifier
        flow_id: Optional flow identifier

    Returns:
        New ConversationDocument instance
    """
    from ..types import generate_conversation_id

    conversation_id = generate_conversation_id()

    conversation = ConversationDocument(
        conversation_id=conversation_id,
        tenant_id=tenant_id,
        user_id=user_id,
        session_id=session_id,
        channel=channel,
        flow_id=flow_id
    )

    return conversation