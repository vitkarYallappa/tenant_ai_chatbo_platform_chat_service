# src/models/redis/session_cache.py
"""
Redis data structures for session management.
Handle active session state and caching with optimized Redis operations.
"""

from datetime import datetime, timedelta
from typing import Dict, Any, Optional, List
from pydantic import BaseModel, Field, validator
import json
import hashlib

from src.models.base_model import BaseRedisModel
from src.models.types import (
    TenantId, UserId, ConversationId, SessionId, ChannelType
)


class SessionCache(BaseRedisModel):
    """
    Redis cache structure for active session data.
    Optimized for fast lookups and updates during active conversations.
    """

    # Core identifiers
    session_id: SessionId
    tenant_id: TenantId
    user_id: UserId
    conversation_id: Optional[ConversationId] = None
    channel: ChannelType

    # Session lifecycle
    last_activity: datetime = Field(default_factory=datetime.utcnow)
    expires_at: datetime = Field(
        default_factory=lambda: datetime.utcnow() + timedelta(hours=1)
    )

    # Quick access context
    context: Dict[str, Any] = Field(default_factory=dict)
    preferences: Dict[str, Any] = Field(default_factory=dict)

    # User information (minimal for performance)
    user_info: Dict[str, Any] = Field(default_factory=dict)

    # Feature flags (for quick feature checks)
    features: Dict[str, bool] = Field(default_factory=dict)

    # Current conversation state
    current_state: Optional[str] = None
    current_intent: Optional[str] = None

    # Activity tracking
    message_count: int = Field(default=0, ge=0)
    idle_start: Optional[datetime] = None

    @staticmethod
    def get_cache_key(tenant_id: TenantId, session_id: SessionId) -> str:
        """Generate Redis cache key for session"""
        return f"session:{tenant_id}:{session_id}"

    @staticmethod
    def get_user_sessions_key(tenant_id: TenantId, user_id: UserId) -> str:
        """Generate Redis key for user's active sessions list"""
        return f"user_sessions:{tenant_id}:{user_id}"

    @staticmethod
    def get_conversation_session_key(tenant_id: TenantId, conversation_id: ConversationId) -> str:
        """Generate Redis key for conversation to session mapping"""
        return f"conv_session:{tenant_id}:{conversation_id}"

    def update_activity(self) -> None:
        """Update last activity timestamp and clear idle state"""
        self.last_activity = datetime.utcnow()
        self.idle_start = None
        self.update_timestamp()

    def mark_idle(self) -> None:
        """Mark session as idle"""
        if not self.idle_start:
            self.idle_start = datetime.utcnow()

    def is_expired(self) -> bool:
        """Check if session is expired"""
        return datetime.utcnow() > self.expires_at

    def is_idle(self, minutes: int = 30) -> bool:
        """Check if session is idle for specified minutes"""
        if not self.last_activity:
            return True

        idle_threshold = datetime.utcnow() - timedelta(minutes=minutes)
        return self.last_activity < idle_threshold

    def extend_session(self, hours: int = 1) -> None:
        """Extend session expiration"""
        self.expires_at = datetime.utcnow() + timedelta(hours=hours)
        self.update_activity()

    def increment_message_count(self) -> None:
        """Increment message counter"""
        self.message_count += 1
        self.update_activity()

    def set_conversation(self, conversation_id: ConversationId) -> None:
        """Set current conversation ID"""
        self.conversation_id = conversation_id
        self.update_activity()

    def set_context_value(self, key: str, value: Any) -> None:
        """Set a context value"""
        self.context[key] = value
        self.update_activity()

    def get_context_value(self, key: str, default: Any = None) -> Any:
        """Get a context value"""
        return self.context.get(key, default)

    def clear_context_value(self, key: str) -> None:
        """Clear a context value"""
        self.context.pop(key, None)
        self.update_timestamp()

    def set_preference(self, key: str, value: Any) -> None:
        """Set a user preference"""
        self.preferences[key] = value
        self.update_timestamp()

    def get_preference(self, key: str, default: Any = None) -> Any:
        """Get a user preference"""
        return self.preferences.get(key, default)

    def enable_feature(self, feature_name: str) -> None:
        """Enable a feature flag"""
        self.features[feature_name] = True
        self.update_timestamp()

    def disable_feature(self, feature_name: str) -> None:
        """Disable a feature flag"""
        self.features[feature_name] = False
        self.update_timestamp()

    def is_feature_enabled(self, feature_name: str, default: bool = False) -> bool:
        """Check if a feature is enabled"""
        return self.features.get(feature_name, default)

    def get_ttl_seconds(self) -> int:
        """Get TTL in seconds for Redis expiration"""
        if self.is_expired():
            return 0

        delta = self.expires_at - datetime.utcnow()
        return max(0, int(delta.total_seconds()))


class ConversationState(BaseRedisModel):
    """
    Redis cache structure for conversation state.
    Optimized for conversation flow management and state transitions.
    """

    # Core identifiers
    conversation_id: ConversationId
    tenant_id: TenantId
    session_id: Optional[SessionId] = None

    # Flow state
    current_state: Optional[str] = None
    previous_state: Optional[str] = None
    flow_id: Optional[str] = None

    # Intent and entity tracking
    intent_stack: List[str] = Field(default_factory=list, max_items=10)
    current_intent: Optional[str] = None
    intent_confidence: Optional[float] = Field(None, ge=0, le=1)

    # Slot and entity management
    slots: Dict[str, Any] = Field(default_factory=dict)
    entities: Dict[str, Any] = Field(default_factory=dict)

    # Context variables
    variables: Dict[str, Any] = Field(default_factory=dict)

    # Flow execution metadata
    turn_count: int = Field(default=0, ge=0)
    last_user_input: Optional[str] = None
    next_expected_inputs: List[str] = Field(default_factory=list, max_items=5)

    # Processing state
    processing_lock: bool = Field(default=False)
    processing_node: Optional[str] = None
    processing_started: Optional[datetime] = None

    @staticmethod
    def get_cache_key(tenant_id: TenantId, conversation_id: ConversationId) -> str:
        """Generate Redis cache key for conversation state"""
        return f"conversation_state:{tenant_id}:{conversation_id}"

    @staticmethod
    def get_lock_key(conversation_id: ConversationId) -> str:
        """Generate Redis lock key for conversation processing"""
        return f"conv_lock:{conversation_id}"

    def push_intent(self, intent: str, confidence: Optional[float] = None) -> None:
        """Push new intent to stack"""
        if intent != self.current_intent:
            if self.current_intent:
                self.intent_stack.append(self.current_intent)

            # Keep stack limited to last 10 intents
            if len(self.intent_stack) > 10:
                self.intent_stack = self.intent_stack[-10:]

        self.current_intent = intent
        self.intent_confidence = confidence
        self.update_timestamp()

    def pop_intent(self) -> Optional[str]:
        """Pop previous intent from stack"""
        if self.intent_stack:
            previous_intent = self.intent_stack.pop()
            self.current_intent = previous_intent
            self.update_timestamp()
            return previous_intent
        return None

    def transition_state(self, new_state: str) -> None:
        """Transition to new conversation state"""
        self.previous_state = self.current_state
        self.current_state = new_state
        self.turn_count += 1
        self.update_timestamp()

    def set_slot(self, slot_name: str, value: Any) -> None:
        """Set slot value"""
        self.slots[slot_name] = value
        self.update_timestamp()

    def get_slot(self, slot_name: str, default: Any = None) -> Any:
        """Get slot value"""
        return self.slots.get(slot_name, default)

    def clear_slot(self, slot_name: str) -> None:
        """Clear slot value"""
        self.slots.pop(slot_name, None)
        self.update_timestamp()

    def has_required_slots(self, required_slots: List[str]) -> bool:
        """Check if all required slots are filled"""
        return all(slot in self.slots and self.slots[slot] is not None for slot in required_slots)

    def set_entity(self, entity_type: str, entity_value: Any) -> None:
        """Set entity value"""
        self.entities[entity_type] = entity_value
        self.update_timestamp()

    def get_entity(self, entity_type: str, default: Any = None) -> Any:
        """Get entity value"""
        return self.entities.get(entity_type, default)

    def set_variable(self, var_name: str, value: Any) -> None:
        """Set context variable"""
        self.variables[var_name] = value
        self.update_timestamp()

    def get_variable(self, var_name: str, default: Any = None) -> Any:
        """Get context variable"""
        return self.variables.get(var_name, default)

    def acquire_processing_lock(self, processing_node: str, timeout_seconds: int = 30) -> bool:
        """Acquire processing lock for conversation"""
        if self.processing_lock and self.processing_started:
            # Check if lock has timed out
            timeout = self.processing_started + timedelta(seconds=timeout_seconds)
            if datetime.utcnow() > timeout:
                self.release_processing_lock()
            else:
                return False  # Lock is still active

        self.processing_lock = True
        self.processing_node = processing_node
        self.processing_started = datetime.utcnow()
        self.update_timestamp()
        return True

    def release_processing_lock(self) -> None:
        """Release processing lock"""
        self.processing_lock = False
        self.processing_node = None
        self.processing_started = None
        self.update_timestamp()

    def is_processing_locked(self) -> bool:
        """Check if conversation is processing locked"""
        return self.processing_lock

    def set_expected_inputs(self, inputs: List[str]) -> None:
        """Set next expected user inputs"""
        self.next_expected_inputs = inputs[:5]  # Limit to 5
        self.update_timestamp()

    def clear_expected_inputs(self) -> None:
        """Clear expected inputs"""
        self.next_expected_inputs = []
        self.update_timestamp()

    def reset_conversation(self) -> None:
        """Reset conversation state for new conversation"""
        self.current_state = None
        self.previous_state = None
        self.intent_stack = []
        self.current_intent = None
        self.intent_confidence = None
        self.slots = {}
        self.entities = {}
        self.variables = {}
        self.turn_count = 0
        self.next_expected_inputs = []
        self.release_processing_lock()
        self.update_timestamp()


class ActiveConversations(BaseRedisModel):
    """
    Redis structure for tracking active conversations per tenant.
    Optimized for quick lookups and conversation management.
    """

    tenant_id: TenantId
    active_conversations: Dict[str, Dict[str, Any]] = Field(default_factory=dict)  # conv_id -> metadata
    last_updated: datetime = Field(default_factory=datetime.utcnow)

    @staticmethod
    def get_cache_key(tenant_id: TenantId) -> str:
        """Generate Redis cache key for active conversations"""
        return f"active_conversations:{tenant_id}"

    def add_conversation(
            self,
            conversation_id: ConversationId,
            user_id: UserId,
            channel: str,
            started_at: Optional[datetime] = None
    ) -> None:
        """Add a conversation to active list"""
        self.active_conversations[conversation_id] = {
            "user_id": user_id,
            "channel": channel,
            "started_at": (started_at or datetime.utcnow()).isoformat(),
            "last_activity": datetime.utcnow().isoformat()
        }
        self.last_updated = datetime.utcnow()
        self.update_timestamp()

    def remove_conversation(self, conversation_id: ConversationId) -> None:
        """Remove conversation from active list"""
        self.active_conversations.pop(conversation_id, None)
        self.last_updated = datetime.utcnow()
        self.update_timestamp()

    def update_conversation_activity(self, conversation_id: ConversationId) -> None:
        """Update last activity for a conversation"""
        if conversation_id in self.active_conversations:
            self.active_conversations[conversation_id]["last_activity"] = datetime.utcnow().isoformat()
            self.last_updated = datetime.utcnow()
            self.update_timestamp()

    def get_conversation_count(self) -> int:
        """Get count of active conversations"""
        return len(self.active_conversations)

    def get_conversations_by_user(self, user_id: UserId) -> List[str]:
        """Get conversation IDs for a specific user"""
        return [
            conv_id for conv_id, metadata in self.active_conversations.items()
            if metadata.get("user_id") == user_id
        ]

    def get_conversations_by_channel(self, channel: str) -> List[str]:
        """Get conversation IDs for a specific channel"""
        return [
            conv_id for conv_id, metadata in self.active_conversations.items()
            if metadata.get("channel") == channel
        ]

    def cleanup_stale_conversations(self, hours: int = 24) -> List[str]:
        """Remove stale conversations and return their IDs"""
        cutoff_time = datetime.utcnow() - timedelta(hours=hours)
        stale_conversations = []

        for conv_id, metadata in list(self.active_conversations.items()):
            last_activity_str = metadata.get("last_activity")
            if last_activity_str:
                try:
                    last_activity = datetime.fromisoformat(last_activity_str)
                    if last_activity < cutoff_time:
                        stale_conversations.append(conv_id)
                        del self.active_conversations[conv_id]
                except ValueError:
                    # Invalid datetime, remove conversation
                    stale_conversations.append(conv_id)
                    del self.active_conversations[conv_id]

        if stale_conversations:
            self.last_updated = datetime.utcnow()
            self.update_timestamp()

        return stale_conversations


# Utility functions for Redis operations
class SessionCacheManager:
    """
    Utility class for managing session cache operations.
    Provides high-level methods for session management.
    """

    @staticmethod
    def generate_session_fingerprint(
            user_agent: Optional[str],
            ip_address: Optional[str],
            additional_data: Optional[Dict[str, str]] = None
    ) -> str:
        """Generate a session fingerprint for security"""
        fingerprint_data = []

        if user_agent:
            fingerprint_data.append(user_agent)
        if ip_address:
            fingerprint_data.append(ip_address)
        if additional_data:
            for key in sorted(additional_data.keys()):
                fingerprint_data.append(f"{key}:{additional_data[key]}")

        # Hash the combined data
        combined = "|".join(fingerprint_data)
        return hashlib.sha256(combined.encode()).hexdigest()

    @staticmethod
    def calculate_session_ttl(
            last_activity: datetime,
            max_idle_minutes: int = 30,
            max_session_hours: int = 24
    ) -> int:
        """Calculate appropriate TTL for session cache"""
        now = datetime.utcnow()

        # Calculate time since last activity
        idle_time = now - last_activity
        max_idle = timedelta(minutes=max_idle_minutes)

        # If idle time exceeds max, session should expire soon
        if idle_time > max_idle:
            return 60  # 1 minute TTL for cleanup

        # Calculate remaining time based on max session duration
        session_start = last_activity  # Approximate
        session_duration = now - session_start
        max_duration = timedelta(hours=max_session_hours)

        remaining = max_duration - session_duration
        if remaining.total_seconds() <= 0:
            return 60  # 1 minute TTL for cleanup

        # Return TTL in seconds, minimum 5 minutes
        return max(300, int(remaining.total_seconds()))

    @staticmethod
    def validate_session_data(session_data: Dict[str, str]) -> bool:
        """Validate session data integrity"""
        required_fields = ["session_id", "tenant_id", "user_id", "channel"]

        for field in required_fields:
            if field not in session_data or not session_data[field]:
                return False

        # Validate timestamps
        timestamp_fields = ["created_at", "updated_at", "last_activity", "expires_at"]
        for field in timestamp_fields:
            if field in session_data:
                try:
                    datetime.fromisoformat(session_data[field])
                except ValueError:
                    return False

        return True


# Export classes and utilities
__all__ = [
    "SessionCache",
    "ConversationState",
    "ActiveConversations",
    "SessionCacheManager"
]