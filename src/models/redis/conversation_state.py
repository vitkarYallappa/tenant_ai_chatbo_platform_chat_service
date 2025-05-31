# src/models/redis/conversation_state.py
"""
Redis data structures for conversation state management.
Optimized for conversation flow management and state transitions.
"""

from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional
from pydantic import BaseModel, Field, validator
import json

from src.models.base_model import BaseRedisModel
from src.models.types import (
    TenantId, ConversationId, SessionId, Priority
)


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
    flow_version: Optional[str] = None

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
    processing_timeout_seconds: int = Field(default=30, ge=5, le=300)

    # State transition history (limited for performance)
    state_transitions: List[Dict[str, Any]] = Field(default_factory=list, max_items=20)

    # Error handling
    error_count: int = Field(default=0, ge=0)
    last_error: Optional[str] = None
    last_error_timestamp: Optional[datetime] = None

    @staticmethod
    def get_cache_key(tenant_id: TenantId, conversation_id: ConversationId) -> str:
        """Generate Redis cache key for conversation state"""
        return f"conversation_state:{tenant_id}:{conversation_id}"

    @staticmethod
    def get_lock_key(conversation_id: ConversationId) -> str:
        """Generate Redis lock key for conversation processing"""
        return f"conv_lock:{conversation_id}"

    @staticmethod
    def get_flow_state_key(tenant_id: TenantId, flow_id: str, state: str) -> str:
        """Generate Redis key for flow state configuration cache"""
        return f"flow_state:{tenant_id}:{flow_id}:{state}"

    def push_intent(self, intent: str, confidence: Optional[float] = None) -> None:
        """
        Push new intent to stack.

        Args:
            intent: Intent name
            confidence: Intent confidence score
        """
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
        """
        Pop previous intent from stack.

        Returns:
            Previous intent or None if stack is empty
        """
        if self.intent_stack:
            previous_intent = self.intent_stack.pop()
            self.current_intent = previous_intent
            self.intent_confidence = None  # Clear confidence for popped intent
            self.update_timestamp()
            return previous_intent
        return None

    def get_intent_history(self, limit: int = 5) -> List[str]:
        """
        Get recent intent history.

        Args:
            limit: Maximum number of intents to return

        Returns:
            List of recent intents
        """
        history = []
        if self.current_intent:
            history.append(self.current_intent)
        history.extend(reversed(self.intent_stack[-limit:]))
        return history[:limit]

    def transition_state(self, new_state: str, metadata: Optional[Dict[str, Any]] = None) -> None:
        """
        Transition to new conversation state.

        Args:
            new_state: Target state name
            metadata: Optional transition metadata
        """
        # Record state transition
        transition = {
            "from_state": self.current_state,
            "to_state": new_state,
            "timestamp": datetime.utcnow().isoformat(),
            "turn_count": self.turn_count,
            "metadata": metadata or {}
        }

        self.state_transitions.append(transition)

        # Keep transition history limited
        if len(self.state_transitions) > 20:
            self.state_transitions = self.state_transitions[-20:]

        # Update state
        self.previous_state = self.current_state
        self.current_state = new_state
        self.turn_count += 1
        self.update_timestamp()

    def can_transition_to(self, target_state: str, allowed_transitions: Dict[str, List[str]]) -> bool:
        """
        Check if transition to target state is allowed.

        Args:
            target_state: Target state name
            allowed_transitions: Dictionary of state -> allowed next states

        Returns:
            True if transition is allowed
        """
        if not self.current_state:
            return True  # Can transition to any state from initial state

        allowed_states = allowed_transitions.get(self.current_state, [])
        return target_state in allowed_states

    def set_slot(self, slot_name: str, value: Any) -> None:
        """
        Set slot value with validation.

        Args:
            slot_name: Slot name
            value: Slot value
        """
        self.slots[slot_name] = value
        self.update_timestamp()

    def get_slot(self, slot_name: str, default: Any = None) -> Any:
        """
        Get slot value.

        Args:
            slot_name: Slot name
            default: Default value if slot doesn't exist

        Returns:
            Slot value or default
        """
        return self.slots.get(slot_name, default)

    def clear_slot(self, slot_name: str) -> None:
        """
        Clear slot value.

        Args:
            slot_name: Slot name to clear
        """
        self.slots.pop(slot_name, None)
        self.update_timestamp()

    def clear_all_slots(self) -> None:
        """Clear all slot values."""
        self.slots.clear()
        self.update_timestamp()

    def has_required_slots(self, required_slots: List[str]) -> bool:
        """
        Check if all required slots are filled.

        Args:
            required_slots: List of required slot names

        Returns:
            True if all required slots have values
        """
        return all(
            slot in self.slots and self.slots[slot] is not None
            for slot in required_slots
        )

    def get_missing_slots(self, required_slots: List[str]) -> List[str]:
        """
        Get list of missing required slots.

        Args:
            required_slots: List of required slot names

        Returns:
            List of missing slot names
        """
        return [
            slot for slot in required_slots
            if slot not in self.slots or self.slots[slot] is None
        ]

    def set_entity(self, entity_type: str, entity_value: Any, confidence: Optional[float] = None) -> None:
        """
        Set entity value.

        Args:
            entity_type: Entity type
            entity_value: Entity value
            confidence: Optional confidence score
        """
        entity_data = {
            "value": entity_value,
            "timestamp": datetime.utcnow().isoformat()
        }
        if confidence is not None:
            entity_data["confidence"] = confidence

        self.entities[entity_type] = entity_data
        self.update_timestamp()

    def get_entity(self, entity_type: str, default: Any = None) -> Any:
        """
        Get entity value.

        Args:
            entity_type: Entity type
            default: Default value if entity doesn't exist

        Returns:
            Entity value or default
        """
        entity_data = self.entities.get(entity_type)
        if entity_data and isinstance(entity_data, dict):
            return entity_data.get("value", default)
        return default

    def get_entity_with_metadata(self, entity_type: str) -> Optional[Dict[str, Any]]:
        """
        Get entity with full metadata.

        Args:
            entity_type: Entity type

        Returns:
            Entity data with metadata or None
        """
        return self.entities.get(entity_type)

    def set_variable(self, var_name: str, value: Any) -> None:
        """
        Set context variable.

        Args:
            var_name: Variable name
            value: Variable value
        """
        self.variables[var_name] = value
        self.update_timestamp()

    def get_variable(self, var_name: str, default: Any = None) -> Any:
        """
        Get context variable.

        Args:
            var_name: Variable name
            default: Default value if variable doesn't exist

        Returns:
            Variable value or default
        """
        return self.variables.get(var_name, default)

    def clear_variable(self, var_name: str) -> None:
        """
        Clear context variable.

        Args:
            var_name: Variable name to clear
        """
        self.variables.pop(var_name, None)
        self.update_timestamp()

    def acquire_processing_lock(self, processing_node: str, timeout_seconds: Optional[int] = None) -> bool:
        """
        Acquire processing lock for conversation.

        Args:
            processing_node: Identifier of the processing node
            timeout_seconds: Optional custom timeout

        Returns:
            True if lock acquired, False if already locked
        """
        timeout = timeout_seconds or self.processing_timeout_seconds

        if self.processing_lock and self.processing_started:
            # Check if lock has timed out
            timeout_threshold = self.processing_started + timedelta(seconds=timeout)
            if datetime.utcnow() > timeout_threshold:
                self.release_processing_lock()
            else:
                return False  # Lock is still active

        self.processing_lock = True
        self.processing_node = processing_node
        self.processing_started = datetime.utcnow()
        self.update_timestamp()
        return True

    def release_processing_lock(self) -> None:
        """Release processing lock."""
        self.processing_lock = False
        self.processing_node = None
        self.processing_started = None
        self.update_timestamp()

    def is_processing_locked(self) -> bool:
        """
        Check if conversation is processing locked.

        Returns:
            True if locked and not timed out
        """
        if not self.processing_lock or not self.processing_started:
            return False

        # Check if lock has timed out
        timeout_threshold = self.processing_started + timedelta(seconds=self.processing_timeout_seconds)
        if datetime.utcnow() > timeout_threshold:
            self.release_processing_lock()
            return False

        return True

    def get_lock_remaining_time(self) -> Optional[int]:
        """
        Get remaining time for processing lock in seconds.

        Returns:
            Remaining seconds or None if not locked
        """
        if not self.processing_lock or not self.processing_started:
            return None

        timeout_threshold = self.processing_started + timedelta(seconds=self.processing_timeout_seconds)
        remaining = timeout_threshold - datetime.utcnow()
        return max(0, int(remaining.total_seconds()))

    def set_expected_inputs(self, inputs: List[str]) -> None:
        """
        Set next expected user inputs.

        Args:
            inputs: List of expected input types or patterns
        """
        self.next_expected_inputs = inputs[:5]  # Limit to 5
        self.update_timestamp()

    def add_expected_input(self, input_type: str) -> None:
        """
        Add an expected input type.

        Args:
            input_type: Expected input type or pattern
        """
        if input_type not in self.next_expected_inputs and len(self.next_expected_inputs) < 5:
            self.next_expected_inputs.append(input_type)
            self.update_timestamp()

    def clear_expected_inputs(self) -> None:
        """Clear expected inputs."""
        self.next_expected_inputs = []
        self.update_timestamp()

    def is_input_expected(self, input_type: str) -> bool:
        """
        Check if input type is expected.

        Args:
            input_type: Input type to check

        Returns:
            True if input is expected
        """
        return input_type in self.next_expected_inputs

    def record_error(self, error_message: str) -> None:
        """
        Record an error in conversation processing.

        Args:
            error_message: Error description
        """
        self.error_count += 1
        self.last_error = error_message
        self.last_error_timestamp = datetime.utcnow()
        self.update_timestamp()

    def clear_errors(self) -> None:
        """Clear error information."""
        self.error_count = 0
        self.last_error = None
        self.last_error_timestamp = None
        self.update_timestamp()

    def has_recent_errors(self, minutes: int = 5) -> bool:
        """
        Check if there are recent errors.

        Args:
            minutes: Time window to check for errors

        Returns:
            True if there are errors within the time window
        """
        if not self.last_error_timestamp:
            return False

        threshold = datetime.utcnow() - timedelta(minutes=minutes)
        return self.last_error_timestamp > threshold

    def reset_conversation(self, keep_flow_id: bool = True) -> None:
        """
        Reset conversation state for new conversation.

        Args:
            keep_flow_id: Whether to keep the current flow_id
        """
        flow_id_backup = self.flow_id if keep_flow_id else None
        flow_version_backup = self.flow_version if keep_flow_id else None

        self.current_state = None
        self.previous_state = None
        if not keep_flow_id:
            self.flow_id = None
            self.flow_version = None
        else:
            self.flow_id = flow_id_backup
            self.flow_version = flow_version_backup

        self.intent_stack = []
        self.current_intent = None
        self.intent_confidence = None
        self.slots = {}
        self.entities = {}
        self.variables = {}
        self.turn_count = 0
        self.last_user_input = None
        self.next_expected_inputs = []
        self.state_transitions = []
        self.clear_errors()
        self.release_processing_lock()
        self.update_timestamp()

    def get_conversation_summary(self) -> Dict[str, Any]:
        """
        Get a summary of the conversation state.

        Returns:
            Dictionary with conversation state summary
        """
        return {
            "conversation_id": self.conversation_id,
            "current_state": self.current_state,
            "current_intent": self.current_intent,
            "intent_confidence": self.intent_confidence,
            "turn_count": self.turn_count,
            "slots_count": len(self.slots),
            "entities_count": len(self.entities),
            "variables_count": len(self.variables),
            "is_locked": self.is_processing_locked(),
            "error_count": self.error_count,
            "last_updated": self.updated_at.isoformat() if self.updated_at else None,
            "flow_id": self.flow_id,
            "expected_inputs": self.next_expected_inputs
        }

    def get_context_for_llm(self) -> Dict[str, Any]:
        """
        Get conversation context formatted for LLM processing.

        Returns:
            Dictionary with context for LLM
        """
        return {
            "current_state": self.current_state,
            "current_intent": self.current_intent,
            "intent_confidence": self.intent_confidence,
            "intent_history": self.get_intent_history(),
            "slots": self.slots,
            "entities": {k: v.get("value") if isinstance(v, dict) else v for k, v in self.entities.items()},
            "variables": self.variables,
            "turn_count": self.turn_count,
            "expected_inputs": self.next_expected_inputs,
            "recent_transitions": self.state_transitions[-5:] if self.state_transitions else []
        }


class ConversationStateManager:
    """
    Utility class for managing conversation state operations.
    Provides high-level methods for conversation state management.
    """

    @staticmethod
    def create_initial_state(
            conversation_id: ConversationId,
            tenant_id: TenantId,
            flow_id: Optional[str] = None,
            initial_state: Optional[str] = None,
            session_id: Optional[SessionId] = None
    ) -> ConversationState:
        """
        Create initial conversation state.

        Args:
            conversation_id: Conversation identifier
            tenant_id: Tenant identifier
            flow_id: Optional flow identifier
            initial_state: Optional initial state
            session_id: Optional session identifier

        Returns:
            New ConversationState instance
        """
        state = ConversationState(
            conversation_id=conversation_id,
            tenant_id=tenant_id,
            session_id=session_id,
            flow_id=flow_id,
            current_state=initial_state
        )

        if initial_state:
            state.transition_state(initial_state, {"type": "initial_state"})

        return state

    @staticmethod
    def validate_state_data(state_data: Dict[str, Any]) -> bool:
        """
        Validate conversation state data integrity.

        Args:
            state_data: State data to validate

        Returns:
            True if valid, False otherwise
        """
        required_fields = ["conversation_id", "tenant_id"]

        for field in required_fields:
            if field not in state_data or not state_data[field]:
                return False

        # Validate data types
        if "turn_count" in state_data:
            try:
                int(state_data["turn_count"])
            except (ValueError, TypeError):
                return False

        if "intent_confidence" in state_data and state_data["intent_confidence"] is not None:
            try:
                confidence = float(state_data["intent_confidence"])
                if not 0 <= confidence <= 1:
                    return False
            except (ValueError, TypeError):
                return False

        return True

    @staticmethod
    def merge_context_updates(
            current_state: ConversationState,
            updates: Dict[str, Any]
    ) -> ConversationState:
        """
        Merge context updates into current state.

        Args:
            current_state: Current conversation state
            updates: Updates to apply

        Returns:
            Updated conversation state
        """
        # Update slots
        if "slots" in updates:
            for slot_name, slot_value in updates["slots"].items():
                current_state.set_slot(slot_name, slot_value)

        # Update entities
        if "entities" in updates:
            for entity_type, entity_data in updates["entities"].items():
                if isinstance(entity_data, dict):
                    current_state.set_entity(
                        entity_type,
                        entity_data.get("value"),
                        entity_data.get("confidence")
                    )
                else:
                    current_state.set_entity(entity_type, entity_data)

        # Update variables
        if "variables" in updates:
            for var_name, var_value in updates["variables"].items():
                current_state.set_variable(var_name, var_value)

        # Update intent if provided
        if "intent" in updates:
            current_state.push_intent(
                updates["intent"],
                updates.get("intent_confidence")
            )

        # Update expected inputs
        if "expected_inputs" in updates:
            current_state.set_expected_inputs(updates["expected_inputs"])

        return current_state


# Export classes and utilities
__all__ = [
    "ConversationState",
    "ConversationStateManager"
]