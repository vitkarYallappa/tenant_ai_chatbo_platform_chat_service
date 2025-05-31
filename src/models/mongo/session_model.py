# src/models/mongo/session_model.py
"""
MongoDB document model for session storage.
Represents user sessions with conversation history and context.
"""

from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional
from pydantic import BaseModel, Field, validator
from bson import ObjectId

from src.models.base_model import BaseMongoModel, TimestampMixin
from src.models.types import (
    ChannelType, TenantId, UserId, SessionId, ConversationId,
    Priority
)


class DeviceInfo(BaseModel):
    """Device information for the session"""
    type: str = Field(..., max_length=20)  # mobile, desktop, tablet, voice
    os: Optional[str] = Field(None, max_length=50)
    browser: Optional[str] = Field(None, max_length=50)
    browser_version: Optional[str] = Field(None, max_length=20)
    screen_resolution: Optional[str] = Field(None, max_length=20)
    user_agent: Optional[str] = Field(None, max_length=500)


class LocationInfo(BaseModel):
    """Location information for the session (privacy-compliant)"""
    country: Optional[str] = Field(None, max_length=2)  # ISO country code
    region: Optional[str] = Field(None, max_length=100)
    city: Optional[str] = Field(None, max_length=100)  # May be anonymized
    timezone: Optional[str] = Field(None, max_length=50)
    ip_hash: Optional[str] = Field(None, max_length=64)  # Hashed IP for privacy


class SessionMetrics(BaseModel):
    """Session activity and engagement metrics"""
    total_conversations: int = Field(default=0, ge=0)
    total_messages: int = Field(default=0, ge=0)
    total_duration_seconds: int = Field(default=0, ge=0)

    # Activity tracking
    page_views: int = Field(default=0, ge=0)
    clicks: int = Field(default=0, ge=0)
    idle_time_seconds: int = Field(default=0, ge=0)

    # Engagement metrics
    satisfaction_scores: List[float] = Field(default_factory=list, max_items=10)
    escalations: int = Field(default=0, ge=0)
    successful_completions: int = Field(default=0, ge=0)

    def add_satisfaction_score(self, score: float) -> None:
        """Add a satisfaction score (1-5)"""
        if 1 <= score <= 5:
            self.satisfaction_scores.append(score)
            # Keep only last 10 scores
            if len(self.satisfaction_scores) > 10:
                self.satisfaction_scores = self.satisfaction_scores[-10:]

    def get_average_satisfaction(self) -> Optional[float]:
        """Calculate average satisfaction score"""
        if not self.satisfaction_scores:
            return None
        return sum(self.satisfaction_scores) / len(self.satisfaction_scores)


class ConversationRef(BaseModel):
    """Reference to a conversation within the session"""
    conversation_id: ConversationId
    started_at: datetime
    ended_at: Optional[datetime] = None
    message_count: int = Field(default=0, ge=0)
    status: str = Field(..., max_length=20)  # active, completed, abandoned, escalated
    primary_intent: Optional[str] = Field(None, max_length=100)
    satisfaction_score: Optional[float] = Field(None, ge=1, le=5)


class SessionPreferences(BaseModel):
    """User preferences for the session"""
    language: str = Field(default="en", max_length=5)
    timezone: Optional[str] = Field(None, max_length=50)
    notification_preferences: Dict[str, bool] = Field(default_factory=dict)
    accessibility_settings: Dict[str, Any] = Field(default_factory=dict)
    theme_preferences: Dict[str, str] = Field(default_factory=dict)


class SessionContext(BaseModel):
    """Context information maintained throughout the session"""
    # Entry point and referrer information
    entry_point: Optional[str] = Field(None, max_length=200)
    referrer_url: Optional[str] = Field(None, max_length=500)
    utm_parameters: Dict[str, str] = Field(default_factory=dict)

    # Business context
    customer_segment: Optional[str] = Field(None, max_length=50)
    customer_tier: Optional[str] = Field(None, max_length=20)
    account_type: Optional[str] = Field(None, max_length=50)

    # Session variables (persistent across conversations)
    persistent_variables: Dict[str, Any] = Field(default_factory=dict)

    # Feature flags and experiments
    feature_flags: Dict[str, bool] = Field(default_factory=dict)
    experiment_assignments: Dict[str, str] = Field(default_factory=dict)


class SecurityInfo(BaseModel):
    """Security-related session information"""
    ip_hash: Optional[str] = Field(None, max_length=64)  # Hashed for privacy
    fingerprint_hash: Optional[str] = Field(None, max_length=64)
    is_bot_detected: bool = Field(default=False)
    risk_score: Optional[float] = Field(None, ge=0, le=1)

    # Authentication status
    authenticated: bool = Field(default=False)
    auth_method: Optional[str] = Field(None, max_length=50)
    auth_timestamp: Optional[datetime] = None

    # Suspicious activity flags
    suspicious_activity: bool = Field(default=False)
    suspicious_reasons: List[str] = Field(default_factory=list)


class SessionDocument(BaseMongoModel, TimestampMixin):
    """
    MongoDB document structure for user sessions.
    Tracks user sessions across conversations with comprehensive context.
    """

    # Core identifiers
    session_id: SessionId = Field(..., max_length=100)
    tenant_id: TenantId = Field(..., max_length=100)
    user_id: UserId = Field(..., max_length=255)

    # Session lifecycle
    started_at: datetime = Field(default_factory=datetime.utcnow)
    last_activity_at: datetime = Field(default_factory=datetime.utcnow)
    ended_at: Optional[datetime] = None
    expires_at: datetime = Field(
        default_factory=lambda: datetime.utcnow() + timedelta(hours=24)
    )

    # Channel and device information
    primary_channel: ChannelType
    channels_used: List[ChannelType] = Field(default_factory=list)
    device_info: Optional[DeviceInfo] = None
    location_info: Optional[LocationInfo] = None

    # Session context and preferences
    context: SessionContext = Field(default_factory=SessionContext)
    preferences: SessionPreferences = Field(default_factory=SessionPreferences)

    # Activity tracking
    metrics: SessionMetrics = Field(default_factory=SessionMetrics)
    conversations: List[ConversationRef] = Field(default_factory=list, max_items=50)

    # Security information
    security: SecurityInfo = Field(default_factory=SecurityInfo)

    # Session status
    status: str = Field(default="active", regex=r'^(active|idle|ended|expired|terminated)$')
    termination_reason: Optional[str] = Field(None, max_length=100)

    # Compliance and privacy
    data_retention_until: Optional[datetime] = None
    privacy_consent: Dict[str, bool] = Field(default_factory=dict)
    anonymized: bool = Field(default=False)

    class Config:
        collection_name = "sessions"

    @validator('session_id', 'tenant_id', 'user_id')
    def validate_required_ids(cls, v):
        if not v or not v.strip():
            raise ValueError("ID fields cannot be empty")
        return v.strip()

    def update_activity(self) -> None:
        """Update last activity timestamp"""
        self.last_activity_at = datetime.utcnow()
        self.update_timestamp()

    def extend_session(self, hours: int = 1) -> None:
        """Extend session expiration time"""
        self.expires_at = datetime.utcnow() + timedelta(hours=hours)
        self.update_activity()

    def is_expired(self) -> bool:
        """Check if session has expired"""
        return datetime.utcnow() > self.expires_at

    def is_idle(self, minutes: int = 30) -> bool:
        """Check if session is idle (no activity for specified minutes)"""
        if not self.last_activity_at:
            return True

        idle_threshold = datetime.utcnow() - timedelta(minutes=minutes)
        return self.last_activity_at < idle_threshold

    def end_session(self, reason: str = "user_logout") -> None:
        """End the session"""
        self.status = "ended"
        self.ended_at = datetime.utcnow()
        self.termination_reason = reason
        self.update_timestamp()

    def add_conversation(
            self,
            conversation_id: ConversationId,
            primary_intent: Optional[str] = None
    ) -> ConversationRef:
        """Add a new conversation to the session"""
        conversation_ref = ConversationRef(
            conversation_id=conversation_id,
            started_at=datetime.utcnow(),
            status="active",
            primary_intent=primary_intent
        )

        self.conversations.append(conversation_ref)

        # Keep only last 50 conversations
        if len(self.conversations) > 50:
            self.conversations = self.conversations[-50:]

        self.metrics.total_conversations += 1
        self.update_activity()

        return conversation_ref

    def update_conversation(
            self,
            conversation_id: ConversationId,
            status: Optional[str] = None,
            message_count: Optional[int] = None,
            satisfaction_score: Optional[float] = None
    ) -> None:
        """Update conversation information in the session"""
        for conv_ref in self.conversations:
            if conv_ref.conversation_id == conversation_id:
                if status:
                    conv_ref.status = status
                    if status in ["completed", "abandoned", "escalated"]:
                        conv_ref.ended_at = datetime.utcnow()

                if message_count is not None:
                    conv_ref.message_count = message_count

                if satisfaction_score is not None:
                    conv_ref.satisfaction_score = satisfaction_score
                    self.metrics.add_satisfaction_score(satisfaction_score)

                break

        self.update_activity()

    def add_channel(self, channel: ChannelType) -> None:
        """Add a channel to the list of channels used in this session"""
        if channel not in self.channels_used:
            self.channels_used.append(channel)
            self.update_activity()

    def set_device_info(
            self,
            device_type: str,
            os: Optional[str] = None,
            browser: Optional[str] = None,
            user_agent: Optional[str] = None
    ) -> None:
        """Set device information for the session"""
        self.device_info = DeviceInfo(
            type=device_type,
            os=os,
            browser=browser,
            user_agent=user_agent
        )
        self.update_timestamp()

    def set_location_info(
            self,
            country: Optional[str] = None,
            region: Optional[str] = None,
            city: Optional[str] = None,
            timezone: Optional[str] = None
    ) -> None:
        """Set location information for the session"""
        self.location_info = LocationInfo(
            country=country,
            region=region,
            city=city,
            timezone=timezone
        )
        self.update_timestamp()

    def set_security_info(
            self,
            ip_hash: Optional[str] = None,
            fingerprint_hash: Optional[str] = None,
            is_bot_detected: bool = False
    ) -> None:
        """Set security information for the session"""
        if not self.security:
            self.security = SecurityInfo()

        if ip_hash:
            self.security.ip_hash = ip_hash
        if fingerprint_hash:
            self.security.fingerprint_hash = fingerprint_hash
        self.security.is_bot_detected = is_bot_detected

        self.update_timestamp()

    def mark_authenticated(self, auth_method: str) -> None:
        """Mark session as authenticated"""
        self.security.authenticated = True
        self.security.auth_method = auth_method
        self.security.auth_timestamp = datetime.utcnow()
        self.update_activity()

    def flag_suspicious_activity(self, reasons: List[str]) -> None:
        """Flag session for suspicious activity"""
        self.security.suspicious_activity = True
        self.security.suspicious_reasons.extend(reasons)
        self.update_timestamp()

    def set_persistent_variable(self, key: str, value: Any) -> None:
        """Set a persistent session variable"""
        self.context.persistent_variables[key] = value
        self.update_activity()

    def get_persistent_variable(self, key: str, default: Any = None) -> Any:
        """Get a persistent session variable"""
        return self.context.persistent_variables.get(key, default)

    def clear_persistent_variable(self, key: str) -> None:
        """Clear a persistent session variable"""
        self.context.persistent_variables.pop(key, None)
        self.update_activity()

    def set_feature_flag(self, flag_name: str, enabled: bool) -> None:
        """Set a feature flag for the session"""
        self.context.feature_flags[flag_name] = enabled
        self.update_timestamp()

    def is_feature_enabled(self, flag_name: str, default: bool = False) -> bool:
        """Check if a feature flag is enabled"""
        return self.context.feature_flags.get(flag_name, default)

    def set_data_retention(self, days: int) -> None:
        """Set data retention period for the session"""
        self.data_retention_until = datetime.utcnow() + timedelta(days=days)
        self.update_timestamp()

    def should_be_cleaned_up(self) -> bool:
        """Check if session should be cleaned up"""
        # Check retention period
        if self.data_retention_until and datetime.utcnow() > self.data_retention_until:
            return True

        # Check if session is very old (default: 90 days)
        if self.started_at < datetime.utcnow() - timedelta(days=90):
            return True

        return False

    def get_active_conversation(self) -> Optional[ConversationRef]:
        """Get the currently active conversation"""
        for conv_ref in reversed(self.conversations):
            if conv_ref.status == "active":
                return conv_ref
        return None

    def calculate_total_duration(self) -> int:
        """Calculate total session duration in seconds"""
        end_time = self.ended_at or datetime.utcnow()
        delta = end_time - self.started_at
        return int(delta.total_seconds())

    @classmethod
    def create_new(
            cls,
            session_id: SessionId,
            tenant_id: TenantId,
            user_id: UserId,
            primary_channel: ChannelType,
            **kwargs
    ) -> "SessionDocument":
        """
        Create a new session document.

        Args:
            session_id: Unique session identifier
            tenant_id: Tenant identifier
            user_id: User identifier
            primary_channel: Primary communication channel
            **kwargs: Additional fields

        Returns:
            New SessionDocument instance
        """
        session = cls(
            session_id=session_id,
            tenant_id=tenant_id,
            user_id=user_id,
            primary_channel=primary_channel,
            **kwargs
        )

        # Add primary channel to channels used
        session.channels_used.append(primary_channel)

        return session

    def to_summary_dict(self) -> Dict[str, Any]:
        """
        Convert to summary dictionary for API responses.

        Returns:
            Summary dictionary with key session information
        """
        return {
            "session_id": self.session_id,
            "user_id": self.user_id,
            "status": self.status,
            "started_at": self.started_at,
            "last_activity_at": self.last_activity_at,
            "duration_seconds": self.calculate_total_duration(),
            "primary_channel": self.primary_channel,
            "channels_used": self.channels_used,
            "total_conversations": self.metrics.total_conversations,
            "total_messages": self.metrics.total_messages,
            "average_satisfaction": self.metrics.get_average_satisfaction(),
            "device_type": self.device_info.type if self.device_info else None,
            "country": self.location_info.country if self.location_info else None,
            "authenticated": self.security.authenticated if self.security else False,
            "suspicious": self.security.suspicious_activity if self.security else False
        }