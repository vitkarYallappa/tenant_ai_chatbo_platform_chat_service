"""
Redis Session Cache Models
=========================

Data models for session management and caching in Redis with
comprehensive session lifecycle management and multi-tenant support.

Features:
- Session data structure with TTL management
- User context and preferences
- Device and location tracking
- Security metadata
- Serialization utilities
"""

from datetime import datetime, timedelta
from typing import Optional, Dict, Any, List
from dataclasses import dataclass, asdict, field
from enum import Enum
import json
import structlog

logger = structlog.get_logger(__name__)

# Type aliases
TenantId = str
SessionId = str
UserId = str


class SessionStatus(str, Enum):
    """Session status enumeration"""
    ACTIVE = "active"
    EXPIRED = "expired"
    TERMINATED = "terminated"
    SUSPENDED = "suspended"


class ChannelType(str, Enum):
    """Communication channel types"""
    WEB = "web"
    WHATSAPP = "whatsapp"
    MESSENGER = "messenger"
    SLACK = "slack"
    TEAMS = "teams"
    VOICE = "voice"
    SMS = "sms"
    API = "api"


@dataclass
class DeviceInfo:
    """Device information for session tracking"""
    device_type: str = "unknown"  # mobile, desktop, tablet, voice, etc.
    operating_system: Optional[str] = None
    browser: Optional[str] = None
    browser_version: Optional[str] = None
    screen_resolution: Optional[str] = None
    timezone: Optional[str] = None
    language: str = "en"
    user_agent: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "DeviceInfo":
        """Create from dictionary"""
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


@dataclass
class LocationInfo:
    """Location information for session tracking"""
    country: Optional[str] = None
    region: Optional[str] = None
    city: Optional[str] = None
    latitude: Optional[float] = None
    longitude: Optional[float] = None
    timezone: Optional[str] = None
    ip_address: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "LocationInfo":
        """Create from dictionary"""
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


@dataclass
class UserPreferences:
    """User preferences stored in session"""
    language: str = "en"
    timezone: Optional[str] = None
    theme: str = "light"
    notifications_enabled: bool = True
    voice_enabled: bool = False
    accessibility_features: List[str] = field(default_factory=list)
    custom_settings: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "UserPreferences":
        """Create from dictionary"""
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


@dataclass
class SessionContext:
    """Session context for conversation state"""
    current_flow_id: Optional[str] = None
    current_state: Optional[str] = None
    conversation_id: Optional[str] = None
    intent_stack: List[str] = field(default_factory=list)
    entities: Dict[str, Any] = field(default_factory=dict)
    slots: Dict[str, Any] = field(default_factory=dict)
    variables: Dict[str, Any] = field(default_factory=dict)
    conversation_history: List[str] = field(default_factory=list)

    def add_intent(self, intent: str) -> None:
        """Add intent to the stack"""
        self.intent_stack.append(intent)
        # Keep only last 10 intents
        self.intent_stack = self.intent_stack[-10:]

    def set_slot(self, key: str, value: Any) -> None:
        """Set a slot value"""
        self.slots[key] = value

    def get_slot(self, key: str, default: Any = None) -> Any:
        """Get a slot value"""
        return self.slots.get(key, default)

    def set_variable(self, key: str, value: Any) -> None:
        """Set a session variable"""
        self.variables[key] = value

    def get_variable(self, key: str, default: Any = None) -> Any:
        """Get a session variable"""
        return self.variables.get(key, default)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SessionContext":
        """Create from dictionary"""
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


@dataclass
class SecurityMetadata:
    """Security-related session metadata"""
    ip_address: Optional[str] = None
    last_seen_ip: Optional[str] = None
    suspicious_activity: bool = False
    failed_auth_attempts: int = 0
    last_auth_attempt: Optional[datetime] = None
    requires_mfa: bool = False
    mfa_verified: bool = False
    session_token_hash: Optional[str] = None
    csrf_token: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary with datetime serialization"""
        data = asdict(self)
        if data['last_auth_attempt']:
            data['last_auth_attempt'] = data['last_auth_attempt'].isoformat()
        return data

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SecurityMetadata":
        """Create from dictionary with datetime deserialization"""
        if 'last_auth_attempt' in data and data['last_auth_attempt']:
            if isinstance(data['last_auth_attempt'], str):
                data['last_auth_attempt'] = datetime.fromisoformat(data['last_auth_attempt'])

        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


@dataclass
class SessionData:
    """
    Comprehensive session data structure for Redis storage

    This class provides the complete session state including user context,
    preferences, security metadata, and conversation state.
    """
    # Core session identifiers
    session_id: SessionId
    tenant_id: TenantId
    user_id: UserId

    # Session lifecycle
    created_at: datetime = field(default_factory=datetime.utcnow)
    last_activity_at: datetime = field(default_factory=datetime.utcnow)
    expires_at: datetime = field(default_factory=lambda: datetime.utcnow() + timedelta(hours=1))
    status: SessionStatus = SessionStatus.ACTIVE

    # Communication context
    channel: ChannelType = ChannelType.WEB
    conversation_id: Optional[str] = None

    # User context and preferences
    user_preferences: UserPreferences = field(default_factory=UserPreferences)
    device_info: DeviceInfo = field(default_factory=DeviceInfo)
    location_info: LocationInfo = field(default_factory=LocationInfo)

    # Session context for conversation state
    context: SessionContext = field(default_factory=SessionContext)

    # Security metadata
    security: SecurityMetadata = field(default_factory=SecurityMetadata)

    # Business context
    customer_tier: Optional[str] = None
    subscription_type: Optional[str] = None
    feature_flags: Dict[str, bool] = field(default_factory=dict)

    # Session metadata
    session_tags: List[str] = field(default_factory=list)
    custom_data: Dict[str, Any] = field(default_factory=dict)

    @staticmethod
    def get_cache_key(tenant_id: TenantId, session_id: SessionId) -> str:
        """
        Generate Redis cache key for session

        Args:
            tenant_id: Tenant identifier
            session_id: Session identifier

        Returns:
            Redis cache key
        """
        return f"session:{tenant_id}:{session_id}"

    @staticmethod
    def get_user_sessions_key(tenant_id: TenantId, user_id: UserId) -> str:
        """
        Generate Redis key for user sessions index

        Args:
            tenant_id: Tenant identifier
            user_id: User identifier

        Returns:
            Redis key for user sessions index
        """
        return f"user_sessions:{tenant_id}:{user_id}"

    def to_redis_hash(self) -> Dict[str, str]:
        """
        Convert to Redis hash format (all string values)

        Returns:
            Dictionary with string values for Redis hash storage
        """
        data = {}

        # Core fields
        data['session_id'] = self.session_id
        data['tenant_id'] = self.tenant_id
        data['user_id'] = self.user_id
        data['status'] = self.status.value
        data['channel'] = self.channel.value

        # Timestamps
        data['created_at'] = self.created_at.isoformat()
        data['last_activity_at'] = self.last_activity_at.isoformat()
        data['expires_at'] = self.expires_at.isoformat()

        # Optional fields
        data['conversation_id'] = self.conversation_id or ""
        data['customer_tier'] = self.customer_tier or ""
        data['subscription_type'] = self.subscription_type or ""

        # Complex objects as JSON
        data['user_preferences'] = json.dumps(self.user_preferences.to_dict())
        data['device_info'] = json.dumps(self.device_info.to_dict())
        data['location_info'] = json.dumps(self.location_info.to_dict())
        data['context'] = json.dumps(self.context.to_dict())
        data['security'] = json.dumps(self.security.to_dict())
        data['feature_flags'] = json.dumps(self.feature_flags)
        data['session_tags'] = json.dumps(self.session_tags)
        data['custom_data'] = json.dumps(self.custom_data)

        return data

    @classmethod
    def from_redis_hash(cls, data: Dict[str, str]) -> "SessionData":
        """
        Create from Redis hash data

        Args:
            data: Redis hash data with string values

        Returns:
            SessionData instance
        """
        try:
            # Parse core fields
            session_data = {
                'session_id': data['session_id'],
                'tenant_id': data['tenant_id'],
                'user_id': data['user_id'],
                'status': SessionStatus(data['status']),
                'channel': ChannelType(data['channel'])
            }

            # Parse timestamps
            session_data['created_at'] = datetime.fromisoformat(data['created_at'])
            session_data['last_activity_at'] = datetime.fromisoformat(data['last_activity_at'])
            session_data['expires_at'] = datetime.fromisoformat(data['expires_at'])

            # Parse optional fields
            if data.get('conversation_id'):
                session_data['conversation_id'] = data['conversation_id']
            if data.get('customer_tier'):
                session_data['customer_tier'] = data['customer_tier']
            if data.get('subscription_type'):
                session_data['subscription_type'] = data['subscription_type']

            # Parse complex objects
            if data.get('user_preferences'):
                prefs_data = json.loads(data['user_preferences'])
                session_data['user_preferences'] = UserPreferences.from_dict(prefs_data)

            if data.get('device_info'):
                device_data = json.loads(data['device_info'])
                session_data['device_info'] = DeviceInfo.from_dict(device_data)

            if data.get('location_info'):
                location_data = json.loads(data['location_info'])
                session_data['location_info'] = LocationInfo.from_dict(location_data)

            if data.get('context'):
                context_data = json.loads(data['context'])
                session_data['context'] = SessionContext.from_dict(context_data)

            if data.get('security'):
                security_data = json.loads(data['security'])
                session_data['security'] = SecurityMetadata.from_dict(security_data)

            if data.get('feature_flags'):
                session_data['feature_flags'] = json.loads(data['feature_flags'])

            if data.get('session_tags'):
                session_data['session_tags'] = json.loads(data['session_tags'])

            if data.get('custom_data'):
                session_data['custom_data'] = json.loads(data['custom_data'])

            return cls(**session_data)

        except Exception as e:
            logger.error("Failed to deserialize session data", error=str(e), data_keys=list(data.keys()))
            raise ValueError(f"Failed to deserialize session data: {e}")

    def is_expired(self) -> bool:
        """
        Check if session is expired

        Returns:
            True if session is expired
        """
        return datetime.utcnow() > self.expires_at

    def update_activity(self) -> None:
        """Update last activity timestamp"""
        self.last_activity_at = datetime.utcnow()

    def extend_session(self, hours: int = 1) -> None:
        """
        Extend session expiration

        Args:
            hours: Number of hours to extend
        """
        self.expires_at = datetime.utcnow() + timedelta(hours=hours)
        self.update_activity()

    def add_context(self, key: str, value: Any) -> None:
        """
        Add context data to session

        Args:
            key: Context key
            value: Context value
        """
        self.context.set_variable(key, value)
        self.update_activity()

    def get_context(self, key: str, default: Any = None) -> Any:
        """
        Get context data from session

        Args:
            key: Context key
            default: Default value if key not found

        Returns:
            Context value or default
        """
        return self.context.get_variable(key, default)

    def set_conversation(self, conversation_id: str) -> None:
        """
        Set current conversation

        Args:
            conversation_id: Conversation identifier
        """
        self.conversation_id = conversation_id
        self.context.conversation_id = conversation_id
        self.update_activity()

    def add_tag(self, tag: str) -> None:
        """
        Add tag to session

        Args:
            tag: Tag to add
        """
        if tag not in self.session_tags:
            self.session_tags.append(tag)

    def remove_tag(self, tag: str) -> None:
        """
        Remove tag from session

        Args:
            tag: Tag to remove
        """
        if tag in self.session_tags:
            self.session_tags.remove(tag)

    def has_tag(self, tag: str) -> bool:
        """
        Check if session has tag

        Args:
            tag: Tag to check

        Returns:
            True if session has tag
        """
        return tag in self.session_tags

    def get_feature_flag(self, flag_name: str, default: bool = False) -> bool:
        """
        Get feature flag value

        Args:
            flag_name: Feature flag name
            default: Default value if flag not set

        Returns:
            Feature flag value
        """
        return self.feature_flags.get(flag_name, default)

    def set_feature_flag(self, flag_name: str, value: bool) -> None:
        """
        Set feature flag value

        Args:
            flag_name: Feature flag name
            value: Feature flag value
        """
        self.feature_flags[flag_name] = value

    def get_ttl_seconds(self) -> int:
        """
        Calculate TTL in seconds

        Returns:
            TTL in seconds (0 if expired)
        """
        delta = self.expires_at - datetime.utcnow()
        return max(0, int(delta.total_seconds()))

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert to dictionary for API responses

        Returns:
            Dictionary representation
        """
        return {
            'session_id': self.session_id,
            'tenant_id': self.tenant_id,
            'user_id': self.user_id,
            'status': self.status.value,
            'channel': self.channel.value,
            'created_at': self.created_at.isoformat(),
            'last_activity_at': self.last_activity_at.isoformat(),
            'expires_at': self.expires_at.isoformat(),
            'conversation_id': self.conversation_id,
            'customer_tier': self.customer_tier,
            'subscription_type': self.subscription_type,
            'user_preferences': self.user_preferences.to_dict(),
            'device_info': self.device_info.to_dict(),
            'location_info': self.location_info.to_dict(),
            'context': self.context.to_dict(),
            'security': self.security.to_dict(),
            'feature_flags': self.feature_flags,
            'session_tags': self.session_tags,
            'custom_data': self.custom_data,
            'ttl_seconds': self.get_ttl_seconds(),
            'is_expired': self.is_expired()
        }


# Utility functions for session management
def create_session(
        tenant_id: TenantId,
        user_id: UserId,
        channel: ChannelType = ChannelType.WEB,
        ttl_hours: int = 1,
        **kwargs
) -> SessionData:
    """
    Create a new session with default values

    Args:
        tenant_id: Tenant identifier
        user_id: User identifier
        channel: Communication channel
        ttl_hours: Session TTL in hours
        **kwargs: Additional session data

    Returns:
        New SessionData instance
    """
    import uuid

    session_id = str(uuid.uuid4())
    expires_at = datetime.utcnow() + timedelta(hours=ttl_hours)

    session_data = SessionData(
        session_id=session_id,
        tenant_id=tenant_id,
        user_id=user_id,
        channel=channel,
        expires_at=expires_at,
        **kwargs
    )

    return session_data


def parse_user_agent(user_agent: str) -> DeviceInfo:
    """
    Parse user agent string to extract device information

    Args:
        user_agent: User agent string

    Returns:
        DeviceInfo instance
    """
    # This is a simplified parser - in production you'd use a library like user-agents
    device_info = DeviceInfo(user_agent=user_agent)

    if user_agent:
        user_agent_lower = user_agent.lower()

        # Detect device type
        if any(mobile in user_agent_lower for mobile in ['mobile', 'android', 'iphone']):
            device_info.device_type = "mobile"
        elif 'tablet' in user_agent_lower or 'ipad' in user_agent_lower:
            device_info.device_type = "tablet"
        else:
            device_info.device_type = "desktop"

        # Detect browser
        if 'chrome' in user_agent_lower:
            device_info.browser = "Chrome"
        elif 'firefox' in user_agent_lower:
            device_info.browser = "Firefox"
        elif 'safari' in user_agent_lower:
            device_info.browser = "Safari"
        elif 'edge' in user_agent_lower:
            device_info.browser = "Edge"

        # Detect OS
        if 'windows' in user_agent_lower:
            device_info.operating_system = "Windows"
        elif 'mac' in user_agent_lower:
            device_info.operating_system = "macOS"
        elif 'linux' in user_agent_lower:
            device_info.operating_system = "Linux"
        elif 'android' in user_agent_lower:
            device_info.operating_system = "Android"
        elif 'ios' in user_agent_lower or 'iphone' in user_agent_lower:
            device_info.operating_system = "iOS"

    return device_info