# src/models/schemas/validation_schemas.py
"""
Specialized validation schemas and validators.
Provides custom validation logic for complex business rules and data integrity.
"""

from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Union, Callable
from pydantic import BaseModel, Field, validator, root_validator
import re
from enum import Enum

from src.models.base_model import BaseRequestModel
from src.models.types import ChannelType, MessageType, Priority


# ============================================================================
# VALIDATION RESULT SCHEMAS
# ============================================================================

class ValidationResult(BaseModel):
    """Result of a validation operation."""

    is_valid: bool
    errors: List[str] = Field(default_factory=list)
    warnings: List[str] = Field(default_factory=list)
    validated_data: Optional[Dict[str, Any]] = None

    def add_error(self, error: str) -> None:
        """Add a validation error."""
        self.errors.append(error)
        self.is_valid = False

    def add_warning(self, warning: str) -> None:
        """Add a validation warning."""
        self.warnings.append(warning)

    def has_errors(self) -> bool:
        """Check if there are validation errors."""
        return len(self.errors) > 0

    def has_warnings(self) -> bool:
        """Check if there are validation warnings."""
        return len(self.warnings) > 0


class FieldValidationResult(BaseModel):
    """Validation result for a specific field."""

    field_name: str
    is_valid: bool
    error_message: Optional[str] = None
    warning_message: Optional[str] = None
    normalized_value: Optional[Any] = None


# ============================================================================
# BUSINESS RULE VALIDATION SCHEMAS
# ============================================================================

class MessageContentValidation(BaseRequestModel):
    """Validation schema for message content business rules."""

    content_type: MessageType
    content_data: Dict[str, Any]
    channel: ChannelType
    user_context: Dict[str, Any] = Field(default_factory=dict)

    @root_validator
    def validate_content_rules(cls, values):
        """Validate content based on business rules."""
        content_type = values.get('content_type')
        content_data = values.get('content_data', {})
        channel = values.get('channel')

        errors = []

        # Text message validation
        if content_type == MessageType.TEXT:
            text = content_data.get('text', '')
            if not text or not text.strip():
                errors.append("Text content is required for text messages")
            elif len(text) > 4096:
                errors.append("Text content exceeds maximum length of 4096 characters")

            # Channel-specific text validation
            if channel == ChannelType.SMS and len(text) > 160:
                errors.append("SMS messages cannot exceed 160 characters")

        # Media message validation
        elif content_type in [MessageType.IMAGE, MessageType.FILE, MessageType.AUDIO, MessageType.VIDEO]:
            media = content_data.get('media')
            if not media:
                errors.append(f"Media content is required for {content_type} messages")
            else:
                # Validate media URL
                url = media.get('url', '')
                if not url or not url.startswith(('http://', 'https://')):
                    errors.append("Valid media URL is required")

                # Validate file size
                size_bytes = media.get('size_bytes', 0)
                max_sizes = {
                    MessageType.IMAGE: 10 * 1024 * 1024,  # 10MB
                    MessageType.AUDIO: 25 * 1024 * 1024,  # 25MB
                    MessageType.VIDEO: 50 * 1024 * 1024,  # 50MB
                    MessageType.FILE: 25 * 1024 * 1024  # 25MB
                }

                if size_bytes > max_sizes.get(content_type, 25 * 1024 * 1024):
                    errors.append(f"File size exceeds maximum allowed for {content_type}")

        # Location message validation
        elif content_type == MessageType.LOCATION:
            location = content_data.get('location')
            if not location:
                errors.append("Location data is required for location messages")
            else:
                lat = location.get('latitude')
                lng = location.get('longitude')
                if lat is None or lng is None:
                    errors.append("Latitude and longitude are required for location messages")
                elif not (-90 <= lat <= 90) or not (-180 <= lng <= 180):
                    errors.append("Invalid latitude/longitude coordinates")

        if errors:
            raise ValueError(f"Content validation failed: {'; '.join(errors)}")

        return values


class RateLimitValidation(BaseRequestModel):
    """Validation schema for rate limiting rules."""

    tenant_id: str
    user_id: Optional[str] = None
    api_key_id: Optional[str] = None
    endpoint: Optional[str] = None

    # Rate limit configuration
    requests_per_minute: int = Field(..., gt=0, le=100000)
    requests_per_hour: int = Field(..., gt=0, le=1000000)
    requests_per_day: int = Field(..., gt=0, le=10000000)

    # Burst configuration
    burst_capacity: int = Field(..., gt=0, le=10000)
    burst_refill_rate: float = Field(..., gt=0, le=1000)

    # Current usage (for validation)
    current_minute_usage: int = Field(default=0, ge=0)
    current_hour_usage: int = Field(default=0, ge=0)
    current_day_usage: int = Field(default=0, ge=0)

    @validator('requests_per_hour')
    def validate_hour_limit(cls, v, values):
        minute_limit = values.get('requests_per_minute', 0)
        if v < minute_limit:
            raise ValueError('Hourly limit must be >= minute limit')
        if v < minute_limit * 10:  # Reasonable ratio
            raise ValueError('Hourly limit should be at least 10x minute limit')
        return v

    @validator('requests_per_day')
    def validate_day_limit(cls, v, values):
        hour_limit = values.get('requests_per_hour', 0)
        if v < hour_limit:
            raise ValueError('Daily limit must be >= hourly limit')
        if v < hour_limit * 10:  # Reasonable ratio
            raise ValueError('Daily limit should be at least 10x hourly limit')
        return v

    @validator('burst_refill_rate')
    def validate_refill_rate(cls, v, values):
        minute_limit = values.get('requests_per_minute', 0)
        expected_rate = minute_limit / 60.0  # requests per second
        if v > expected_rate * 2:  # Allow some burst
            raise ValueError('Burst refill rate is too high compared to minute limit')
        return v

    def check_rate_limit_exceeded(self) -> ValidationResult:
        """Check if current usage exceeds any rate limits."""
        result = ValidationResult(is_valid=True)

        if self.current_minute_usage >= self.requests_per_minute:
            result.add_error(f"Minute rate limit exceeded: {self.current_minute_usage}/{self.requests_per_minute}")

        if self.current_hour_usage >= self.requests_per_hour:
            result.add_error(f"Hour rate limit exceeded: {self.current_hour_usage}/{self.requests_per_hour}")

        if self.current_day_usage >= self.requests_per_day:
            result.add_error(f"Day rate limit exceeded: {self.current_day_usage}/{self.requests_per_day}")

        # Warning thresholds (80% of limit)
        if self.current_minute_usage >= self.requests_per_minute * 0.8:
            result.add_warning(f"Approaching minute rate limit: {self.current_minute_usage}/{self.requests_per_minute}")

        return result


# ============================================================================
# SECURITY VALIDATION SCHEMAS
# ============================================================================

class SecurityValidation(BaseRequestModel):
    """Validation schema for security-related checks."""

    # Input data to validate
    user_input: str
    content_type: str = "text"

    # Security settings
    max_length: int = Field(default=4096, gt=0, le=50000)
    allow_html: bool = Field(default=False)
    allow_scripts: bool = Field(default=False)
    check_pii: bool = Field(default=True)
    check_profanity: bool = Field(default=True)

    # Context
    user_id: Optional[str] = None
    session_id: Optional[str] = None

    def validate_security(self) -> ValidationResult:
        """Perform comprehensive security validation."""
        result = ValidationResult(is_valid=True)

        # Length validation
        if len(self.user_input) > self.max_length:
            result.add_error(f"Input exceeds maximum length of {self.max_length} characters")

        # HTML/Script validation
        if not self.allow_html and self._contains_html():
            result.add_error("HTML content is not allowed")

        if not self.allow_scripts and self._contains_scripts():
            result.add_error("Script content is not allowed")

        # PII detection
        if self.check_pii:
            pii_found = self._detect_pii()
            if pii_found:
                result.add_warning(f"Potential PII detected: {', '.join(pii_found)}")

        # Profanity detection
        if self.check_profanity and self._contains_profanity():
            result.add_warning("Potentially inappropriate content detected")

        # Injection attack detection
        if self._detect_injection_attempts():
            result.add_error("Potential injection attack detected")

        return result

    def _contains_html(self) -> bool:
        """Check if input contains HTML tags."""
        html_pattern = re.compile(r'<[^>]+>')
        return bool(html_pattern.search(self.user_input))

    def _contains_scripts(self) -> bool:
        """Check if input contains script tags or javascript."""
        script_patterns = [
            r'<script[^>]*>',
            r'javascript:',
            r'on\w+\s*=',
            r'eval\s*\(',
            r'setTimeout\s*\(',
            r'setInterval\s*\('
        ]

        for pattern in script_patterns:
            if re.search(pattern, self.user_input, re.IGNORECASE):
                return True
        return False

    def _detect_pii(self) -> List[str]:
        """Detect potential PII in the input."""
        pii_patterns = {
            'email': r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',
            'phone': r'\b(?:\+?1[-.\s]?)?\(?([0-9]{3})\)?[-.\s]?([0-9]{3})[-.\s]?([0-9]{4})\b',
            'ssn': r'\b(?:\d{3}-?\d{2}-?\d{4}|\d{9})\b',
            'credit_card': r'\b(?:\d{4}[-\s]?){3}\d{4}\b'
        }

        detected = []
        for pii_type, pattern in pii_patterns.items():
            if re.search(pattern, self.user_input):
                detected.append(pii_type)

        return detected

    def _contains_profanity(self) -> bool:
        """Check for profanity (simplified implementation)."""
        # This would normally use a comprehensive profanity filter
        profanity_words = ['damn', 'shit', 'fuck', 'bitch']  # Simplified list
        text_lower = self.user_input.lower()
        return any(word in text_lower for word in profanity_words)

    def _detect_injection_attempts(self) -> bool:
        """Detect potential injection attacks."""
        injection_patterns = [
            r'union\s+select',
            r'drop\s+table',
            r'delete\s+from',
            r'insert\s+into',
            r'update\s+.*set',
            r'exec\s*\(',
            r'xp_cmdshell',
            r'sp_executesql'
        ]

        for pattern in injection_patterns:
            if re.search(pattern, self.user_input, re.IGNORECASE):
                return True
        return False


# ============================================================================
# DATA INTEGRITY VALIDATION SCHEMAS
# ============================================================================

class ConversationIntegrityValidation(BaseRequestModel):
    """Validation schema for conversation data integrity."""

    conversation_id: str
    tenant_id: str
    user_id: str
    messages: List[Dict[str, Any]]

    # Integrity rules
    require_sequential_numbers: bool = Field(default=True)
    require_alternating_directions: bool = Field(default=False)
    max_message_gap_minutes: int = Field(default=1440)  # 24 hours

    def validate_integrity(self) -> ValidationResult:
        """Validate conversation data integrity."""
        result = ValidationResult(is_valid=True)

        if not self.messages:
            result.add_warning("Conversation has no messages")
            return result

        # Sort messages by sequence number
        sorted_messages = sorted(self.messages, key=lambda x: x.get('sequence_number', 0))

        # Check sequential numbering
        if self.require_sequential_numbers:
            expected_seq = 1
            for msg in sorted_messages:
                actual_seq = msg.get('sequence_number', 0)
                if actual_seq != expected_seq:
                    result.add_error(f"Message sequence gap: expected {expected_seq}, got {actual_seq}")
                expected_seq += 1

        # Check message timestamps
        prev_timestamp = None
        for msg in sorted_messages:
            timestamp_str = msg.get('timestamp')
            if timestamp_str:
                try:
                    timestamp = datetime.fromisoformat(timestamp_str.replace('Z', '+00:00'))

                    if prev_timestamp:
                        gap = timestamp - prev_timestamp
                        if gap.total_seconds() > self.max_message_gap_minutes * 60:
                            result.add_warning(f"Large time gap between messages: {gap}")

                    prev_timestamp = timestamp
                except ValueError:
                    result.add_error(f"Invalid timestamp format: {timestamp_str}")

        # Check alternating directions (if required)
        if self.require_alternating_directions and len(sorted_messages) > 1:
            prev_direction = None
            for msg in sorted_messages:
                direction = msg.get('direction')
                if prev_direction and direction == prev_direction:
                    result.add_warning(f"Non-alternating message directions detected")
                    break
                prev_direction = direction

        return result


class SessionIntegrityValidation(BaseRequestModel):
    """Validation schema for session data integrity."""

    session_id: str
    tenant_id: str
    user_id: str
    started_at: datetime
    last_activity_at: datetime
    conversations: List[Dict[str, Any]] = Field(default_factory=list)

    def validate_integrity(self) -> ValidationResult:
        """Validate session data integrity."""
        result = ValidationResult(is_valid=True)

        # Check timestamp consistency
        if self.last_activity_at < self.started_at:
            result.add_error("Last activity cannot be before session start")

        # Check session duration
        duration = self.last_activity_at - self.started_at
        if duration.total_seconds() > 86400 * 7:  # 7 days
            result.add_warning(f"Unusually long session duration: {duration}")

        # Validate conversations
        for conv in self.conversations:
            conv_start = conv.get('started_at')
            if conv_start:
                try:
                    conv_timestamp = datetime.fromisoformat(conv_start.replace('Z', '+00:00'))
                    if conv_timestamp < self.started_at:
                        result.add_error(f"Conversation started before session: {conv.get('conversation_id')}")
                except ValueError:
                    result.add_error(f"Invalid conversation timestamp: {conv_start}")

        return result


# ============================================================================
# CUSTOM VALIDATORS
# ============================================================================

class CustomValidator:
    """Base class for custom validators."""

    def __init__(self, name: str, description: str):
        self.name = name
        self.description = description

    def validate(self, value: Any, context: Dict[str, Any] = None) -> ValidationResult:
        """Perform validation - to be implemented by subclasses."""
        raise NotImplementedError


class RegexValidator(CustomValidator):
    """Validator using regular expressions."""

    def __init__(self, name: str, pattern: str, description: str, flags: int = 0):
        super().__init__(name, description)
        self.pattern = re.compile(pattern, flags)

    def validate(self, value: Any, context: Dict[str, Any] = None) -> ValidationResult:
        result = ValidationResult(is_valid=True)

        if not isinstance(value, str):
            result.add_error(f"Value must be a string for regex validation")
            return result

        if not self.pattern.match(value):
            result.add_error(f"Value does not match required pattern: {self.description}")

        return result


class LengthValidator(CustomValidator):
    """Validator for string/list length."""

    def __init__(self, name: str, min_length: int = None, max_length: int = None):
        description = f"Length validation (min: {min_length}, max: {max_length})"
        super().__init__(name, description)
        self.min_length = min_length
        self.max_length = max_length

    def validate(self, value: Any, context: Dict[str, Any] = None) -> ValidationResult:
        result = ValidationResult(is_valid=True)

        try:
            length = len(value)
        except TypeError:
            result.add_error("Value must have a length")
            return result

        if self.min_length is not None and length < self.min_length:
            result.add_error(f"Length {length} is below minimum {self.min_length}")

        if self.max_length is not None and length > self.max_length:
            result.add_error(f"Length {length} exceeds maximum {self.max_length}")

        return result


class RangeValidator(CustomValidator):
    """Validator for numeric ranges."""

    def __init__(self, name: str, min_value: Union[int, float] = None, max_value: Union[int, float] = None):
        description = f"Range validation (min: {min_value}, max: {max_value})"
        super().__init__(name, description)
        self.min_value = min_value
        self.max_value = max_value

    def validate(self, value: Any, context: Dict[str, Any] = None) -> ValidationResult:
        result = ValidationResult(is_valid=True)

        if not isinstance(value, (int, float)):
            result.add_error("Value must be numeric for range validation")
            return result

        if self.min_value is not None and value < self.min_value:
            result.add_error(f"Value {value} is below minimum {self.min_value}")

        if self.max_value is not None and value > self.max_value:
            result.add_error(f"Value {value} exceeds maximum {self.max_value}")

        return result


# ============================================================================
# VALIDATION RULE SETS
# ============================================================================

class ValidationRuleSet:
    """Collection of validation rules for a specific use case."""

    def __init__(self, name: str, description: str):
        self.name = name
        self.description = description
        self.validators: List[CustomValidator] = []
        self.required_fields: List[str] = []
        self.optional_fields: List[str] = []

    def add_validator(self, validator: CustomValidator, field: str = None) -> None:
        """Add a validator to the rule set."""
        validator.field = field
        self.validators.append(validator)

    def add_required_field(self, field: str) -> None:
        """Add a required field."""
        if field not in self.required_fields:
            self.required_fields.append(field)

    def validate(self, data: Dict[str, Any]) -> ValidationResult:
        """Validate data against all rules in the set."""
        result = ValidationResult(is_valid=True)

        # Check required fields
        for field in self.required_fields:
            if field not in data or data[field] is None:
                result.add_error(f"Required field missing: {field}")

        # Apply validators
        for validator in self.validators:
            field = getattr(validator, 'field', None)
            if field and field in data:
                field_result = validator.validate(data[field], data)
                if field_result.has_errors():
                    for error in field_result.errors:
                        result.add_error(f"{field}: {error}")
                if field_result.has_warnings():
                    for warning in field_result.warnings:
                        result.add_warning(f"{field}: {warning}")

        return result


# ============================================================================
# PREDEFINED VALIDATION RULE SETS
# ============================================================================

# User ID validation
USER_ID_RULES = ValidationRuleSet("user_id_validation", "Validation rules for user IDs")
USER_ID_RULES.add_validator(LengthValidator("user_id_length", min_length=1, max_length=255), "user_id")
USER_ID_RULES.add_validator(
    RegexValidator("user_id_format", r'^[a-zA-Z0-9_-]+$', "Alphanumeric with underscores and hyphens"), "user_id")
USER_ID_RULES.add_required_field("user_id")

# Email validation
EMAIL_RULES = ValidationRuleSet("email_validation", "Validation rules for email addresses")
EMAIL_RULES.add_validator(
    RegexValidator("email_format", r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$', "Valid email format"), "email")
EMAIL_RULES.add_validator(LengthValidator("email_length", max_length=320), "email")

# Phone number validation
PHONE_RULES = ValidationRuleSet("phone_validation", "Validation rules for phone numbers")
PHONE_RULES.add_validator(RegexValidator("phone_format", r'^\+[1-9]\d{1,14}$', "E.164 format"), "phone")

# Export all validation schemas and utilities
__all__ = [
    # Result classes
    "ValidationResult",
    "FieldValidationResult",

    # Business rule validation
    "MessageContentValidation",
    "RateLimitValidation",

    # Security validation
    "SecurityValidation",

    # Data integrity validation
    "ConversationIntegrityValidation",
    "SessionIntegrityValidation",

    # Custom validators
    "CustomValidator",
    "RegexValidator",
    "LengthValidator",
    "RangeValidator",

    # Rule sets
    "ValidationRuleSet",
    "USER_ID_RULES",
    "EMAIL_RULES",
    "PHONE_RULES"
]