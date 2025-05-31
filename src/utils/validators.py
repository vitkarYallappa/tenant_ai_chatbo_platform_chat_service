"""
Validation utilities and custom validators.

This module provides comprehensive validation functions for various
data types, formats, and business rules used throughout the Chat Service.
"""

import re
import uuid
from typing import Any, Dict, List, Optional, Union, Callable
from datetime import datetime, timezone
from urllib.parse import urlparse
import ipaddress

import phonenumbers
from email_validator import validate_email, EmailNotValidError
from pydantic import BaseModel, validator, ValidationError

import structlog
from src.config.constants import (
    VALIDATION_PATTERNS,
    SUPPORTED_CHANNELS,
    MESSAGE_TYPES,
    SUPPORTED_MEDIA_TYPES,
    MAX_MESSAGE_SIZE,
    MAX_MEDIA_SIZE,
    SUPPORTED_LANGUAGES,
)
from src.utils.logger import get_logger

# Initialize logger
logger = get_logger(__name__)


class ValidationError(Exception):
    """Custom validation error with detailed information."""

    def __init__(
            self,
            message: str,
            field: Optional[str] = None,
            value: Optional[Any] = None,
            code: Optional[str] = None
    ):
        super().__init__(message)
        self.message = message
        self.field = field
        self.value = value
        self.code = code or "VALIDATION_ERROR"


class DataValidator:
    """
    Comprehensive data validation utilities.

    Provides various validation methods for different data types
    and business rules specific to the Chat Service.
    """

    @staticmethod
    def validate_email_address(email: str) -> str:
        """
        Validate email address format.

        Args:
            email: Email address to validate

        Returns:
            Normalized email address

        Raises:
            ValidationError: If email is invalid
        """
        if not email or not isinstance(email, str):
            raise ValidationError("Email address is required", field="email", value=email)

        try:
            # Use email-validator library for comprehensive validation
            validation = validate_email(email.strip())
            return validation.email

        except EmailNotValidError as e:
            raise ValidationError(
                f"Invalid email address: {str(e)}",
                field="email",
                value=email,
                code="INVALID_EMAIL"
            )

    @staticmethod
    def validate_phone_number(
            phone: str,
            country_code: Optional[str] = None
    ) -> str:
        """
        Validate phone number format.

        Args:
            phone: Phone number to validate
            country_code: Optional country code (e.g., 'US')

        Returns:
            Formatted phone number in E.164 format

        Raises:
            ValidationError: If phone number is invalid
        """
        if not phone or not isinstance(phone, str):
            raise ValidationError("Phone number is required", field="phone", value=phone)

        try:
            # Parse phone number
            parsed_number = phonenumbers.parse(phone, country_code)

            # Validate the number
            if not phonenumbers.is_valid_number(parsed_number):
                raise ValidationError(
                    "Invalid phone number",
                    field="phone",
                    value=phone,
                    code="INVALID_PHONE"
                )

            # Return in E.164 format
            return phonenumbers.format_number(parsed_number, phonenumbers.PhoneNumberFormat.E164)

        except phonenumbers.NumberParseException as e:
            raise ValidationError(
                f"Phone number parsing failed: {str(e)}",
                field="phone",
                value=phone,
                code="INVALID_PHONE_FORMAT"
            )

    @staticmethod
    def validate_uuid(value: str, field_name: str = "id") -> str:
        """
        Validate UUID format.

        Args:
            value: UUID string to validate
            field_name: Name of the field for error reporting

        Returns:
            Validated UUID string

        Raises:
            ValidationError: If UUID is invalid
        """
        if not value or not isinstance(value, str):
            raise ValidationError(
                f"{field_name} is required",
                field=field_name,
                value=value
            )

        try:
            # Validate UUID format
            uuid_obj = uuid.UUID(value)
            return str(uuid_obj)

        except ValueError:
            raise ValidationError(
                f"Invalid {field_name} format",
                field=field_name,
                value=value,
                code="INVALID_UUID"
            )

    @staticmethod
    def validate_url(url: str, allowed_schemes: Optional[List[str]] = None) -> str:
        """
        Validate URL format and scheme.

        Args:
            url: URL to validate
            allowed_schemes: List of allowed URL schemes

        Returns:
            Validated URL

        Raises:
            ValidationError: If URL is invalid
        """
        if not url or not isinstance(url, str):
            raise ValidationError("URL is required", field="url", value=url)

        allowed_schemes = allowed_schemes or ["http", "https"]

        try:
            parsed = urlparse(url.strip())

            if not parsed.scheme:
                raise ValidationError(
                    "URL must include scheme (http/https)",
                    field="url",
                    value=url,
                    code="MISSING_URL_SCHEME"
                )

            if parsed.scheme not in allowed_schemes:
                raise ValidationError(
                    f"URL scheme must be one of: {', '.join(allowed_schemes)}",
                    field="url",
                    value=url,
                    code="INVALID_URL_SCHEME"
                )

            if not parsed.netloc:
                raise ValidationError(
                    "URL must include domain",
                    field="url",
                    value=url,
                    code="MISSING_URL_DOMAIN"
                )

            return url.strip()

        except Exception as e:
            if isinstance(e, ValidationError):
                raise

            raise ValidationError(
                f"Invalid URL format: {str(e)}",
                field="url",
                value=url,
                code="INVALID_URL"
            )

    @staticmethod
    def validate_ip_address(ip: str) -> str:
        """
        Validate IP address format.

        Args:
            ip: IP address to validate

        Returns:
            Validated IP address

        Raises:
            ValidationError: If IP address is invalid
        """
        if not ip or not isinstance(ip, str):
            raise ValidationError("IP address is required", field="ip", value=ip)

        try:
            # Validate IPv4 or IPv6 address
            ipaddress.ip_address(ip.strip())
            return ip.strip()

        except ValueError:
            raise ValidationError(
                "Invalid IP address format",
                field="ip",
                value=ip,
                code="INVALID_IP"
            )

    @staticmethod
    def validate_text_length(
            text: str,
            min_length: int = 0,
            max_length: int = MAX_MESSAGE_SIZE,
            field_name: str = "text"
    ) -> str:
        """
        Validate text length constraints.

        Args:
            text: Text to validate
            min_length: Minimum allowed length
            max_length: Maximum allowed length
            field_name: Field name for error reporting

        Returns:
            Validated text

        Raises:
            ValidationError: If text length is invalid
        """
        if text is None:
            if min_length > 0:
                raise ValidationError(
                    f"{field_name} is required",
                    field=field_name,
                    value=text
                )
            return ""

        if not isinstance(text, str):
            raise ValidationError(
                f"{field_name} must be a string",
                field=field_name,
                value=text
            )

        text_length = len(text)

        if text_length < min_length:
            raise ValidationError(
                f"{field_name} must be at least {min_length} characters",
                field=field_name,
                value=text,
                code="TEXT_TOO_SHORT"
            )

        if text_length > max_length:
            raise ValidationError(
                f"{field_name} must not exceed {max_length} characters",
                field=field_name,
                value=text,
                code="TEXT_TOO_LONG"
            )

        return text

    @staticmethod
    def validate_channel_type(channel: str) -> str:
        """
        Validate channel type.

        Args:
            channel: Channel type to validate

        Returns:
            Validated channel type

        Raises:
            ValidationError: If channel type is invalid
        """
        if not channel or not isinstance(channel, str):
            raise ValidationError("Channel type is required", field="channel", value=channel)

        channel = channel.lower().strip()

        if channel not in SUPPORTED_CHANNELS:
            raise ValidationError(
                f"Unsupported channel type. Must be one of: {', '.join(SUPPORTED_CHANNELS)}",
                field="channel",
                value=channel,
                code="INVALID_CHANNEL"
            )

        return channel

    @staticmethod
    def validate_message_type(message_type: str) -> str:
        """
        Validate message type.

        Args:
            message_type: Message type to validate

        Returns:
            Validated message type

        Raises:
            ValidationError: If message type is invalid
        """
        if not message_type or not isinstance(message_type, str):
            raise ValidationError("Message type is required", field="message_type", value=message_type)

        message_type = message_type.lower().strip()

        if message_type not in MESSAGE_TYPES:
            raise ValidationError(
                f"Unsupported message type. Must be one of: {', '.join(MESSAGE_TYPES)}",
                field="message_type",
                value=message_type,
                code="INVALID_MESSAGE_TYPE"
            )

        return message_type

    @staticmethod
    def validate_language_code(language: str) -> str:
        """
        Validate language code.

        Args:
            language: Language code to validate (ISO 639-1)

        Returns:
            Validated language code

        Raises:
            ValidationError: If language code is invalid
        """
        if not language or not isinstance(language, str):
            raise ValidationError("Language code is required", field="language", value=language)

        language = language.lower().strip()

        if language not in SUPPORTED_LANGUAGES:
            raise ValidationError(
                f"Unsupported language code. Must be one of: {', '.join(SUPPORTED_LANGUAGES.keys())}",
                field="language",
                value=language,
                code="INVALID_LANGUAGE"
            )

        return language

    @staticmethod
    def validate_mime_type(mime_type: str) -> str:
        """
        Validate MIME type for media files.

        Args:
            mime_type: MIME type to validate

        Returns:
            Validated MIME type

        Raises:
            ValidationError: If MIME type is not supported
        """
        if not mime_type or not isinstance(mime_type, str):
            raise ValidationError("MIME type is required", field="mime_type", value=mime_type)

        mime_type = mime_type.lower().strip()

        if mime_type not in SUPPORTED_MEDIA_TYPES:
            raise ValidationError(
                "Unsupported media type",
                field="mime_type",
                value=mime_type,
                code="INVALID_MIME_TYPE"
            )

        return mime_type

    @staticmethod
    def validate_file_size(size_bytes: int, max_size: int = MAX_MEDIA_SIZE) -> int:
        """
        Validate file size constraints.

        Args:
            size_bytes: File size in bytes
            max_size: Maximum allowed size in bytes

        Returns:
            Validated file size

        Raises:
            ValidationError: If file size is invalid
        """
        if not isinstance(size_bytes, int) or size_bytes < 0:
            raise ValidationError(
                "File size must be a non-negative integer",
                field="size_bytes",
                value=size_bytes
            )

        if size_bytes == 0:
            raise ValidationError(
                "File cannot be empty",
                field="size_bytes",
                value=size_bytes,
                code="EMPTY_FILE"
            )

        if size_bytes > max_size:
            max_mb = max_size / (1024 * 1024)
            raise ValidationError(
                f"File size exceeds maximum of {max_mb:.1f} MB",
                field="size_bytes",
                value=size_bytes,
                code="FILE_TOO_LARGE"
            )

        return size_bytes

    @staticmethod
    def validate_coordinates(latitude: float, longitude: float) -> tuple[float, float]:
        """
        Validate geographical coordinates.

        Args:
            latitude: Latitude value
            longitude: Longitude value

        Returns:
            Tuple of validated (latitude, longitude)

        Raises:
            ValidationError: If coordinates are invalid
        """
        if not isinstance(latitude, (int, float)):
            raise ValidationError(
                "Latitude must be a number",
                field="latitude",
                value=latitude
            )

        if not isinstance(longitude, (int, float)):
            raise ValidationError(
                "Longitude must be a number",
                field="longitude",
                value=longitude
            )

        if not (-90 <= latitude <= 90):
            raise ValidationError(
                "Latitude must be between -90 and 90 degrees",
                field="latitude",
                value=latitude,
                code="INVALID_LATITUDE"
            )

        if not (-180 <= longitude <= 180):
            raise ValidationError(
                "Longitude must be between -180 and 180 degrees",
                field="longitude",
                value=longitude,
                code="INVALID_LONGITUDE"
            )

        return (float(latitude), float(longitude))

    @staticmethod
    def validate_datetime(
            value: Union[str, datetime],
            field_name: str = "datetime"
    ) -> datetime:
        """
        Validate and parse datetime value.

        Args:
            value: Datetime string or datetime object
            field_name: Field name for error reporting

        Returns:
            Validated datetime object

        Raises:
            ValidationError: If datetime is invalid
        """
        if isinstance(value, datetime):
            return value

        if not isinstance(value, str):
            raise ValidationError(
                f"{field_name} must be a datetime string or datetime object",
                field=field_name,
                value=value
            )

        try:
            # Try parsing ISO format
            if 'T' in value:
                return datetime.fromisoformat(value.replace('Z', '+00:00'))
            else:
                return datetime.fromisoformat(value)

        except ValueError:
            raise ValidationError(
                f"Invalid {field_name} format. Use ISO 8601 format",
                field=field_name,
                value=value,
                code="INVALID_DATETIME"
            )

    @staticmethod
    def validate_regex_pattern(
            value: str,
            pattern: str,
            field_name: str,
            error_message: Optional[str] = None
    ) -> str:
        """
        Validate value against regex pattern.

        Args:
            value: Value to validate
            pattern: Regex pattern
            field_name: Field name for error reporting
            error_message: Custom error message

        Returns:
            Validated value

        Raises:
            ValidationError: If value doesn't match pattern
        """
        if not value or not isinstance(value, str):
            raise ValidationError(
                f"{field_name} is required",
                field=field_name,
                value=value
            )

        if not re.match(pattern, value):
            message = error_message or f"Invalid {field_name} format"
            raise ValidationError(
                message,
                field=field_name,
                value=value,
                code="INVALID_FORMAT"
            )

        return value


class BusinessRuleValidator:
    """
    Business rule validation for Chat Service specific logic.
    """

    @staticmethod
    def validate_tenant_subdomain(subdomain: str) -> str:
        """
        Validate tenant subdomain format.

        Args:
            subdomain: Subdomain to validate

        Returns:
            Validated subdomain

        Raises:
            ValidationError: If subdomain is invalid
        """
        return DataValidator.validate_regex_pattern(
            subdomain,
            VALIDATION_PATTERNS["tenant_subdomain"],
            "subdomain",
            "Subdomain must be 3-63 characters, lowercase letters, numbers, and hyphens only"
        )

    @staticmethod
    def validate_username(username: str) -> str:
        """
        Validate username format.

        Args:
            username: Username to validate

        Returns:
            Validated username

        Raises:
            ValidationError: If username is invalid
        """
        return DataValidator.validate_regex_pattern(
            username,
            VALIDATION_PATTERNS["username"],
            "username",
            "Username must be 3-30 characters, letters, numbers, underscores, and hyphens only"
        )

    @staticmethod
    def validate_api_key_format(api_key: str) -> str:
        """
        Validate API key format.

        Args:
            api_key: API key to validate

        Returns:
            Validated API key

        Raises:
            ValidationError: If API key format is invalid
        """
        return DataValidator.validate_regex_pattern(
            api_key,
            VALIDATION_PATTERNS["api_key"],
            "api_key",
            "API key must follow format: cb_{env}_{32_hex_chars}"
        )

    @staticmethod
    def validate_message_content(
            content: Dict[str, Any],
            channel: str
    ) -> Dict[str, Any]:
        """
        Validate message content based on channel capabilities.

        Args:
            content: Message content dictionary
            channel: Channel type

        Returns:
            Validated content dictionary

        Raises:
            ValidationError: If content is invalid for channel
        """
        from src.config.constants import CHANNEL_CONFIG

        # Validate channel first
        channel = DataValidator.validate_channel_type(channel)

        if not isinstance(content, dict):
            raise ValidationError(
                "Message content must be a dictionary",
                field="content",
                value=content
            )

        message_type = content.get("type")
        if not message_type:
            raise ValidationError(
                "Message type is required in content",
                field="content.type",
                value=message_type
            )

        # Validate message type
        message_type = DataValidator.validate_message_type(message_type)

        # Get channel configuration
        channel_config = CHANNEL_CONFIG.get(channel, {})

        # Validate text content
        if message_type == "text":
            text = content.get("text", "")
            max_length = channel_config.get("max_message_length", MAX_MESSAGE_SIZE)

            DataValidator.validate_text_length(
                text,
                min_length=1,
                max_length=max_length,
                field_name="content.text"
            )

        # Validate media content
        elif message_type in ["image", "audio", "video", "file"]:
            if not channel_config.get("supports_media", False):
                raise ValidationError(
                    f"Channel '{channel}' does not support media messages",
                    field="content.type",
                    value=message_type,
                    code="CHANNEL_NOT_SUPPORT_MEDIA"
                )

            media = content.get("media")
            if not media:
                raise ValidationError(
                    "Media content is required for media messages",
                    field="content.media",
                    value=media
                )

            # Validate media URL and type
            if "url" in media:
                DataValidator.validate_url(media["url"])

            if "type" in media:
                DataValidator.validate_mime_type(media["type"])

        # Validate location content
        elif message_type == "location":
            if not channel_config.get("supports_location", False):
                raise ValidationError(
                    f"Channel '{channel}' does not support location messages",
                    field="content.type",
                    value=message_type,
                    code="CHANNEL_NOT_SUPPORT_LOCATION"
                )

            location = content.get("location")
            if not location:
                raise ValidationError(
                    "Location content is required for location messages",
                    field="content.location",
                    value=location
                )

            if "latitude" in location and "longitude" in location:
                DataValidator.validate_coordinates(
                    location["latitude"],
                    location["longitude"]
                )

        # Validate quick replies
        if "quick_replies" in content:
            if not channel_config.get("supports_quick_replies", False):
                raise ValidationError(
                    f"Channel '{channel}' does not support quick replies",
                    field="content.quick_replies",
                    code="CHANNEL_NOT_SUPPORT_QUICK_REPLIES"
                )

            quick_replies = content["quick_replies"]
            max_quick_replies = channel_config.get("max_quick_replies", 10)

            if len(quick_replies) > max_quick_replies:
                raise ValidationError(
                    f"Channel '{channel}' supports maximum {max_quick_replies} quick replies",
                    field="content.quick_replies",
                    value=len(quick_replies),
                    code="TOO_MANY_QUICK_REPLIES"
                )

        return content


def validate_request_data(
        data: Dict[str, Any],
        validation_rules: Dict[str, Callable]
) -> Dict[str, Any]:
    """
    Validate request data using provided validation rules.

    Args:
        data: Data to validate
        validation_rules: Dictionary mapping field names to validation functions

    Returns:
        Validated data dictionary

    Raises:
        ValidationError: If any validation fails
    """
    validated_data = {}
    errors = []

    for field, validator_func in validation_rules.items():
        try:
            if field in data:
                validated_data[field] = validator_func(data[field])
            elif hasattr(validator_func, "__annotations__"):
                # Check if field is required (no default value)
                import inspect
                sig = inspect.signature(validator_func)
                params = list(sig.parameters.values())
                if params and params[0].default == inspect.Parameter.empty:
                    errors.append(f"Required field '{field}' is missing")

        except ValidationError as e:
            errors.append(f"{field}: {e.message}")
        except Exception as e:
            errors.append(f"{field}: Validation error - {str(e)}")

    if errors:
        raise ValidationError(
            f"Validation failed: {'; '.join(errors)}",
            code="MULTIPLE_VALIDATION_ERRORS"
        )

    return validated_data


# Export commonly used validators and classes
__all__ = [
    'ValidationError',
    'DataValidator',
    'BusinessRuleValidator',
    'validate_request_data',
]