"""
ID generation utilities for Chat Service.

This module provides various ID generation functions including UUIDs,
short IDs, sequential IDs, and other identifier types used throughout
the Chat Service application.
"""

import secrets
import string
import time
import uuid
from typing import Optional, List, Dict, Any
from datetime import datetime, timezone
from hashlib import sha256
import base64

import structlog
from src.utils.logger import get_logger
from src.config.constants import SERVICE_NAME

# Initialize logger
logger = get_logger(__name__)

# Character sets for different ID types
ALPHANUMERIC = string.ascii_letters + string.digits
ALPHANUMERIC_LOWERCASE = string.ascii_lowercase + string.digits
ALPHANUMERIC_UPPERCASE = string.ascii_uppercase + string.digits
NUMERIC = string.digits
HEX_CHARS = '0123456789abcdef'


class IDGenerator:
    """
    Comprehensive ID generation utilities.

    Provides various types of ID generation including UUIDs,
    short IDs, prefixed IDs, and time-based identifiers.
    """

    @staticmethod
    def generate_uuid() -> str:
        """
        Generate a standard UUID4.

        Returns:
            UUID4 string
        """
        return str(uuid.uuid4())

    @staticmethod
    def generate_uuid_hex() -> str:
        """
        Generate UUID4 as hex string (no hyphens).

        Returns:
            UUID4 hex string
        """
        return uuid.uuid4().hex

    @staticmethod
    def generate_short_uuid() -> str:
        """
        Generate a shortened UUID using base64 encoding.

        Returns:
            Shortened UUID string (22 characters)
        """
        uuid_bytes = uuid.uuid4().bytes
        # Use URL-safe base64 encoding and remove padding
        short_id = base64.urlsafe_b64encode(uuid_bytes).decode('ascii').rstrip('=')
        return short_id

    @staticmethod
    def generate_random_string(
            length: int = 8,
            charset: str = ALPHANUMERIC
    ) -> str:
        """
        Generate a random string with specified length and character set.

        Args:
            length: Length of the string
            charset: Character set to use

        Returns:
            Random string
        """
        return ''.join(secrets.choice(charset) for _ in range(length))

    @staticmethod
    def generate_numeric_id(length: int = 8) -> str:
        """
        Generate a numeric ID with specified length.

        Args:
            length: Length of the numeric ID

        Returns:
            Numeric ID string
        """
        # Ensure first digit is not 0
        first_digit = secrets.choice('123456789')
        remaining_digits = ''.join(secrets.choice(NUMERIC) for _ in range(length - 1))
        return first_digit + remaining_digits

    @staticmethod
    def generate_hex_id(length: int = 16) -> str:
        """
        Generate a hexadecimal ID with specified length.

        Args:
            length: Length of the hex ID

        Returns:
            Hexadecimal ID string
        """
        return ''.join(secrets.choice(HEX_CHARS) for _ in range(length))

    @staticmethod
    def generate_prefixed_id(
            prefix: str,
            separator: str = "_",
            id_length: int = 8,
            charset: str = ALPHANUMERIC_LOWERCASE
    ) -> str:
        """
        Generate an ID with a prefix.

        Args:
            prefix: Prefix to add
            separator: Separator between prefix and ID
            id_length: Length of the random part
            charset: Character set for the random part

        Returns:
            Prefixed ID string
        """
        random_part = IDGenerator.generate_random_string(id_length, charset)
        return f"{prefix}{separator}{random_part}"

    @staticmethod
    def generate_time_based_id(
            prefix: Optional[str] = None,
            include_random: bool = True,
            random_length: int = 4
    ) -> str:
        """
        Generate a time-based ID with optional prefix and random suffix.

        Args:
            prefix: Optional prefix
            include_random: Whether to include random suffix
            random_length: Length of random suffix

        Returns:
            Time-based ID string
        """
        # Use milliseconds since epoch
        timestamp = int(time.time() * 1000)

        parts = []
        if prefix:
            parts.append(prefix)

        parts.append(str(timestamp))

        if include_random:
            random_suffix = IDGenerator.generate_random_string(
                random_length,
                ALPHANUMERIC_LOWERCASE
            )
            parts.append(random_suffix)

        return "_".join(parts)

    @staticmethod
    def generate_sequential_id(
            prefix: str,
            sequence_number: int,
            padding: int = 6
    ) -> str:
        """
        Generate a sequential ID with prefix and zero-padded number.

        Args:
            prefix: Prefix for the ID
            sequence_number: Sequential number
            padding: Zero padding length

        Returns:
            Sequential ID string
        """
        padded_number = str(sequence_number).zfill(padding)
        return f"{prefix}_{padded_number}"

    @staticmethod
    def generate_checksum_id(
            base_string: str,
            checksum_length: int = 4
    ) -> str:
        """
        Generate an ID with checksum for validation.

        Args:
            base_string: Base string to generate checksum from
            checksum_length: Length of checksum

        Returns:
            ID with checksum
        """
        # Generate SHA256 hash of base string
        hash_obj = sha256(base_string.encode('utf-8'))
        checksum = hash_obj.hexdigest()[:checksum_length]

        return f"{base_string}_{checksum}"


class ConversationIDGenerator:
    """
    Specialized ID generator for conversation-related identifiers.
    """

    @staticmethod
    def generate_conversation_id(tenant_id: str) -> str:
        """
        Generate a conversation ID.

        Args:
            tenant_id: Tenant identifier

        Returns:
            Conversation ID
        """
        return IDGenerator.generate_prefixed_id("conv", id_length=12)

    @staticmethod
    def generate_message_id(conversation_id: str) -> str:
        """
        Generate a message ID.

        Args:
            conversation_id: Parent conversation ID

        Returns:
            Message ID
        """
        return IDGenerator.generate_prefixed_id("msg", id_length=10)

    @staticmethod
    def generate_session_id(user_id: str, channel: str) -> str:
        """
        Generate a session ID.

        Args:
            user_id: User identifier
            channel: Channel type

        Returns:
            Session ID
        """
        return IDGenerator.generate_prefixed_id("sess", id_length=8)

    @staticmethod
    def generate_attachment_id(message_id: str) -> str:
        """
        Generate an attachment ID.

        Args:
            message_id: Parent message ID

        Returns:
            Attachment ID
        """
        return IDGenerator.generate_prefixed_id("att", id_length=8)


class BusinessIDGenerator:
    """
    Business-specific ID generators for various entities.
    """

    @staticmethod
    def generate_tenant_id() -> str:
        """
        Generate a tenant ID.

        Returns:
            Tenant ID (UUID format)
        """
        return IDGenerator.generate_uuid()

    @staticmethod
    def generate_user_id(tenant_id: str) -> str:
        """
        Generate a user ID.

        Args:
            tenant_id: Tenant identifier

        Returns:
            User ID
        """
        return IDGenerator.generate_prefixed_id("user", id_length=8)

    @staticmethod
    def generate_api_key_id() -> str:
        """
        Generate an API key ID.

        Returns:
            API key ID
        """
        return IDGenerator.generate_prefixed_id("key", id_length=8)

    @staticmethod
    def generate_webhook_id() -> str:
        """
        Generate a webhook ID.

        Returns:
            Webhook ID
        """
        return IDGenerator.generate_prefixed_id("hook", id_length=8)

    @staticmethod
    def generate_integration_id() -> str:
        """
        Generate an integration ID.

        Returns:
            Integration ID
        """
        return IDGenerator.generate_prefixed_id("int", id_length=8)

    @staticmethod
    def generate_flow_id() -> str:
        """
        Generate a conversation flow ID.

        Returns:
            Flow ID
        """
        return IDGenerator.generate_prefixed_id("flow", id_length=8)


class TrackingIDGenerator:
    """
    Tracking and monitoring ID generators.
    """

    @staticmethod
    def generate_request_id() -> str:
        """
        Generate a request tracking ID.

        Returns:
            Request ID
        """
        return IDGenerator.generate_prefixed_id("req", id_length=12)

    @staticmethod
    def generate_correlation_id() -> str:
        """
        Generate a correlation ID for distributed tracing.

        Returns:
            Correlation ID
        """
        return IDGenerator.generate_prefixed_id("corr", id_length=16)

    @staticmethod
    def generate_trace_id() -> str:
        """
        Generate a trace ID for monitoring.

        Returns:
            Trace ID
        """
        return IDGenerator.generate_hex_id(32)  # 128-bit trace ID

    @staticmethod
    def generate_span_id() -> str:
        """
        Generate a span ID for distributed tracing.

        Returns:
            Span ID
        """
        return IDGenerator.generate_hex_id(16)  # 64-bit span ID

    @staticmethod
    def generate_transaction_id() -> str:
        """
        Generate a transaction ID.

        Returns:
            Transaction ID
        """
        return IDGenerator.generate_time_based_id("txn", include_random=True)


class FileIDGenerator:
    """
    File and media-specific ID generators.
    """

    @staticmethod
    def generate_file_id(filename: str) -> str:
        """
        Generate a file ID based on filename.

        Args:
            filename: Original filename

        Returns:
            File ID
        """
        # Create a deterministic but unique ID based on filename and timestamp
        timestamp = int(time.time() * 1000)
        base_string = f"{filename}_{timestamp}"
        return IDGenerator.generate_checksum_id(base_string, 8)

    @staticmethod
    def generate_media_id(media_type: str) -> str:
        """
        Generate a media ID.

        Args:
            media_type: Type of media (image, audio, video, etc.)

        Returns:
            Media ID
        """
        return IDGenerator.generate_prefixed_id(media_type[:3], id_length=10)

    @staticmethod
    def generate_upload_id() -> str:
        """
        Generate an upload session ID.

        Returns:
            Upload ID
        """
        return IDGenerator.generate_time_based_id("upload", include_random=True)


class IDValidator:
    """
    ID validation utilities.
    """

    @staticmethod
    def is_valid_uuid(id_string: str) -> bool:
        """
        Validate UUID format.

        Args:
            id_string: String to validate

        Returns:
            True if valid UUID
        """
        try:
            uuid.UUID(id_string)
            return True
        except ValueError:
            return False

    @staticmethod
    def is_valid_prefixed_id(id_string: str, expected_prefix: str) -> bool:
        """
        Validate prefixed ID format.

        Args:
            id_string: ID to validate
            expected_prefix: Expected prefix

        Returns:
            True if valid prefixed ID
        """
        if not id_string:
            return False

        parts = id_string.split('_', 1)
        if len(parts) != 2:
            return False

        prefix, suffix = parts
        return prefix == expected_prefix and len(suffix) > 0

    @staticmethod
    def extract_timestamp_from_time_based_id(id_string: str) -> Optional[datetime]:
        """
        Extract timestamp from time-based ID.

        Args:
            id_string: Time-based ID

        Returns:
            Datetime object or None if invalid
        """
        try:
            parts = id_string.split('_')
            if len(parts) < 2:
                return None

            # Try to find timestamp part (numeric)
            for part in parts:
                if part.isdigit() and len(part) >= 10:  # Unix timestamp
                    timestamp = int(part)
                    # Handle milliseconds
                    if len(part) == 13:
                        timestamp = timestamp / 1000
                    return datetime.fromtimestamp(timestamp, tz=timezone.utc)

            return None
        except (ValueError, OSError):
            return None


class BatchIDGenerator:
    """
    Batch ID generation for high-volume operations.
    """

    @staticmethod
    def generate_id_batch(
            count: int,
            generator_func,
            *args,
            **kwargs
    ) -> List[str]:
        """
        Generate a batch of IDs using specified generator function.

        Args:
            count: Number of IDs to generate
            generator_func: ID generator function
            *args: Arguments for generator function
            **kwargs: Keyword arguments for generator function

        Returns:
            List of generated IDs
        """
        return [generator_func(*args, **kwargs) for _ in range(count)]

    @staticmethod
    def generate_sequential_batch(
            prefix: str,
            start_number: int,
            count: int,
            padding: int = 6
    ) -> List[str]:
        """
        Generate a batch of sequential IDs.

        Args:
            prefix: ID prefix
            start_number: Starting sequence number
            count: Number of IDs to generate
            padding: Zero padding length

        Returns:
            List of sequential IDs
        """
        return [
            IDGenerator.generate_sequential_id(prefix, start_number + i, padding)
            for i in range(count)
        ]


# Convenience functions for common ID generation
def generate_conversation_id() -> str:
    """Generate a conversation ID."""
    return ConversationIDGenerator.generate_conversation_id("")


def generate_message_id() -> str:
    """Generate a message ID."""
    return ConversationIDGenerator.generate_message_id("")


def generate_user_id() -> str:
    """Generate a user ID."""
    return BusinessIDGenerator.generate_user_id("")


def generate_request_id() -> str:
    """Generate a request tracking ID."""
    return TrackingIDGenerator.generate_request_id()


def generate_correlation_id() -> str:
    """Generate a correlation ID."""
    return TrackingIDGenerator.generate_correlation_id()


def generate_api_key() -> str:
    """
    Generate a secure API key.

    Returns:
        API key in format: cb_{env}_{random_hex}
    """
    from src.config.settings import get_settings

    settings = get_settings()
    env_prefix = settings.ENVIRONMENT.value[:4]  # First 4 chars of environment
    random_part = IDGenerator.generate_hex_id(32)  # 32 hex chars

    return f"cb_{env_prefix}_{random_part}"


def generate_tenant_subdomain(tenant_name: str) -> str:
    """
    Generate a tenant subdomain from tenant name.

    Args:
        tenant_name: Tenant name

    Returns:
        Valid subdomain string
    """
    # Clean tenant name for subdomain use
    import re

    # Convert to lowercase and replace spaces/special chars with hyphens
    subdomain = re.sub(r'[^a-z0-9\-]', '-', tenant_name.lower())
    # Remove multiple consecutive hyphens
    subdomain = re.sub(r'-+', '-', subdomain)
    # Remove leading/trailing hyphens
    subdomain = subdomain.strip('-')

    # Ensure minimum length and add random suffix if needed
    if len(subdomain) < 3:
        subdomain = f"tenant-{IDGenerator.generate_random_string(6, ALPHANUMERIC_LOWERCASE)}"
    elif len(subdomain) > 50:
        subdomain = subdomain[:50]

    # Add random suffix to ensure uniqueness
    unique_suffix = IDGenerator.generate_random_string(4, ALPHANUMERIC_LOWERCASE)
    return f"{subdomain}-{unique_suffix}"


def create_deterministic_id(input_string: str, prefix: str = "") -> str:
    """
    Create a deterministic ID from input string.

    Args:
        input_string: Input string to hash
        prefix: Optional prefix

    Returns:
        Deterministic ID
    """
    # Create SHA256 hash of input
    hash_obj = sha256(input_string.encode('utf-8'))
    hash_hex = hash_obj.hexdigest()[:16]  # Take first 16 chars

    if prefix:
        return f"{prefix}_{hash_hex}"
    else:
        return hash_hex


# Export commonly used classes and functions
__all__ = [
    'IDGenerator',
    'ConversationIDGenerator',
    'BusinessIDGenerator',
    'TrackingIDGenerator',
    'FileIDGenerator',
    'IDValidator',
    'BatchIDGenerator',
    'generate_conversation_id',
    'generate_message_id',
    'generate_user_id',
    'generate_request_id',
    'generate_correlation_id',
    'generate_api_key',
    'generate_tenant_subdomain',
    'create_deterministic_id',
]