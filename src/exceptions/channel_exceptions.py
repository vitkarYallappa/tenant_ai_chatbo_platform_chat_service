"""
Channel-specific exceptions for Chat Service.

This module provides custom exception classes for channel-related
errors including channel configuration, message delivery, and
platform-specific issues.
"""

from typing import Optional, Dict, Any

from src.exceptions.base_exceptions import ChatServiceException
from src.config.constants import ErrorCategory


class ChannelException(ChatServiceException):
    """Base exception for all channel-related errors."""

    def __init__(
            self,
            message: str,
            channel: Optional[str] = None,
            **kwargs
    ):
        details = kwargs.pop("details", {})
        if channel:
            details["channel"] = channel

        super().__init__(
            message=message,
            error_code="CHANNEL_ERROR",
            status_code=502,
            category=ErrorCategory.EXTERNAL,
            details=details,
            **kwargs
        )


class ChannelNotSupportedException(ChannelException):
    """Exception for unsupported channel types."""

    def __init__(
            self,
            channel: str,
            supported_channels: Optional[list] = None,
            **kwargs
    ):
        details = kwargs.pop("details", {})
        details["unsupported_channel"] = channel

        if supported_channels:
            details["supported_channels"] = supported_channels

        message = f"Channel '{channel}' is not supported"
        user_message = f"The channel '{channel}' is not supported by this service"

        super().__init__(
            message=message,
            channel=channel,
            error_code="CHANNEL_NOT_SUPPORTED",
            status_code=400,
            category=ErrorCategory.VALIDATION,
            user_message=user_message,
            details=details,
            **kwargs
        )


class ChannelConfigurationError(ChannelException):
    """Exception for channel configuration errors."""

    def __init__(
            self,
            message: str,
            channel: str,
            config_key: Optional[str] = None,
            **kwargs
    ):
        details = kwargs.pop("details", {})
        if config_key:
            details["config_key"] = config_key

        super().__init__(
            message=message,
            channel=channel,
            error_code="CHANNEL_CONFIG_ERROR",
            status_code=500,
            category=ErrorCategory.INTERNAL,
            user_message="Channel configuration error. Please contact support.",
            details=details,
            **kwargs
        )


class ChannelConnectionError(ChannelException):
    """Exception for channel connection failures."""

    def __init__(
            self,
            message: str,
            channel: str,
            endpoint: Optional[str] = None,
            **kwargs
    ):
        details = kwargs.pop("details", {})
        if endpoint:
            details["endpoint"] = endpoint

        super().__init__(
            message=message,
            channel=channel,
            error_code="CHANNEL_CONNECTION_ERROR",
            status_code=502,
            category=ErrorCategory.NETWORK,
            user_message=f"Unable to connect to {channel} service. Please try again later.",
            retryable=True,
            details=details,
            **kwargs
        )


class ChannelAuthenticationError(ChannelException):
    """Exception for channel authentication failures."""

    def __init__(
            self,
            message: str,
            channel: str,
            **kwargs
    ):
        super().__init__(
            message=message,
            channel=channel,
            error_code="CHANNEL_AUTH_ERROR",
            status_code=401,
            category=ErrorCategory.AUTHENTICATION,
            user_message=f"Authentication failed for {channel} channel. Please check your credentials.",
            **kwargs
        )


class MessageDeliveryError(ChannelException):
    """Exception for message delivery failures."""

    def __init__(
            self,
            message: str,
            channel: str,
            message_id: Optional[str] = None,
            recipient: Optional[str] = None,
            delivery_attempt: int = 1,
            **kwargs
    ):
        details = kwargs.pop("details", {})
        if message_id:
            details["message_id"] = message_id
        if recipient:
            details["recipient"] = recipient
        details["delivery_attempt"] = delivery_attempt

        super().__init__(
            message=message,
            channel=channel,
            error_code="MESSAGE_DELIVERY_ERROR",
            status_code=502,
            category=ErrorCategory.EXTERNAL,
            user_message="Failed to deliver message. We'll retry automatically.",
            retryable=True,
            details=details,
            **kwargs
        )


class MessageSizeExceededError(ChannelException):
    """Exception for message size limit exceeded."""

    def __init__(
            self,
            channel: str,
            actual_size: int,
            max_size: int,
            message_type: str = "text",
            **kwargs
    ):
        details = kwargs.pop("details", {})
        details.update({
            "actual_size": actual_size,
            "max_size": max_size,
            "message_type": message_type
        })

        message = f"Message size {actual_size} exceeds limit {max_size} for channel {channel}"
        user_message = f"Message is too large for {channel}. Maximum size is {max_size} characters."

        super().__init__(
            message=message,
            channel=channel,
            error_code="MESSAGE_SIZE_EXCEEDED",
            status_code=413,
            category=ErrorCategory.VALIDATION,
            user_message=user_message,
            details=details,
            **kwargs
        )


class UnsupportedMessageTypeError(ChannelException):
    """Exception for unsupported message types on specific channels."""

    def __init__(
            self,
            channel: str,
            message_type: str,
            supported_types: Optional[list] = None,
            **kwargs
    ):
        details = kwargs.pop("details", {})
        details["unsupported_message_type"] = message_type

        if supported_types:
            details["supported_message_types"] = supported_types

        message = f"Message type '{message_type}' is not supported on channel '{channel}'"
        user_message = f"This message type is not supported on {channel}"

        super().__init__(
            message=message,
            channel=channel,
            error_code="UNSUPPORTED_MESSAGE_TYPE",
            status_code=400,
            category=ErrorCategory.VALIDATION,
            user_message=user_message,
            details=details,
            **kwargs
        )


class ChannelRateLimitError(ChannelException):
    """Exception for channel-specific rate limiting."""

    def __init__(
            self,
            channel: str,
            limit: int,
            window_seconds: int,
            retry_after: Optional[int] = None,
            **kwargs
    ):
        details = kwargs.pop("details", {})
        details.update({
            "rate_limit": limit,
            "window_seconds": window_seconds
        })

        if retry_after:
            details["retry_after_seconds"] = retry_after

        message = f"Rate limit exceeded for channel {channel}: {limit} requests per {window_seconds} seconds"
        user_message = f"Too many messages sent to {channel}. Please wait before sending more."

        super().__init__(
            message=message,
            channel=channel,
            error_code="CHANNEL_RATE_LIMIT",
            status_code=429,
            category=ErrorCategory.RATE_LIMIT,
            user_message=user_message,
            retryable=True,
            details=details,
            **kwargs
        )


# WhatsApp-specific exceptions
class WhatsAppException(ChannelException):
    """Base exception for WhatsApp-specific errors."""

    def __init__(self, message: str, **kwargs):
        super().__init__(
            message=message,
            channel="whatsapp",
            **kwargs
        )


class WhatsAppBusinessAccountError(WhatsAppException):
    """Exception for WhatsApp Business Account issues."""

    def __init__(
            self,
            message: str = "WhatsApp Business Account error",
            account_id: Optional[str] = None,
            **kwargs
    ):
        details = kwargs.pop("details", {})
        if account_id:
            details["account_id"] = account_id

        super().__init__(
            message=message,
            error_code="WHATSAPP_ACCOUNT_ERROR",
            user_message="WhatsApp Business Account issue. Please contact support.",
            details=details,
            **kwargs
        )


class WhatsAppPhoneNumberError(WhatsAppException):
    """Exception for WhatsApp phone number validation issues."""

    def __init__(
            self,
            phone_number: str,
            reason: str = "Invalid phone number format",
            **kwargs
    ):
        details = kwargs.pop("details", {})
        details["phone_number"] = phone_number
        details["reason"] = reason

        message = f"WhatsApp phone number error: {reason}"
        user_message = "Invalid phone number format for WhatsApp"

        super().__init__(
            message=message,
            error_code="WHATSAPP_PHONE_ERROR",
            status_code=400,
            category=ErrorCategory.VALIDATION,
            user_message=user_message,
            details=details,
            **kwargs
        )


class WhatsAppMediaUploadError(WhatsAppException):
    """Exception for WhatsApp media upload failures."""

    def __init__(
            self,
            media_type: str,
            file_size: Optional[int] = None,
            reason: Optional[str] = None,
            **kwargs
    ):
        details = kwargs.pop("details", {})
        details["media_type"] = media_type

        if file_size:
            details["file_size"] = file_size
        if reason:
            details["reason"] = reason

        message = f"WhatsApp media upload failed for {media_type}"
        user_message = "Failed to upload media to WhatsApp. Please try again."

        super().__init__(
            message=message,
            error_code="WHATSAPP_MEDIA_UPLOAD_ERROR",
            status_code=502,
            category=ErrorCategory.EXTERNAL,
            user_message=user_message,
            retryable=True,
            details=details,
            **kwargs
        )


# Slack-specific exceptions
class SlackException(ChannelException):
    """Base exception for Slack-specific errors."""

    def __init__(self, message: str, **kwargs):
        super().__init__(
            message=message,
            channel="slack",
            **kwargs
        )


class SlackWorkspaceError(SlackException):
    """Exception for Slack workspace access issues."""

    def __init__(
            self,
            workspace_id: Optional[str] = None,
            reason: str = "Workspace access denied",
            **kwargs
    ):
        details = kwargs.pop("details", {})
        if workspace_id:
            details["workspace_id"] = workspace_id
        details["reason"] = reason

        message = f"Slack workspace error: {reason}"
        user_message = "Unable to access Slack workspace. Please check permissions."

        super().__init__(
            message=message,
            error_code="SLACK_WORKSPACE_ERROR",
            status_code=403,
            category=ErrorCategory.AUTHORIZATION,
            user_message=user_message,
            details=details,
            **kwargs
        )


class SlackChannelNotFoundError(SlackException):
    """Exception for Slack channel not found."""

    def __init__(
            self,
            channel_id: str,
            **kwargs
    ):
        details = kwargs.pop("details", {})
        details["slack_channel_id"] = channel_id

        message = f"Slack channel not found: {channel_id}"
        user_message = "Slack channel not found or not accessible"

        super().__init__(
            message=message,
            error_code="SLACK_CHANNEL_NOT_FOUND",
            status_code=404,
            category=ErrorCategory.NOT_FOUND,
            user_message=user_message,
            details=details,
            **kwargs
        )


# Microsoft Teams-specific exceptions
class TeamsException(ChannelException):
    """Base exception for Microsoft Teams-specific errors."""

    def __init__(self, message: str, **kwargs):
        super().__init__(
            message=message,
            channel="teams",
            **kwargs
        )


class TeamsAuthenticationError(TeamsException):
    """Exception for Teams authentication issues."""

    def __init__(
            self,
            reason: str = "Authentication failed",
            **kwargs
    ):
        details = kwargs.pop("details", {})
        details["reason"] = reason

        message = f"Teams authentication error: {reason}"
        user_message = "Microsoft Teams authentication failed. Please re-authenticate."

        super().__init__(
            message=message,
            error_code="TEAMS_AUTH_ERROR",
            status_code=401,
            category=ErrorCategory.AUTHENTICATION,
            user_message=user_message,
            details=details,
            **kwargs
        )


class TeamsConversationError(TeamsException):
    """Exception for Teams conversation access issues."""

    def __init__(
            self,
            conversation_id: str,
            reason: str = "Conversation not accessible",
            **kwargs
    ):
        details = kwargs.pop("details", {})
        details["teams_conversation_id"] = conversation_id
        details["reason"] = reason

        message = f"Teams conversation error: {reason}"
        user_message = "Unable to access Teams conversation"

        super().__init__(
            message=message,
            error_code="TEAMS_CONVERSATION_ERROR",
            status_code=403,
            category=ErrorCategory.AUTHORIZATION,
            user_message=user_message,
            details=details,
            **kwargs
        )


# Web channel-specific exceptions
class WebChannelException(ChannelException):
    """Base exception for web channel-specific errors."""

    def __init__(self, message: str, **kwargs):
        super().__init__(
            message=message,
            channel="web",
            **kwargs
        )


class WebSocketConnectionError(WebChannelException):
    """Exception for WebSocket connection issues."""

    def __init__(
            self,
            session_id: Optional[str] = None,
            reason: str = "WebSocket connection failed",
            **kwargs
    ):
        details = kwargs.pop("details", {})
        if session_id:
            details["session_id"] = session_id
        details["reason"] = reason

        message = f"WebSocket connection error: {reason}"
        user_message = "Connection lost. Please refresh the page."

        super().__init__(
            message=message,
            error_code="WEBSOCKET_CONNECTION_ERROR",
            status_code=503,
            category=ErrorCategory.NETWORK,
            user_message=user_message,
            retryable=True,
            details=details,
            **kwargs
        )


class SessionExpiredError(WebChannelException):
    """Exception for expired web sessions."""

    def __init__(
            self,
            session_id: str,
            **kwargs
    ):
        details = kwargs.pop("details", {})
        details["session_id"] = session_id

        message = f"Web session expired: {session_id}"
        user_message = "Your session has expired. Please refresh the page."

        super().__init__(
            message=message,
            error_code="SESSION_EXPIRED",
            status_code=401,
            category=ErrorCategory.AUTHENTICATION,
            user_message=user_message,
            details=details,
            **kwargs
        )


# Voice channel-specific exceptions
class VoiceChannelException(ChannelException):
    """Base exception for voice channel-specific errors."""

    def __init__(self, message: str, **kwargs):
        super().__init__(
            message=message,
            channel="voice",
            **kwargs
        )


class SpeechRecognitionError(VoiceChannelException):
    """Exception for speech recognition failures."""

    def __init__(
            self,
            audio_duration: Optional[float] = None,
            confidence_score: Optional[float] = None,
            **kwargs
    ):
        details = kwargs.pop("details", {})
        if audio_duration:
            details["audio_duration"] = audio_duration
        if confidence_score:
            details["confidence_score"] = confidence_score

        message = "Speech recognition failed"
        user_message = "Sorry, I couldn't understand what you said. Please try again."

        super().__init__(
            message=message,
            error_code="SPEECH_RECOGNITION_ERROR",
            status_code=422,
            category=ErrorCategory.VALIDATION,
            user_message=user_message,
            retryable=True,
            details=details,
            **kwargs
        )


class TextToSpeechError(VoiceChannelException):
    """Exception for text-to-speech conversion failures."""

    def __init__(
            self,
            text_length: Optional[int] = None,
            voice_id: Optional[str] = None,
            **kwargs
    ):
        details = kwargs.pop("details", {})
        if text_length:
            details["text_length"] = text_length
        if voice_id:
            details["voice_id"] = voice_id

        message = "Text-to-speech conversion failed"
        user_message = "Unable to generate voice response. Please try again."

        super().__init__(
            message=message,
            error_code="TEXT_TO_SPEECH_ERROR",
            status_code=502,
            category=ErrorCategory.EXTERNAL,
            user_message=user_message,
            retryable=True,
            details=details,
            **kwargs
        )


# Convenience functions for raising channel-specific errors
def raise_channel_not_supported(channel: str, supported_channels: list = None):
    """Raise ChannelNotSupportedException."""
    raise ChannelNotSupportedException(
        channel=channel,
        supported_channels=supported_channels
    )


def raise_message_delivery_error(
        channel: str,
        message_id: str = None,
        recipient: str = None,
        reason: str = "Delivery failed"
):
    """Raise MessageDeliveryError."""
    raise MessageDeliveryError(
        message=f"Message delivery failed: {reason}",
        channel=channel,
        message_id=message_id,
        recipient=recipient
    )


def raise_unsupported_message_type(channel: str, message_type: str, supported_types: list = None):
    """Raise UnsupportedMessageTypeError."""
    raise UnsupportedMessageTypeError(
        channel=channel,
        message_type=message_type,
        supported_types=supported_types
    )


# Export all exception classes
__all__ = [
    # Base channel exceptions
    'ChannelException',
    'ChannelNotSupportedException',
    'ChannelConfigurationError',
    'ChannelConnectionError',
    'ChannelAuthenticationError',
    'MessageDeliveryError',
    'MessageSizeExceededError',
    'UnsupportedMessageTypeError',
    'ChannelRateLimitError',

    # WhatsApp exceptions
    'WhatsAppException',
    'WhatsAppBusinessAccountError',
    'WhatsAppPhoneNumberError',
    'WhatsAppMediaUploadError',

    # Slack exceptions
    'SlackException',
    'SlackWorkspaceError',
    'SlackChannelNotFoundError',

    # Teams exceptions
    'TeamsException',
    'TeamsAuthenticationError',
    'TeamsConversationError',

    # Web channel exceptions
    'WebChannelException',
    'WebSocketConnectionError',
    'SessionExpiredError',

    # Voice channel exceptions
    'VoiceChannelException',
    'SpeechRecognitionError',
    'TextToSpeechError',

    # Convenience functions
    'raise_channel_not_supported',
    'raise_message_delivery_error',
    'raise_unsupported_message_type',
]