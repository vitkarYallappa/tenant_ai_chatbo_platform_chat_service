"""
Abstract base class defining the channel interface and common functionality.

This module provides the foundation for all channel implementations with standardized
message handling, validation, and response formatting.
"""

from abc import ABC, abstractmethod
from datetime import datetime
from typing import Dict, Any, Optional, List
from pydantic import BaseModel, Field
import structlog

from src.models.types import MessageContent, ChannelType, DeliveryStatus

logger = structlog.get_logger()


class ChannelConfig(BaseModel):
    """Configuration model for channel settings."""

    channel_type: ChannelType
    enabled: bool = True

    # Authentication
    api_token: Optional[str] = None
    api_secret: Optional[str] = None
    webhook_secret: Optional[str] = None

    # Rate limiting
    requests_per_minute: int = 60
    requests_per_day: int = 10000
    burst_limit: int = 120  # Allow burst up to 2x normal rate

    # Message formatting
    max_message_length: int = 4096
    supported_message_types: List[str] = Field(default_factory=lambda: ["text"])
    supports_rich_media: bool = False
    supports_buttons: bool = False
    supports_quick_replies: bool = False
    supports_carousel: bool = False
    supports_typing_indicators: bool = False

    # Delivery settings
    retry_attempts: int = 3
    retry_delay_seconds: int = 5
    timeout_seconds: int = 30
    delivery_confirmation_required: bool = True

    # Security settings
    verify_ssl: bool = True
    allowed_origins: List[str] = Field(default_factory=list)
    webhook_verification_enabled: bool = True

    # Feature flags
    features: Dict[str, bool] = Field(default_factory=dict)

    # Advanced configuration
    advanced_config: Dict[str, Any] = Field(default_factory=dict)

    class Config:
        use_enum_values = True


class ChannelResponse(BaseModel):
    """Standardized response from channel operations."""

    success: bool
    channel_type: ChannelType
    message_id: Optional[str] = None
    platform_message_id: Optional[str] = None
    delivery_status: DeliveryStatus = DeliveryStatus.SENT
    timestamp: datetime = Field(default_factory=datetime.utcnow)

    # Error information
    error_code: Optional[str] = None
    error_message: Optional[str] = None
    is_retryable: bool = True

    # Delivery metadata
    recipient: Optional[str] = None
    retry_count: int = 0
    delivery_attempt_at: Optional[datetime] = None
    delivery_confirmed_at: Optional[datetime] = None

    # Performance metrics
    processing_time_ms: Optional[int] = None
    queue_time_ms: Optional[int] = None

    # Channel-specific metadata
    metadata: Dict[str, Any] = Field(default_factory=dict)

    # Webhook information
    webhook_url: Optional[str] = None
    webhook_delivery_status: Optional[str] = None

    class Config:
        use_enum_values = True


class ChannelMetrics(BaseModel):
    """Channel performance and usage metrics."""

    total_messages_sent: int = 0
    total_messages_failed: int = 0
    success_rate: float = 0.0
    average_response_time_ms: float = 0.0
    rate_limit_hits: int = 0

    # Time-based metrics
    messages_last_hour: int = 0
    messages_last_day: int = 0

    # Error tracking
    common_errors: Dict[str, int] = Field(default_factory=dict)
    last_error_at: Optional[datetime] = None

    # Health status
    health_status: str = "unknown"  # healthy, degraded, unhealthy, unknown
    last_health_check: Optional[datetime] = None
    consecutive_failures: int = 0


class BaseChannel(ABC):
    """Abstract base class for all channel implementations."""

    def __init__(self, config: ChannelConfig):
        self.config = config
        self.logger = structlog.get_logger(self.__class__.__name__)
        self.metrics = ChannelMetrics()
        self._validate_config()

        # Initialize rate limiting state
        self._rate_limit_window_start = datetime.utcnow()
        self._rate_limit_count = 0

        self.logger.info(
            "Channel initialized",
            channel_type=self.channel_type.value,
            enabled=self.config.enabled,
            max_message_length=self.config.max_message_length
        )

    @property
    @abstractmethod
    def channel_type(self) -> ChannelType:
        """Return the channel type."""
        pass

    @abstractmethod
    async def send_message(
            self,
            recipient: str,
            content: MessageContent,
            metadata: Optional[Dict[str, Any]] = None
    ) -> ChannelResponse:
        """
        Send a message through this channel.

        Args:
            recipient: Channel-specific recipient identifier
            content: Message content to send
            metadata: Optional channel-specific metadata

        Returns:
            ChannelResponse with delivery information

        Raises:
            ChannelError: When message sending fails
            ValidationError: When input validation fails
        """
        pass

    @abstractmethod
    async def validate_recipient(self, recipient: str) -> bool:
        """
        Validate recipient format for this channel.

        Args:
            recipient: Recipient identifier to validate

        Returns:
            True if recipient format is valid
        """
        pass

    @abstractmethod
    async def format_message(
            self,
            content: MessageContent
    ) -> Dict[str, Any]:
        """
        Format message content for channel API.

        Args:
            content: Message content to format

        Returns:
            Formatted message for channel API

        Raises:
            ValidationError: When content cannot be formatted for channel
        """
        pass

    async def validate_content(self, content: MessageContent) -> bool:
        """
        Validate message content for this channel.

        Args:
            content: Message content to validate

        Returns:
            True if content is valid for this channel
        """
        try:
            # Check if channel is enabled
            if not self.config.enabled:
                self.logger.warning(
                    "Channel is disabled",
                    channel=self.channel_type.value
                )
                return False

            # Check message type support
            if content.type.value not in self.config.supported_message_types:
                self.logger.warning(
                    "Unsupported message type",
                    channel=self.channel_type.value,
                    message_type=content.type.value,
                    supported_types=self.config.supported_message_types
                )
                return False

            # Check text length
            if content.text and len(content.text) > self.config.max_message_length:
                self.logger.warning(
                    "Message text too long",
                    channel=self.channel_type.value,
                    length=len(content.text),
                    max_length=self.config.max_message_length
                )
                return False

            # Check rich media support
            if content.media and not self.config.supports_rich_media:
                self.logger.warning(
                    "Rich media not supported",
                    channel=self.channel_type.value
                )
                return False

            # Check buttons support
            if content.buttons and not self.config.supports_buttons:
                self.logger.warning(
                    "Buttons not supported",
                    channel=self.channel_type.value
                )
                return False

            # Check quick replies support
            if content.quick_replies and not self.config.supports_quick_replies:
                self.logger.warning(
                    "Quick replies not supported",
                    channel=self.channel_type.value
                )
                return False

            # Check carousel support
            if hasattr(content, 'carousel') and content.carousel and not self.config.supports_carousel:
                self.logger.warning(
                    "Carousel not supported",
                    channel=self.channel_type.value
                )
                return False

            return True

        except Exception as e:
            self.logger.error(
                "Content validation failed",
                channel=self.channel_type.value,
                error=str(e)
            )
            return False

    async def check_rate_limit(self) -> bool:
        """
        Check if request is within rate limits.

        Returns:
            True if request is allowed, False if rate limited
        """
        now = datetime.utcnow()

        # Reset window if minute has passed
        if (now - self._rate_limit_window_start).total_seconds() >= 60:
            self._rate_limit_window_start = now
            self._rate_limit_count = 0

        # Check if within limits
        if self._rate_limit_count >= self.config.requests_per_minute:
            self.metrics.rate_limit_hits += 1
            self.logger.warning(
                "Rate limit exceeded",
                channel=self.channel_type.value,
                requests_per_minute=self.config.requests_per_minute,
                current_count=self._rate_limit_count
            )
            return False

        self._rate_limit_count += 1
        return True

    async def health_check(self) -> bool:
        """
        Perform health check for this channel.

        Returns:
            True if channel is healthy
        """
        try:
            # Update last health check time
            self.metrics.last_health_check = datetime.utcnow()

            # Basic checks
            if not self.config.enabled:
                self.metrics.health_status = "unhealthy"
                return False

            # Check configuration
            if not self._validate_config():
                self.metrics.health_status = "unhealthy"
                return False

            # Check rate limit status
            if self.metrics.rate_limit_hits > 10:  # Threshold for degraded service
                self.metrics.health_status = "degraded"
                return True

            # Check error rate
            total_messages = self.metrics.total_messages_sent + self.metrics.total_messages_failed
            if total_messages > 0:
                error_rate = self.metrics.total_messages_failed / total_messages
                if error_rate > 0.1:  # 10% error rate threshold
                    self.metrics.health_status = "degraded"
                    return True
                elif error_rate > 0.25:  # 25% error rate threshold
                    self.metrics.health_status = "unhealthy"
                    return False

            # Channel-specific health checks (override in subclasses)
            channel_healthy = await self._channel_specific_health_check()

            if channel_healthy:
                self.metrics.health_status = "healthy"
                self.metrics.consecutive_failures = 0
            else:
                self.metrics.consecutive_failures += 1
                if self.metrics.consecutive_failures >= 3:
                    self.metrics.health_status = "unhealthy"
                else:
                    self.metrics.health_status = "degraded"

            return channel_healthy

        except Exception as e:
            self.logger.error(
                "Health check failed",
                channel=self.channel_type.value,
                error=str(e)
            )
            self.metrics.health_status = "unhealthy"
            self.metrics.consecutive_failures += 1
            return False

    async def _channel_specific_health_check(self) -> bool:
        """
        Perform channel-specific health checks.
        Override in subclasses for custom health validation.

        Returns:
            True if channel-specific checks pass
        """
        return True

    def update_metrics(
            self,
            success: bool,
            response_time_ms: int,
            error_code: Optional[str] = None
    ) -> None:
        """Update channel metrics after a request."""
        if success:
            self.metrics.total_messages_sent += 1
            self.metrics.messages_last_hour += 1
            self.metrics.messages_last_day += 1
        else:
            self.metrics.total_messages_failed += 1
            self.metrics.last_error_at = datetime.utcnow()

            if error_code:
                if error_code not in self.metrics.common_errors:
                    self.metrics.common_errors[error_code] = 0
                self.metrics.common_errors[error_code] += 1

        # Update success rate
        total = self.metrics.total_messages_sent + self.metrics.total_messages_failed
        if total > 0:
            self.metrics.success_rate = self.metrics.total_messages_sent / total

        # Update average response time
        if self.metrics.average_response_time_ms == 0:
            self.metrics.average_response_time_ms = response_time_ms
        else:
            # Moving average
            self.metrics.average_response_time_ms = (
                    self.metrics.average_response_time_ms * 0.9 + response_time_ms * 0.1
            )

    def get_metrics(self) -> ChannelMetrics:
        """Get current channel metrics."""
        return self.metrics.copy()

    def _validate_config(self) -> bool:
        """Validate channel configuration."""
        try:
            if self.config.requests_per_minute <= 0:
                raise ValueError("Requests per minute must be positive")

            if self.config.max_message_length <= 0:
                raise ValueError("Max message length must be positive")

            if self.config.retry_attempts < 0:
                raise ValueError("Retry attempts cannot be negative")

            if self.config.timeout_seconds <= 0:
                raise ValueError("Timeout seconds must be positive")

            return True

        except Exception as e:
            self.logger.error(
                "Configuration validation failed",
                channel=self.channel_type.value,
                error=str(e)
            )
            return False

    def _create_error_response(
            self,
            error_code: str,
            error_message: str,
            recipient: Optional[str] = None,
            is_retryable: bool = True,
            processing_time_ms: Optional[int] = None
    ) -> ChannelResponse:
        """Create standardized error response."""
        return ChannelResponse(
            success=False,
            channel_type=self.channel_type,
            delivery_status=DeliveryStatus.FAILED,
            error_code=error_code,
            error_message=error_message,
            recipient=recipient,
            is_retryable=is_retryable,
            processing_time_ms=processing_time_ms,
            timestamp=datetime.utcnow()
        )

    def _create_success_response(
            self,
            message_id: str,
            platform_message_id: Optional[str] = None,
            recipient: Optional[str] = None,
            metadata: Optional[Dict[str, Any]] = None,
            processing_time_ms: Optional[int] = None
    ) -> ChannelResponse:
        """Create standardized success response."""
        return ChannelResponse(
            success=True,
            channel_type=self.channel_type,
            message_id=message_id,
            platform_message_id=platform_message_id,
            delivery_status=DeliveryStatus.SENT,
            recipient=recipient,
            metadata=metadata or {},
            processing_time_ms=processing_time_ms,
            timestamp=datetime.utcnow()
        )

    async def __aenter__(self):
        """Async context manager entry."""
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        # Cleanup resources if needed
        pass

    def __str__(self) -> str:
        """String representation of channel."""
        return f"{self.__class__.__name__}(type={self.channel_type.value}, enabled={self.config.enabled})"

    def __repr__(self) -> str:
        """Detailed string representation of channel."""
        return (
            f"{self.__class__.__name__}("
            f"type={self.channel_type.value}, "
            f"enabled={self.config.enabled}, "
            f"max_length={self.config.max_message_length}, "
            f"rate_limit={self.config.requests_per_minute}/min"
            f")"
        )