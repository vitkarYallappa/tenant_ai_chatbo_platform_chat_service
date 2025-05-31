"""
Core business logic package for the multi-tenant AI chatbot platform.

This package contains the core business logic including channels, processors,
normalizers, and other essential components for message processing.
"""
from datetime import datetime, UTC

# Import main components from subpackages
from src.core import channels
from src.core import processors
from src.core import normalizers
from src.core import exceptions

# Import key classes for easy access
from src.core.channels import (
    BaseChannel,
    ChannelConfig,
    ChannelResponse,
    WebChannel,
    WhatsAppChannel,
    ChannelFactory,
    channel_factory,
    create_channel,
    get_registered_channels,
    get_channel_health_status,
    get_or_create_channel
)

from src.core.processors import (
    BaseProcessor,
    ProcessingContext,
    ProcessingResult,
    TextProcessor,
    MediaProcessor,
    LocationProcessor,
    ProcessorFactory,
    processor_factory,
    get_processor_for_content,
    get_processor_health_status,
    get_processing_strategy
)

from src.core.normalizers import (
    MessageNormalizer,
    ContentNormalizer,
    MetadataNormalizer
)

from src.core.exceptions import (
    CoreError,
    ChannelError,
    ProcessingError,
    ValidationError,
    ContentNormalizationError
)

__all__ = [
    # Subpackages
    "channels",
    "processors",
    "normalizers",
    "exceptions",

    # Channel components
    "BaseChannel",
    "ChannelConfig",
    "ChannelResponse",
    "WebChannel",
    "WhatsAppChannel",
    "ChannelFactory",
    "channel_factory",
    "create_channel",
    "get_registered_channels",

    # Processor components
    "BaseProcessor",
    "ProcessingContext",
    "ProcessingResult",
    "TextProcessor",
    "MediaProcessor",
    "LocationProcessor",
    "ProcessorFactory",
    "processor_factory",
    "get_processor_for_content",
    "get_processing_strategy",

    # Normalizer components
    "MessageNormalizer",
    "ContentNormalizer",
    "MetadataNormalizer",

    # Exception classes
    "CoreError",
    "ChannelError",
    "ProcessingError",
    "ValidationError",
    "ContentNormalizationError"
]

# Version information
__version__ = "1.0.0"
__author__ = "Chatbot Platform Team"
__description__ = "Core business logic for multi-tenant AI chatbot platform"

# Package-level configuration
CORE_CONFIG = {
    "version": __version__,
    "debug_mode": False,
    "performance_tracking": True,
    "default_language": "en",
    "default_timezone": "UTC"
}

# Core processing pipeline configuration
PIPELINE_CONFIG = {
    "max_processing_time_seconds": 30,
    "enable_parallel_processing": True,
    "enable_caching": True,
    "enable_metrics": True,
    "enable_health_checks": True,
    "retry_attempts": 3,
    "retry_delay_seconds": 1
}


class CorePipeline:
    """
    Main processing pipeline that orchestrates channels, processors, and normalizers.

    This class provides a high-level interface for processing messages through
    the complete pipeline from ingestion to delivery.
    """

    def __init__(self, config: dict = None):
        """
        Initialize the core processing pipeline.

        Args:
            config: Optional configuration overrides
        """
        self.config = {**PIPELINE_CONFIG, **(config or {})}

        # Initialize components
        self.channel_factory = channel_factory
        self.processor_factory = processor_factory
        self.message_normalizer = MessageNormalizer()

        # Performance tracking
        self.enable_metrics = self.config.get("enable_metrics", True)
        self.metrics = {
            "messages_processed": 0,
            "messages_failed": 0,
            "average_processing_time_ms": 0.0,
            "channels_used": {},
            "processors_used": {},
            "errors": {}
        }

    async def process_inbound_message(
            self,
            message_data: dict,
            channel_type: str,
            tenant_id: str,
            context: dict = None
    ) -> dict:
        """
        Process an inbound message through the complete pipeline.

        Args:
            message_data: Raw message data from channel
            channel_type: Source channel type
            tenant_id: Tenant identifier
            context: Optional processing context

        Returns:
            Processed message with analysis results
        """
        from datetime import datetime
        from src.models.types import ChannelType

        start_time = datetime.utcnow()

        try:
            # Convert channel type
            channel = ChannelType(channel_type.lower())

            # Step 1: Normalize the message
            normalized_message = await self.message_normalizer.normalize_message(
                message_data, channel, context
            )

            # Step 2: Extract content for processing
            content_data = normalized_message.get("normalized_content", {})

            # Step 3: Get appropriate processor and process content
            from src.models.types import MessageContent, MessageType

            # Create MessageContent from normalized data
            message_content = self._create_message_content(content_data)

            # Create processing context
            processing_context = self._create_processing_context(
                normalized_message, tenant_id, context
            )

            # Process content
            processor = get_processor_for_content(message_content)
            processing_result = await processor.process(message_content, processing_context)

            # Step 4: Combine results
            result = {
                "success": True,
                "message_id": normalized_message.get("normalized_metadata", {}).get("message_id"),
                "tenant_id": tenant_id,
                "channel": channel_type,
                "normalized_message": normalized_message,
                "processing_result": processing_result.dict() if processing_result else None,
                "processing_metadata": {
                    "processing_time_ms": self._calculate_processing_time(start_time),
                    "pipeline_version": __version__,
                    "timestamp": datetime.utcnow().isoformat()
                }
            }

            # Update metrics
            if self.enable_metrics:
                self._update_metrics(True, channel_type, processing_result, start_time)

            return result

        except Exception as e:
            # Handle errors
            error_result = {
                "success": False,
                "error": {
                    "type": type(e).__name__,
                    "message": str(e),
                    "details": getattr(e, 'details', {})
                },
                "tenant_id": tenant_id,
                "channel": channel_type,
                "processing_metadata": {
                    "processing_time_ms": self._calculate_processing_time(start_time),
                    "pipeline_version": __version__,
                    "timestamp": datetime.utcnow().isoformat()
                }
            }

            # Update error metrics
            if self.enable_metrics:
                self._update_metrics(False, channel_type, None, start_time, str(e))

            return error_result

    async def send_outbound_message(
            self,
            recipient: str,
            content,
            channel_type: str,
            tenant_id: str,
            metadata: dict = None
    ) -> dict:
        """
        Send an outbound message through the specified channel.

        Args:
            recipient: Message recipient identifier
            content: MessageContent to send
            channel_type: Target channel type
            tenant_id: Tenant identifier
            metadata: Optional message metadata

        Returns:
            Delivery result from channel
        """
        from datetime import datetime
        from src.models.types import ChannelType

        start_time = datetime.utcnow()

        try:
            # Get channel instance
            channel_enum = ChannelType(channel_type.lower())
            channel = get_or_create_channel(channel_enum, tenant_id)

            # Send message through channel
            channel_response = await channel.send_message(recipient, content, metadata)

            # Format response
            result = {
                "success": channel_response.success,
                "channel": channel_type,
                "tenant_id": tenant_id,
                "recipient": recipient,
                "message_id": channel_response.message_id,
                "platform_message_id": channel_response.platform_message_id,
                "delivery_status": channel_response.delivery_status.value,
                "error": None,
                "processing_metadata": {
                    "processing_time_ms": self._calculate_processing_time(start_time),
                    "pipeline_version": __version__,
                    "timestamp": datetime.utcnow().isoformat()
                }
            }

            if not channel_response.success:
                result["error"] = {
                    "code": channel_response.error_code,
                    "message": channel_response.error_message
                }

            return result

        except Exception as e:
            return {
                "success": False,
                "channel": channel_type,
                "tenant_id": tenant_id,
                "recipient": recipient,
                "error": {
                    "type": type(e).__name__,
                    "message": str(e)
                },
                "processing_metadata": {
                    "processing_time_ms": self._calculate_processing_time(start_time),
                    "pipeline_version": __version__,
                    "timestamp": datetime.utcnow().isoformat()
                }
            }

    def _create_message_content(self, content_data: dict):
        """Create MessageContent from normalized data."""
        from src.models.types import MessageContent, MessageType

        content_type = MessageType(content_data.get("type", "text"))

        # Create appropriate content structure
        return MessageContent(
            type=content_type,
            text=content_data.get("text"),
            language=content_data.get("language", "en"),
            media=content_data.get("media"),
            location=content_data.get("location"),
            quick_replies=content_data.get("quick_replies"),
            buttons=content_data.get("buttons")
        )

    def _create_processing_context(self, normalized_message: dict, tenant_id: str, context: dict):
        """Create ProcessingContext from normalized message."""
        metadata = normalized_message.get("normalized_metadata", {})

        return ProcessingContext(
            tenant_id=tenant_id,
            user_id=metadata.get("user_id", "unknown"),
            conversation_id=metadata.get("conversation_id"),
            session_id=metadata.get("session_id"),
            channel=normalized_message.get("channel", "unknown"),
            channel_metadata=metadata.get("channel_metadata", {}),
            user_profile=metadata.get("user_info", {}),
            conversation_context=context or {},
            request_id=metadata.get("message_id"),
            timestamp=datetime.now(UTC)
        )

    def _calculate_processing_time(self, start_time) -> int:
        """Calculate processing time in milliseconds."""
        from datetime import datetime
        return int((datetime.utcnow() - start_time).total_seconds() * 1000)

    def _update_metrics(
            self,
            success: bool,
            channel_type: str,
            processing_result,
            start_time,
            error_message: str = None
    ):
        """Update pipeline metrics."""
        processing_time = self._calculate_processing_time(start_time)

        # Update counters
        if success:
            self.metrics["messages_processed"] += 1
        else:
            self.metrics["messages_failed"] += 1

        # Update average processing time
        total_messages = self.metrics["messages_processed"] + self.metrics["messages_failed"]
        if total_messages > 0:
            current_avg = self.metrics["average_processing_time_ms"]
            self.metrics["average_processing_time_ms"] = (
                    (current_avg * (total_messages - 1) + processing_time) / total_messages
            )

        # Update channel usage
        if channel_type not in self.metrics["channels_used"]:
            self.metrics["channels_used"][channel_type] = 0
        self.metrics["channels_used"][channel_type] += 1

        # Update processor usage
        if processing_result and hasattr(processing_result, 'processor_version'):
            processor_name = getattr(processing_result, 'processor_name', 'unknown')
            if processor_name not in self.metrics["processors_used"]:
                self.metrics["processors_used"][processor_name] = 0
            self.metrics["processors_used"][processor_name] += 1

        # Update error tracking
        if error_message:
            if error_message not in self.metrics["errors"]:
                self.metrics["errors"][error_message] = 0
            self.metrics["errors"][error_message] += 1

    def get_metrics(self) -> dict:
        """Get current pipeline metrics."""
        return self.metrics.copy()

    def reset_metrics(self):
        """Reset pipeline metrics."""
        self.metrics = {
            "messages_processed": 0,
            "messages_failed": 0,
            "average_processing_time_ms": 0.0,
            "channels_used": {},
            "processors_used": {},
            "errors": {}
        }

    async def health_check(self) -> dict:
        """Perform health check on all pipeline components."""
        health_status = {
            "pipeline": "healthy",
            "components": {}
        }

        try:
            # Check channels
            channel_health = await get_channel_health_status()
            health_status["components"]["channels"] = channel_health

            # Check processors
            processor_health = await get_processor_health_status()
            health_status["components"]["processors"] = processor_health

            # Overall pipeline health
            all_healthy = (
                    all(channel_health.values()) and
                    all(processor_health.values())
            )

            health_status["pipeline"] = "healthy" if all_healthy else "degraded"

        except Exception as e:
            health_status["pipeline"] = "unhealthy"
            health_status["error"] = str(e)

        return health_status


# Global pipeline instance
core_pipeline = CorePipeline()


# Convenience functions
async def process_message(
        message_data: dict,
        channel_type: str,
        tenant_id: str,
        context: dict = None
) -> dict:
    """Process a message using the global pipeline."""
    return await core_pipeline.process_inbound_message(
        message_data, channel_type, tenant_id, context
    )


async def send_message(
        recipient: str,
        content,
        channel_type: str,
        tenant_id: str,
        metadata: dict = None
) -> dict:
    """Send a message using the global pipeline."""
    return await core_pipeline.send_outbound_message(
        recipient, content, channel_type, tenant_id, metadata
    )


def get_core_metrics() -> dict:
    """Get metrics from the global pipeline."""
    return core_pipeline.get_metrics()


async def get_core_health() -> dict:
    """Get health status from the global pipeline."""
    return await core_pipeline.health_check()


def get_core_config() -> dict:
    """Get current core configuration."""
    return CORE_CONFIG.copy()


def update_core_config(config_updates: dict):
    """Update core configuration."""
    CORE_CONFIG.update(config_updates)
