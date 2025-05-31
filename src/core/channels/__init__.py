"""
Channels package for multi-channel message handling.

This package provides channel implementations for different messaging platforms
including WhatsApp, Web, Messenger, Slack, and Teams.
"""

from src.core.channels.base_channel import (
    BaseChannel,
    ChannelConfig,
    ChannelResponse,
    ChannelMetrics
)
from src.core.channels.web_channel import WebChannel
from src.core.channels.whatsapp_channel import WhatsAppChannel
from src.core.channels.channel_factory import (
    ChannelFactory,
    channel_factory,
    create_channel,
    get_or_create_channel,
    get_registered_channels,
    register_custom_channel
)

__all__ = [
    # Base classes and data models
    "BaseChannel",
    "ChannelConfig",
    "ChannelResponse",
    "ChannelMetrics",

    # Channel implementations
    "WebChannel",
    "WhatsAppChannel",

    # Factory and convenience functions
    "ChannelFactory",
    "channel_factory",
    "create_channel",
    "get_or_create_channel",
    "get_registered_channels",
    "register_custom_channel",
    "get_channel_health_status"
]

# Version information
__version__ = "1.0.0"
__author__ = "Chatbot Platform Team"
__description__ = "Multi-channel message handling and delivery"

# Module-level configuration
DEFAULT_CHANNEL_CONFIGS = {
    "web": {
        "channel_type": "web",
        "enabled": True,
        "max_message_length": 4096,
        "supported_message_types": ["text", "image", "file", "location"],
        "supports_rich_media": True,
        "supports_buttons": True,
        "supports_quick_replies": True,
        "features": {
            "websocket_enabled": True,
            "typing_indicators": True,
            "message_history": True,
            "file_upload": True,
            "markdown_support": True,
            "html_support": False,
            "max_file_size_mb": 10
        }
    },

    "whatsapp": {
        "channel_type": "whatsapp",
        "enabled": True,
        "max_message_length": 4096,
        "supported_message_types": ["text", "image", "audio", "video", "file", "location"],
        "supports_rich_media": True,
        "supports_buttons": True,
        "supports_quick_replies": True,
        "retry_attempts": 3,
        "timeout_seconds": 30,
        "features": {
            "phone_number_id": None,  # Must be configured per tenant
            "business_account_id": None,  # Must be configured per tenant
            "verify_token": None  # Must be configured per tenant
        }
    },

    "messenger": {
        "channel_type": "messenger",
        "enabled": False,  # Disabled by default
        "max_message_length": 2000,
        "supported_message_types": ["text", "image", "audio", "video", "file"],
        "supports_rich_media": True,
        "supports_buttons": True,
        "supports_quick_replies": True,
        "features": {
            "page_access_token": None,
            "app_secret": None,
            "verify_token": None
        }
    },

    "slack": {
        "channel_type": "slack",
        "enabled": False,  # Disabled by default
        "max_message_length": 40000,
        "supported_message_types": ["text", "image", "file"],
        "supports_rich_media": True,
        "supports_buttons": True,
        "supports_quick_replies": False,
        "features": {
            "bot_token": None,
            "signing_secret": None,
            "app_token": None
        }
    },

    "teams": {
        "channel_type": "teams",
        "enabled": False,  # Disabled by default
        "max_message_length": 28000,
        "supported_message_types": ["text", "image", "file"],
        "supports_rich_media": True,
        "supports_buttons": True,
        "supports_quick_replies": False,
        "features": {
            "app_id": None,
            "app_password": None
        }
    }
}

def get_default_channel_config(channel_type: str = None) -> dict:
    """
    Get default configuration for channels.

    Args:
        channel_type: Specific channel type ('web', 'whatsapp', etc.)
                     or None for all configs

    Returns:
        Configuration dictionary
    """
    if channel_type and channel_type in DEFAULT_CHANNEL_CONFIGS:
        return DEFAULT_CHANNEL_CONFIGS[channel_type].copy()
    return DEFAULT_CHANNEL_CONFIGS.copy()


def create_channel_suite(tenant_configs: dict, tenant_id: str = None) -> dict:
    """
    Create a complete suite of channels for a tenant.

    Args:
        tenant_configs: Dictionary of channel configurations by channel type
        tenant_id: Optional tenant identifier

    Returns:
        Dictionary mapping channel types to channel instances
    """
    return channel_factory.create_channels_from_config(tenant_configs, tenant_id)


def get_enabled_channels(tenant_configs: dict) -> list:
    """
    Get list of enabled channel types from configuration.

    Args:
        tenant_configs: Dictionary of channel configurations

    Returns:
        List of enabled channel type strings
    """
    enabled = []
    for channel_type, config in tenant_configs.items():
        if config.get("enabled", False):
            enabled.append(channel_type)
    return enabled


def validate_channel_configs(tenant_configs: dict) -> dict:
    """
    Validate channel configurations and return validation results.

    Args:
        tenant_configs: Dictionary of channel configurations

    Returns:
        Dictionary with validation results for each channel
    """
    from src.models.types import ChannelType

    results = {}

    for channel_name, config in tenant_configs.items():
        try:
            # Convert to enum to validate
            channel_type = ChannelType(channel_name.lower())

            # Create ChannelConfig to validate structure
            channel_config = ChannelConfig(**config)

            # Use factory validation
            try:
                channel_factory._validate_config(channel_type, channel_config)
                results[channel_name] = {
                    "valid": True,
                    "errors": [],
                    "warnings": []
                }
            except Exception as e:
                results[channel_name] = {
                    "valid": False,
                    "errors": [str(e)],
                    "warnings": []
                }

        except ValueError:
            results[channel_name] = {
                "valid": False,
                "errors": [f"Unknown channel type: {channel_name}"],
                "warnings": []
            }
        except Exception as e:
            results[channel_name] = {
                "valid": False,
                "errors": [f"Configuration error: {str(e)}"],
                "warnings": []
            }

    return results


async def send_message_multi_channel(
    channels: dict,
    recipient: str,
    content,
    metadata: dict = None,
    preferred_channel: str = None
) -> dict:
    """
    Send message through multiple channels with fallback support.

    Args:
        channels: Dictionary of channel instances
        recipient: Recipient identifier
        content: MessageContent to send
        metadata: Optional metadata
        preferred_channel: Preferred channel type to try first

    Returns:
        Dictionary with results from each channel attempt
    """
    results = {}

    # Try preferred channel first
    if preferred_channel and preferred_channel in channels:
        try:
            result = await channels[preferred_channel].send_message(
                recipient, content, metadata
            )
            results[preferred_channel] = result

            if result.success:
                return results  # Success, no need to try other channels
        except Exception as e:
            results[preferred_channel] = {
                "success": False,
                "error": str(e)
            }

    # Try other channels as fallback
    for channel_type, channel in channels.items():
        if channel_type == preferred_channel:
            continue  # Already tried

        try:
            result = await channel.send_message(recipient, content, metadata)
            results[channel_type] = result

            if result.success:
                break  # Success, stop trying other channels
        except Exception as e:
            results[channel_type] = {
                "success": False,
                "error": str(e)
            }

    return results


async def get_channel_health_status(channels: dict = None, tenant_id: str = None) -> dict:
    """
    Get health status of channels.

    Args:
        channels: Dictionary of channel instances, or None to check factory instances
        tenant_id: Tenant ID for factory lookup

    Returns:
        Dictionary with health status for each channel
    """
    if channels:
        # Check provided channels
        health_results = {}
        for channel_type, channel in channels.items():
            try:
                is_healthy = await channel.health_check()
                health_results[channel_type] = is_healthy
            except Exception as e:
                health_results[channel_type] = False
        return health_results
    else:
        # Use factory to check all instances
        return await channel_factory.health_check_all_channels(tenant_id)


def get_channel_metrics(channels: dict = None, tenant_id: str = None) -> dict:
    """
    Get metrics for channels.

    Args:
        channels: Dictionary of channel instances, or None to check factory instances
        tenant_id: Tenant ID for factory lookup

    Returns:
        Dictionary with metrics for each channel
    """
    if channels:
        # Get metrics from provided channels
        metrics_results = {}
        for channel_type, channel in channels.items():
            try:
                metrics = channel.get_metrics()
                metrics_results[channel_type] = metrics.dict()
            except Exception as e:
                metrics_results[channel_type] = {"error": str(e)}
        return metrics_results
    else:
        # Use factory to get all metrics
        return channel_factory.get_channel_metrics(tenant_id)


def get_channel_capabilities() -> dict:
    """
    Get capabilities of all registered channels.

    Returns:
        Dictionary mapping channel types to capabilities
    """
    from src.models.types import ChannelType

    capabilities = {}

    for channel_type in channel_factory.get_registered_channels():
        try:
            # Get default config to understand capabilities
            default_config = get_default_channel_config(channel_type.value)

            capabilities[channel_type.value] = {
                "supported_message_types": default_config.get("supported_message_types", []),
                "max_message_length": default_config.get("max_message_length", 0),
                "supports_rich_media": default_config.get("supports_rich_media", False),
                "supports_buttons": default_config.get("supports_buttons", False),
                "supports_quick_replies": default_config.get("supports_quick_replies", False),
                "supports_carousel": default_config.get("supports_carousel", False),
                "features": default_config.get("features", {})
            }
        except Exception as e:
            capabilities[channel_type.value] = {"error": str(e)}

    return capabilities


def reset_channel_factory():
    """Reset the global channel factory to initial state."""
    channel_factory._channel_instances.clear()
    channel_factory._channel_configs.clear()


# Channel registry for easy access
AVAILABLE_CHANNELS = {
    "web": WebChannel,
    "whatsapp": WhatsAppChannel,
    # Add more as they are implemented
    # "messenger": MessengerChannel,
    # "slack": SlackChannel,
    # "teams": TeamsChannel,
}

def get_available_channel_types() -> list:
    """Get list of available channel types."""
    return list(AVAILABLE_CHANNELS.keys())


def is_channel_available(channel_type: str) -> bool:
    """Check if a channel type is available."""
    return channel_type.lower() in AVAILABLE_CHANNELS


# Initialize the factory with built-in channels on import
_factory_initialized = False

def _ensure_factory_initialized():
    """Ensure the factory is initialized with built-in channels."""
    global _factory_initialized
    if not _factory_initialized:
        # Factory auto-initializes with built-in channels
        _factory_initialized = True

# Initialize on module import
_ensure_factory_initialized()