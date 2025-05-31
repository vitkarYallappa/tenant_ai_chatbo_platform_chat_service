"""
Channel factory for dynamic instantiation of channel implementations.

This module provides factory functions for creating channel instances
based on configuration and channel type.
"""

from typing import Dict, Type, Optional, Any, List
import structlog

from src.core.channels.base_channel import BaseChannel, ChannelConfig
from src.core.channels.web_channel import WebChannel
from src.core.channels.whatsapp_channel import WhatsAppChannel
from src.models.types import ChannelType
from src.core.exceptions import ChannelConfigurationError, ChannelError

logger = structlog.get_logger(__name__)


class ChannelFactory:
    """Factory for creating channel instances."""

    def __init__(self):
        self._channel_registry: Dict[ChannelType, Type[BaseChannel]] = {}
        self._channel_configs: Dict[str, ChannelConfig] = {}
        self._channel_instances: Dict[str, BaseChannel] = {}

        # Register built-in channel types
        self._register_builtin_channels()

        logger.info(
            "Channel factory initialized",
            registered_channels=list(self._channel_registry.keys())
        )

    def _register_builtin_channels(self) -> None:
        """Register built-in channel implementations."""
        self.register_channel(ChannelType.WEB, WebChannel)
        self.register_channel(ChannelType.WHATSAPP, WhatsAppChannel)

        # Placeholder registrations for other channels
        # These would be implemented in their respective files
        # self.register_channel(ChannelType.MESSENGER, MessengerChannel)
        # self.register_channel(ChannelType.SLACK, SlackChannel)
        # self.register_channel(ChannelType.TEAMS, TeamsChannel)

    def register_channel(
            self,
            channel_type: ChannelType,
            channel_class: Type[BaseChannel]
    ) -> None:
        """
        Register a channel implementation.

        Args:
            channel_type: Type of channel to register
            channel_class: Channel implementation class

        Raises:
            ValueError: If channel type is already registered
        """
        if channel_type in self._channel_registry:
            logger.warning(
                "Channel type already registered, overriding",
                channel_type=channel_type.value,
                existing_class=self._channel_registry[channel_type].__name__,
                new_class=channel_class.__name__
            )

        self._channel_registry[channel_type] = channel_class

        logger.info(
            "Channel registered",
            channel_type=channel_type.value,
            channel_class=channel_class.__name__
        )

    def unregister_channel(self, channel_type: ChannelType) -> None:
        """
        Unregister a channel implementation.

        Args:
            channel_type: Type of channel to unregister
        """
        if channel_type in self._channel_registry:
            del self._channel_registry[channel_type]
            logger.info(
                "Channel unregistered",
                channel_type=channel_type.value
            )

    def get_registered_channels(self) -> List[ChannelType]:
        """Get list of registered channel types."""
        return list(self._channel_registry.keys())

    def is_channel_registered(self, channel_type: ChannelType) -> bool:
        """Check if a channel type is registered."""
        return channel_type in self._channel_registry

    def create_channel(
            self,
            channel_type: ChannelType,
            config: Optional[ChannelConfig] = None,
            tenant_id: Optional[str] = None
    ) -> BaseChannel:
        """
        Create a channel instance.

        Args:
            channel_type: Type of channel to create
            config: Optional channel configuration
            tenant_id: Optional tenant ID for tenant-specific config

        Returns:
            Configured channel instance

        Raises:
            ChannelConfigurationError: If channel type not registered or config invalid
        """
        try:
            # Check if channel type is registered
            if channel_type not in self._channel_registry:
                available_channels = [ct.value for ct in self._channel_registry.keys()]
                raise ChannelConfigurationError(
                    channel=channel_type.value,
                    config_issue=f"Channel type not registered. Available: {available_channels}"
                )

            # Get configuration
            if config is None:
                config = self._get_default_config(channel_type, tenant_id)

            # Validate configuration
            self._validate_config(channel_type, config)

            # Create channel instance
            channel_class = self._channel_registry[channel_type]
            channel_instance = channel_class(config)

            logger.info(
                "Channel created",
                channel_type=channel_type.value,
                tenant_id=tenant_id,
                enabled=config.enabled
            )

            return channel_instance

        except Exception as e:
            logger.error(
                "Channel creation failed",
                channel_type=channel_type.value,
                tenant_id=tenant_id,
                error=str(e)
            )

            if isinstance(e, (ChannelConfigurationError, ChannelError)):
                raise

            raise ChannelConfigurationError(
                channel=channel_type.value,
                config_issue=f"Failed to create channel: {str(e)}"
            )

    def create_channels_from_config(
            self,
            tenant_configs: Dict[str, Dict[str, Any]],
            tenant_id: Optional[str] = None
    ) -> Dict[ChannelType, BaseChannel]:
        """
        Create multiple channels from configuration dictionary.

        Args:
            tenant_configs: Dictionary of channel configurations
            tenant_id: Optional tenant ID

        Returns:
            Dictionary mapping channel types to instances
        """
        channels = {}

        for channel_name, channel_config in tenant_configs.items():
            try:
                # Convert string to ChannelType enum
                channel_type = ChannelType(channel_name.lower())

                # Create ChannelConfig from dict
                config = ChannelConfig(**channel_config)

                # Create channel instance
                channel = self.create_channel(channel_type, config, tenant_id)
                channels[channel_type] = channel

            except ValueError as e:
                logger.warning(
                    "Skipping unknown channel type",
                    channel_name=channel_name,
                    error=str(e)
                )
                continue

            except Exception as e:
                logger.error(
                    "Failed to create channel from config",
                    channel_name=channel_name,
                    tenant_id=tenant_id,
                    error=str(e)
                )
                continue

        logger.info(
            "Channels created from config",
            tenant_id=tenant_id,
            created_channels=list(channels.keys()),
            total_channels=len(channels)
        )

        return channels

    def get_or_create_channel(
            self,
            channel_type: ChannelType,
            tenant_id: str,
            force_recreate: bool = False
    ) -> BaseChannel:
        """
        Get existing channel instance or create new one.

        Args:
            channel_type: Type of channel
            tenant_id: Tenant identifier
            force_recreate: Force recreation of channel

        Returns:
            Channel instance
        """
        instance_key = f"{tenant_id}:{channel_type.value}"

        # Return existing instance if available and not forcing recreation
        if not force_recreate and instance_key in self._channel_instances:
            return self._channel_instances[instance_key]

        # Create new instance
        channel = self.create_channel(channel_type, tenant_id=tenant_id)
        self._channel_instances[instance_key] = channel

        return channel

    def remove_channel_instance(self, channel_type: ChannelType, tenant_id: str) -> None:
        """Remove cached channel instance."""
        instance_key = f"{tenant_id}:{channel_type.value}"
        if instance_key in self._channel_instances:
            del self._channel_instances[instance_key]
            logger.debug(
                "Channel instance removed",
                channel_type=channel_type.value,
                tenant_id=tenant_id
            )

    def store_config(self, config_name: str, config: ChannelConfig) -> None:
        """Store a named configuration for reuse."""
        self._channel_configs[config_name] = config
        logger.debug(
            "Channel config stored",
            config_name=config_name,
            channel_type=config.channel_type.value
        )

    def get_stored_config(self, config_name: str) -> Optional[ChannelConfig]:
        """Retrieve a stored configuration."""
        return self._channel_configs.get(config_name)

    def _get_default_config(
            self,
            channel_type: ChannelType,
            tenant_id: Optional[str] = None
    ) -> ChannelConfig:
        """Get default configuration for channel type."""

        # Default configurations for each channel type
        default_configs = {
            ChannelType.WEB: {
                "channel_type": ChannelType.WEB,
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

            ChannelType.WHATSAPP: {
                "channel_type": ChannelType.WHATSAPP,
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

            ChannelType.MESSENGER: {
                "channel_type": ChannelType.MESSENGER,
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

            ChannelType.SLACK: {
                "channel_type": ChannelType.SLACK,
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

            ChannelType.TEAMS: {
                "channel_type": ChannelType.TEAMS,
                "enabled": False,  # Disabled by default
                "max_message_length": 28000,
                "supported_message_types": ["text", "image", "file"],
                "supports_rich_media": True,
                "supports_buttons": True,
                "supports_quick_replies": False,
                "features": {
                    "app_id": None,
                    "app_password": None,
                    "tenant_id": tenant_id
                }
            }
        }

        config_data = default_configs.get(channel_type, {})
        if not config_data:
            raise ChannelConfigurationError(
                channel=channel_type.value,
                config_issue="No default configuration available"
            )

        return ChannelConfig(**config_data)

    def _validate_config(self, channel_type: ChannelType, config: ChannelConfig) -> None:
        """Validate channel configuration."""

        # Basic validation
        if config.channel_type != channel_type:
            raise ChannelConfigurationError(
                channel=channel_type.value,
                config_issue=f"Config channel type mismatch: expected {channel_type.value}, got {config.channel_type.value}"
            )

        # Channel-specific validation
        if channel_type == ChannelType.WHATSAPP:
            self._validate_whatsapp_config(config)
        elif channel_type == ChannelType.WEB:
            self._validate_web_config(config)
        elif channel_type == ChannelType.MESSENGER:
            self._validate_messenger_config(config)
        elif channel_type == ChannelType.SLACK:
            self._validate_slack_config(config)
        elif channel_type == ChannelType.TEAMS:
            self._validate_teams_config(config)

    def _validate_whatsapp_config(self, config: ChannelConfig) -> None:
        """Validate WhatsApp-specific configuration."""
        if config.enabled:
            if not config.api_token:
                raise ChannelConfigurationError(
                    channel="whatsapp",
                    config_issue="API token is required for WhatsApp channel"
                )

            if not config.features.get("phone_number_id"):
                raise ChannelConfigurationError(
                    channel="whatsapp",
                    config_issue="Phone number ID is required for WhatsApp channel"
                )

    def _validate_web_config(self, config: ChannelConfig) -> None:
        """Validate Web-specific configuration."""
        # Web channel has minimal required configuration
        pass

    def _validate_messenger_config(self, config: ChannelConfig) -> None:
        """Validate Messenger-specific configuration."""
        if config.enabled:
            if not config.features.get("page_access_token"):
                raise ChannelConfigurationError(
                    channel="messenger",
                    config_issue="Page access token is required for Messenger channel"
                )

    def _validate_slack_config(self, config: ChannelConfig) -> None:
        """Validate Slack-specific configuration."""
        if config.enabled:
            if not config.features.get("bot_token"):
                raise ChannelConfigurationError(
                    channel="slack",
                    config_issue="Bot token is required for Slack channel"
                )

    def _validate_teams_config(self, config: ChannelConfig) -> None:
        """Validate Teams-specific configuration."""
        if config.enabled:
            if not config.features.get("app_id") or not config.features.get("app_password"):
                raise ChannelConfigurationError(
                    channel="teams",
                    config_issue="App ID and password are required for Teams channel"
                )

    async def health_check_all_channels(
            self,
            tenant_id: Optional[str] = None
    ) -> Dict[ChannelType, bool]:
        """
        Perform health checks on all active channel instances.

        Args:
            tenant_id: Optional tenant ID to filter instances

        Returns:
            Dictionary mapping channel types to health status
        """
        health_results = {}

        # Filter instances by tenant if specified
        instances_to_check = {}
        if tenant_id:
            for key, instance in self._channel_instances.items():
                if key.startswith(f"{tenant_id}:"):
                    channel_type = ChannelType(key.split(":", 1)[1])
                    instances_to_check[channel_type] = instance
        else:
            # Check all instances
            for key, instance in self._channel_instances.items():
                channel_type = ChannelType(key.split(":", 1)[1])
                instances_to_check[channel_type] = instance

        # Perform health checks
        for channel_type, instance in instances_to_check.items():
            try:
                is_healthy = await instance.health_check()
                health_results[channel_type] = is_healthy

                logger.debug(
                    "Channel health check completed",
                    channel_type=channel_type.value,
                    healthy=is_healthy
                )

            except Exception as e:
                health_results[channel_type] = False
                logger.error(
                    "Channel health check failed",
                    channel_type=channel_type.value,
                    error=str(e)
                )

        return health_results

    def get_channel_metrics(
            self,
            tenant_id: Optional[str] = None
    ) -> Dict[ChannelType, Dict[str, Any]]:
        """
        Get metrics for all active channel instances.

        Args:
            tenant_id: Optional tenant ID to filter instances

        Returns:
            Dictionary mapping channel types to metrics
        """
        metrics_results = {}

        # Filter instances by tenant if specified
        instances_to_check = {}
        if tenant_id:
            for key, instance in self._channel_instances.items():
                if key.startswith(f"{tenant_id}:"):
                    channel_type = ChannelType(key.split(":", 1)[1])
                    instances_to_check[channel_type] = instance
        else:
            # Get metrics for all instances
            for key, instance in self._channel_instances.items():
                channel_type = ChannelType(key.split(":", 1)[1])
                instances_to_check[channel_type] = instance

        # Collect metrics
        for channel_type, instance in instances_to_check.items():
            try:
                metrics = instance.get_metrics()
                metrics_results[channel_type] = metrics.dict()

            except Exception as e:
                logger.error(
                    "Failed to get channel metrics",
                    channel_type=channel_type.value,
                    error=str(e)
                )
                metrics_results[channel_type] = {"error": str(e)}

        return metrics_results


# Global factory instance
channel_factory = ChannelFactory()


# Convenience functions
def create_channel(
        channel_type: ChannelType,
        config: Optional[ChannelConfig] = None,
        tenant_id: Optional[str] = None
) -> BaseChannel:
    """Create a channel using the global factory."""
    return channel_factory.create_channel(channel_type, config, tenant_id)


def get_or_create_channel(
        channel_type: ChannelType,
        tenant_id: str,
        force_recreate: bool = False
) -> BaseChannel:
    """Get or create a channel using the global factory."""
    return channel_factory.get_or_create_channel(channel_type, tenant_id, force_recreate)


def get_registered_channels() -> List[ChannelType]:
    """Get list of registered channel types."""
    return channel_factory.get_registered_channels()


def register_custom_channel(
        channel_type: ChannelType,
        channel_class: Type[BaseChannel]
) -> None:
    """Register a custom channel implementation."""
    channel_factory.register_channel(channel_type, channel_class)