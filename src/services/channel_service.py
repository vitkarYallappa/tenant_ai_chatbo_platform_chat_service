"""
Channel Service

Manages channel abstraction and routing for message delivery across different platforms.
Handles channel configuration, health monitoring, and message routing logic.
"""

from datetime import datetime, timedelta
from typing import Dict, Any, Optional, List
from uuid import uuid4

from src.services.base_service import BaseService
from src.services.exceptions import ServiceError, ValidationError, NotFoundError, ExternalServiceError
from src.models.types import ChannelType, MessageContent, TenantId
from src.core.channels.channel_factory import ChannelFactory
from src.core.channels.base_channel import BaseChannel, ChannelResponse


class ChannelService(BaseService):
    """Service for managing channels and message delivery"""

    def __init__(self, channel_factory: ChannelFactory):
        super().__init__()
        self.channel_factory = channel_factory
        self._channel_health_cache = {}
        self._channel_config_cache = {}

    async def send_message(
            self,
            tenant_id: TenantId,
            channel_type: ChannelType,
            recipient: str,
            content: MessageContent,
            metadata: Optional[Dict[str, Any]] = None,
            user_context: Optional[Dict[str, Any]] = None
    ) -> ChannelResponse:
        """
        Send message through specified channel

        Args:
            tenant_id: Tenant identifier
            channel_type: Type of channel to use
            recipient: Message recipient identifier
            content: Message content to send
            metadata: Optional channel-specific metadata
            user_context: User authentication context

        Returns:
            Channel response with delivery information
        """
        try:
            await self.validate_tenant_access(tenant_id, user_context)

            # Get channel implementation
            channel = await self._get_channel_for_tenant(tenant_id, channel_type)

            # Validate channel is healthy
            await self._check_channel_health(channel_type, tenant_id)

            # Send message
            response = await channel.send_message(
                recipient=recipient,
                content=content,
                metadata=metadata or {}
            )

            self.log_operation(
                "send_message",
                tenant_id=tenant_id,
                channel=channel_type.value,
                recipient=recipient,
                message_type=content.type,
                success=response.success
            )

            return response

        except Exception as e:
            error = self.handle_service_error(
                e, "send_message",
                tenant_id=tenant_id,
                channel=channel_type.value
            )
            raise error

    async def process_webhook(
            self,
            tenant_id: TenantId,
            channel_type: ChannelType,
            webhook_data: Dict[str, Any],
            user_context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Process incoming webhook from channel

        Args:
            tenant_id: Tenant identifier
            channel_type: Type of channel
            webhook_data: Webhook payload data
            user_context: User authentication context

        Returns:
            Processed webhook result
        """
        try:
            await self.validate_tenant_access(tenant_id, user_context)

            # Get channel implementation
            channel = await self._get_channel_for_tenant(tenant_id, channel_type)

            # Process webhook
            result = await channel.process_webhook(webhook_data)

            self.log_operation(
                "process_webhook",
                tenant_id=tenant_id,
                channel=channel_type.value,
                event_count=len(result.get("events", []))
            )

            return result

        except Exception as e:
            error = self.handle_service_error(
                e, "process_webhook",
                tenant_id=tenant_id,
                channel=channel_type.value
            )
            raise error

    async def get_channel_health(
            self,
            tenant_id: TenantId,
            channel_type: ChannelType,
            user_context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Get channel health status

        Args:
            tenant_id: Tenant identifier
            channel_type: Type of channel
            user_context: User authentication context

        Returns:
            Channel health information
        """
        try:
            await self.validate_tenant_access(tenant_id, user_context)

            # Get channel implementation
            channel = await self._get_channel_for_tenant(tenant_id, channel_type)

            # Check health
            health_status = await channel.check_health()

            # Cache health status
            cache_key = f"{tenant_id}:{channel_type.value}"
            self._channel_health_cache[cache_key] = {
                "status": health_status,
                "checked_at": datetime.utcnow(),
                "ttl": 300  # 5 minutes
            }

            self.log_operation(
                "get_channel_health",
                tenant_id=tenant_id,
                channel=channel_type.value,
                health_status=health_status.get("status")
            )

            return health_status

        except Exception as e:
            error = self.handle_service_error(
                e, "get_channel_health",
                tenant_id=tenant_id,
                channel=channel_type.value
            )
            raise error

    async def configure_channel(
            self,
            tenant_id: TenantId,
            channel_type: ChannelType,
            configuration: Dict[str, Any],
            user_context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Configure channel for tenant

        Args:
            tenant_id: Tenant identifier
            channel_type: Type of channel to configure
            configuration: Channel configuration
            user_context: User authentication context

        Returns:
            Configuration result
        """
        try:
            await self.validate_tenant_access(tenant_id, user_context)

            # Validate configuration
            await self._validate_channel_configuration(channel_type, configuration)

            # Get channel implementation
            channel = await self._get_channel_for_tenant(tenant_id, channel_type)

            # Apply configuration
            config_result = await channel.configure(configuration)

            # Cache configuration
            cache_key = f"{tenant_id}:{channel_type.value}"
            self._channel_config_cache[cache_key] = {
                "configuration": configuration,
                "configured_at": datetime.utcnow(),
                "version": config_result.get("version", "1.0")
            }

            self.log_operation(
                "configure_channel",
                tenant_id=tenant_id,
                channel=channel_type.value,
                config_keys=list(configuration.keys())
            )

            return config_result

        except Exception as e:
            error = self.handle_service_error(
                e, "configure_channel",
                tenant_id=tenant_id,
                channel=channel_type.value
            )
            raise error

    async def list_available_channels(
            self,
            tenant_id: TenantId,
            user_context: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """
        List available channels for tenant

        Args:
            tenant_id: Tenant identifier
            user_context: User authentication context

        Returns:
            List of available channel information
        """
        try:
            await self.validate_tenant_access(tenant_id, user_context)

            # Get tenant channel configuration (from database/config)
            tenant_config = await self._get_tenant_configuration(tenant_id, "channels")
            enabled_channels = tenant_config.get("enabled_channels", [])

            available_channels = []

            for channel_type_str in enabled_channels:
                try:
                    channel_type = ChannelType(channel_type_str)
                    channel_info = await self._get_channel_info(tenant_id, channel_type)
                    available_channels.append(channel_info)
                except ValueError:
                    self.logger.warning(
                        "Invalid channel type in configuration",
                        tenant_id=tenant_id,
                        channel_type=channel_type_str
                    )
                    continue

            self.log_operation(
                "list_available_channels",
                tenant_id=tenant_id,
                channel_count=len(available_channels)
            )

            return available_channels

        except Exception as e:
            error = self.handle_service_error(
                e, "list_available_channels",
                tenant_id=tenant_id
            )
            raise error

    async def test_channel_connection(
            self,
            tenant_id: TenantId,
            channel_type: ChannelType,
            test_recipient: Optional[str] = None,
            user_context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Test channel connection and configuration

        Args:
            tenant_id: Tenant identifier
            channel_type: Type of channel to test
            test_recipient: Optional test recipient
            user_context: User authentication context

        Returns:
            Test result information
        """
        try:
            await self.validate_tenant_access(tenant_id, user_context)

            # Get channel implementation
            channel = await self._get_channel_for_tenant(tenant_id, channel_type)

            # Perform connection test
            test_result = await channel.test_connection()

            # Optionally send test message
            if test_recipient and test_result.get("connection_ok"):
                test_message = MessageContent(
                    type="text",
                    text="This is a test message from your chatbot platform."
                )

                try:
                    send_result = await channel.send_message(
                        recipient=test_recipient,
                        content=test_message,
                        metadata={"test_message": True}
                    )
                    test_result["test_message_sent"] = send_result.success
                    test_result["test_message_id"] = send_result.message_id
                except Exception as e:
                    test_result["test_message_sent"] = False
                    test_result["test_message_error"] = str(e)

            self.log_operation(
                "test_channel_connection",
                tenant_id=tenant_id,
                channel=channel_type.value,
                connection_ok=test_result.get("connection_ok"),
                test_recipient=test_recipient
            )

            return test_result

        except Exception as e:
            error = self.handle_service_error(
                e, "test_channel_connection",
                tenant_id=tenant_id,
                channel=channel_type.value
            )
            raise error

    async def get_channel_metrics(
            self,
            tenant_id: TenantId,
            channel_type: ChannelType,
            start_date: Optional[datetime] = None,
            end_date: Optional[datetime] = None,
            user_context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Get channel performance metrics

        Args:
            tenant_id: Tenant identifier
            channel_type: Type of channel
            start_date: Optional start date for metrics
            end_date: Optional end date for metrics
            user_context: User authentication context

        Returns:
            Channel metrics data
        """
        try:
            await self.validate_tenant_access(tenant_id, user_context)

            # Set default date range if not provided
            if not end_date:
                end_date = datetime.utcnow()
            if not start_date:
                start_date = end_date - timedelta(days=7)

            # Get channel implementation
            channel = await self._get_channel_for_tenant(tenant_id, channel_type)

            # Get metrics from channel
            metrics = await channel.get_metrics(start_date, end_date)

            self.log_operation(
                "get_channel_metrics",
                tenant_id=tenant_id,
                channel=channel_type.value,
                start_date=start_date.isoformat(),
                end_date=end_date.isoformat()
            )

            return metrics

        except Exception as e:
            error = self.handle_service_error(
                e, "get_channel_metrics",
                tenant_id=tenant_id,
                channel=channel_type.value
            )
            raise error

    async def _get_channel_for_tenant(
            self,
            tenant_id: TenantId,
            channel_type: ChannelType
    ) -> BaseChannel:
        """Get configured channel instance for tenant"""
        try:
            # Get base channel from factory
            channel = await self.channel_factory.get_channel(channel_type)

            # Apply tenant-specific configuration if needed
            tenant_config = await self._get_tenant_configuration(
                tenant_id, f"channel_{channel_type.value}"
            )

            if tenant_config:
                await channel.configure(tenant_config)

            return channel

        except Exception as e:
            raise ServiceError(f"Failed to get channel for tenant: {e}")

    async def _check_channel_health(
            self,
            channel_type: ChannelType,
            tenant_id: TenantId
    ) -> bool:
        """Check if channel is healthy"""
        try:
            cache_key = f"{tenant_id}:{channel_type.value}"

            # Check cache first
            if cache_key in self._channel_health_cache:
                cached = self._channel_health_cache[cache_key]
                cache_age = (datetime.utcnow() - cached["checked_at"]).total_seconds()

                if cache_age < cached["ttl"]:
                    status = cached["status"]
                    if status.get("status") != "healthy":
                        raise ExternalServiceError(
                            f"Channel {channel_type.value} is not healthy: {status.get('error')}"
                        )
                    return True

            # Perform fresh health check
            health_status = await self.get_channel_health(tenant_id, channel_type)

            if health_status.get("status") != "healthy":
                raise ExternalServiceError(
                    f"Channel {channel_type.value} is not healthy: {health_status.get('error')}"
                )

            return True

        except ExternalServiceError:
            raise
        except Exception as e:
            self.logger.warning(
                "Health check failed, allowing request",
                channel=channel_type.value,
                tenant_id=tenant_id,
                error=str(e)
            )
            return True  # Fail open

    async def _validate_channel_configuration(
            self,
            channel_type: ChannelType,
            configuration: Dict[str, Any]
    ) -> None:
        """Validate channel configuration"""
        required_fields = {
            ChannelType.WHATSAPP: ["api_token", "phone_number_id"],
            ChannelType.MESSENGER: ["page_access_token", "app_secret"],
            ChannelType.SLACK: ["bot_token", "signing_secret"],
            ChannelType.TEAMS: ["app_id", "app_password"],
            ChannelType.WEB: [],  # Web channel has minimal requirements
        }

        required = required_fields.get(channel_type, [])
        missing = [field for field in required if field not in configuration]

        if missing:
            raise ValidationError(
                f"Missing required configuration fields for {channel_type.value}: {missing}"
            )

    async def _get_channel_info(
            self,
            tenant_id: TenantId,
            channel_type: ChannelType
    ) -> Dict[str, Any]:
        """Get channel information for tenant"""
        try:
            # Get cached configuration
            cache_key = f"{tenant_id}:{channel_type.value}"
            config_info = self._channel_config_cache.get(cache_key, {})

            # Get health status
            health_info = self._channel_health_cache.get(cache_key, {})

            channel_info = {
                "type": channel_type.value,
                "name": channel_type.value.title(),
                "configured": bool(config_info),
                "health_status": health_info.get("status", {}).get("status", "unknown"),
                "last_health_check": health_info.get("checked_at").isoformat() if health_info.get(
                    "checked_at") else None,
                "configuration_version": config_info.get("version"),
                "configured_at": config_info.get("configured_at").isoformat() if config_info.get(
                    "configured_at") else None
            }

            return channel_info

        except Exception as e:
            return {
                "type": channel_type.value,
                "name": channel_type.value.title(),
                "configured": False,
                "health_status": "error",
                "error": str(e)
            }

    async def enable_channel(
            self,
            tenant_id: TenantId,
            channel_type: ChannelType,
            configuration: Dict[str, Any],
            user_context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Enable and configure a channel for tenant

        Args:
            tenant_id: Tenant identifier
            channel_type: Type of channel to enable
            configuration: Channel configuration
            user_context: User authentication context

        Returns:
            Enable operation result
        """
        try:
            await self.validate_tenant_access(tenant_id, user_context)

            # Configure the channel
            config_result = await self.configure_channel(
                tenant_id, channel_type, configuration, user_context
            )

            # Test the connection
            test_result = await self.test_channel_connection(
                tenant_id, channel_type, user_context=user_context
            )

            # Update tenant configuration to include this channel
            # This would typically update the database

            result = {
                "channel_type": channel_type.value,
                "enabled": True,
                "configuration_status": "success" if config_result else "failed",
                "connection_test": test_result.get("connection_ok", False),
                "enabled_at": datetime.utcnow().isoformat()
            }

            self.log_operation(
                "enable_channel",
                tenant_id=tenant_id,
                channel=channel_type.value,
                success=result["connection_test"]
            )

            return result

        except Exception as e:
            error = self.handle_service_error(
                e, "enable_channel",
                tenant_id=tenant_id,
                channel=channel_type.value
            )
            raise error

    async def disable_channel(
            self,
            tenant_id: TenantId,
            channel_type: ChannelType,
            user_context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Disable a channel for tenant

        Args:
            tenant_id: Tenant identifier
            channel_type: Type of channel to disable
            user_context: User authentication context

        Returns:
            Disable operation result
        """
        try:
            await self.validate_tenant_access(tenant_id, user_context)

            # Remove from caches
            cache_key = f"{tenant_id}:{channel_type.value}"
            self._channel_config_cache.pop(cache_key, None)
            self._channel_health_cache.pop(cache_key, None)

            # Update tenant configuration to remove this channel
            # This would typically update the database

            result = {
                "channel_type": channel_type.value,
                "enabled": False,
                "disabled_at": datetime.utcnow().isoformat()
            }

            self.log_operation(
                "disable_channel",
                tenant_id=tenant_id,
                channel=channel_type.value
            )

            return result

        except Exception as e:
            error = self.handle_service_error(
                e, "disable_channel",
                tenant_id=tenant_id,
                channel=channel_type.value
            )
            raise error