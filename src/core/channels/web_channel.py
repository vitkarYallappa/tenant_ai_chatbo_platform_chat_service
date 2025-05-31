"""
Web channel implementation for browser-based chat interfaces.

This module handles web-specific message processing, formatting, and delivery
through WebSocket connections and HTTP APIs.
"""

import json
import asyncio
from typing import Dict, Any, Optional, List, Set
from datetime import datetime
from dataclasses import dataclass

from src.core.channels.base_channel import BaseChannel, ChannelConfig, ChannelResponse
from src.models.types import MessageContent, MessageType, ChannelType, DeliveryStatus
from src.core.exceptions import ChannelError, ValidationError, ChannelDeliveryError


@dataclass
class WebSocketConnection:
    """Represents an active WebSocket connection."""
    connection_id: str
    user_id: str
    session_id: str
    conversation_id: Optional[str]
    connected_at: datetime
    last_activity: datetime
    metadata: Dict[str, Any]


class WebChannel(BaseChannel):
    """Web channel implementation for browser-based chat."""

    def __init__(self, config: ChannelConfig):
        super().__init__(config)

        # Web-specific configuration
        self.websocket_enabled = config.features.get("websocket_enabled", True)
        self.typing_indicators_enabled = config.features.get("typing_indicators", True)
        self.message_history_enabled = config.features.get("message_history", True)
        self.file_upload_enabled = config.features.get("file_upload", True)
        self.max_file_size_mb = config.features.get("max_file_size_mb", 10)

        # Connection management
        self.active_connections: Dict[str, WebSocketConnection] = {}
        self.user_connections: Dict[str, Set[str]] = {}  # user_id -> connection_ids

        # Message delivery tracking
        self.message_delivery_callbacks: Dict[str, asyncio.Event] = {}

        # Supported web features
        self.supported_features = {
            "rich_text": True,
            "emojis": True,
            "markdown": config.features.get("markdown_support", True),
            "html": config.features.get("html_support", False),
            "file_attachments": self.file_upload_enabled,
            "typing_indicators": self.typing_indicators_enabled,
            "read_receipts": config.features.get("read_receipts", True),
            "message_reactions": config.features.get("message_reactions", False),
            "link_previews": config.features.get("link_previews", True)
        }

        self.logger.info(
            "Web channel initialized",
            websocket_enabled=self.websocket_enabled,
            supported_features=list(self.supported_features.keys())
        )

    @property
    def channel_type(self) -> ChannelType:
        return ChannelType.WEB

    async def send_message(
            self,
            recipient: str,
            content: MessageContent,
            metadata: Optional[Dict[str, Any]] = None
    ) -> ChannelResponse:
        """Send message through web channel."""
        start_time = datetime.utcnow()

        try:
            # For web channel, recipient is typically a user_id or session_id
            if not await self.validate_recipient(recipient):
                return self._create_error_response(
                    "INVALID_RECIPIENT",
                    f"Invalid web recipient: {recipient}",
                    recipient,
                    is_retryable=False,
                    processing_time_ms=self._calculate_processing_time(start_time)
                )

            # Validate content
            if not await self.validate_content(content):
                return self._create_error_response(
                    "INVALID_CONTENT",
                    "Message content is not valid for web channel",
                    recipient,
                    is_retryable=False,
                    processing_time_ms=self._calculate_processing_time(start_time)
                )

            # Format message for web display
            formatted_message = await self.format_message(content)

            # Add web-specific metadata
            web_metadata = {
                "timestamp": datetime.utcnow().isoformat(),
                "channel": "web",
                "supports_markdown": self.supported_features["markdown"],
                "supports_html": self.supported_features["html"],
                **(metadata or {})
            }

            # Attempt delivery through various methods
            delivery_success = False
            platform_message_id = f"web_{int(datetime.utcnow().timestamp() * 1000)}"

            # Try WebSocket delivery first
            if self.websocket_enabled:
                delivery_success = await self._deliver_via_websocket(
                    recipient, formatted_message, web_metadata
                )

            # Fallback to HTTP callback if WebSocket fails
            if not delivery_success:
                delivery_success = await self._deliver_via_http_callback(
                    recipient, formatted_message, web_metadata
                )

            # Store message for retrieval if enabled
            if self.message_history_enabled:
                await self._store_message_for_retrieval(
                    recipient, platform_message_id, formatted_message, web_metadata
                )

            processing_time = self._calculate_processing_time(start_time)

            if delivery_success:
                self.update_metrics(True, processing_time)

                self.logger.info(
                    "Web message delivered successfully",
                    recipient=recipient,
                    message_type=content.type.value,
                    processing_time_ms=processing_time
                )

                return self._create_success_response(
                    message_id=platform_message_id,
                    platform_message_id=platform_message_id,
                    recipient=recipient,
                    metadata={
                        "delivery_method": "websocket" if self.websocket_enabled else "http_callback",
                        "formatted_message": formatted_message,
                        **web_metadata
                    },
                    processing_time_ms=processing_time
                )
            else:
                self.update_metrics(False, processing_time, "DELIVERY_FAILED")

                return self._create_error_response(
                    "DELIVERY_FAILED",
                    "Failed to deliver message through any available method",
                    recipient,
                    is_retryable=True,
                    processing_time_ms=processing_time
                )

        except Exception as e:
            processing_time = self._calculate_processing_time(start_time)
            self.update_metrics(False, processing_time, "SEND_FAILED")

            self.logger.error(
                "Web message send failed",
                recipient=recipient,
                error=str(e),
                error_type=type(e).__name__
            )

            return self._create_error_response(
                "SEND_FAILED",
                f"Failed to send web message: {str(e)}",
                recipient,
                is_retryable=True,
                processing_time_ms=processing_time
            )

    async def validate_recipient(self, recipient: str) -> bool:
        """Validate web recipient identifier."""
        try:
            # Web recipients can be user IDs, session IDs, or connection IDs
            if not recipient or len(recipient.strip()) == 0:
                return False

            # Check for valid format (basic validation)
            if len(recipient) < 3 or len(recipient) > 255:
                return False

            # Additional validation based on recipient type
            if recipient.startswith("user_"):
                return len(recipient) > 5
            elif recipient.startswith("session_"):
                return len(recipient) > 8
            elif recipient.startswith("conn_"):
                return len(recipient) > 5
            else:
                # Generic recipient - allow any reasonable string
                return True

        except Exception as e:
            self.logger.error(
                "Web recipient validation failed",
                recipient=recipient,
                error=str(e)
            )
            return False

    async def format_message(self, content: MessageContent) -> Dict[str, Any]:
        """Format message content for web display."""
        try:
            message = {
                "type": content.type.value,
                "content": {},
                "features": self.supported_features.copy(),
                "formatting": {
                    "supports_markdown": self.supported_features["markdown"],
                    "supports_html": self.supported_features["html"],
                    "supports_emojis": self.supported_features["emojis"]
                }
            }

            if content.type == MessageType.TEXT:
                message["content"] = await self._format_text_content(content)

            elif content.type in [MessageType.IMAGE, MessageType.VIDEO, MessageType.AUDIO, MessageType.FILE]:
                message["content"] = await self._format_media_content(content)

            elif content.type == MessageType.LOCATION:
                message["content"] = await self._format_location_content(content)

            else:
                # Handle other message types generically
                message["content"] = {
                    "text": content.text or "",
                    "raw_content": content.dict()
                }

            # Add interactive elements
            if content.buttons:
                message["buttons"] = await self._format_buttons(content.buttons)

            if content.quick_replies:
                message["quick_replies"] = await self._format_quick_replies(content.quick_replies)

            return message

        except Exception as e:
            self.logger.error(
                "Web message formatting failed",
                message_type=content.type.value,
                error=str(e)
            )
            raise ChannelError(f"Failed to format web message: {e}")

    async def _format_text_content(self, content: MessageContent) -> Dict[str, Any]:
        """Format text content for web display."""
        if not content.text:
            raise ValidationError(
                field="text",
                value="",
                validation_rule="Text content is required for text messages"
            )

        formatted_content = {
            "text": content.text,
            "original_text": content.text
        }

        # Apply markdown formatting if enabled
        if self.supported_features["markdown"]:
            formatted_content["markdown"] = content.text
            formatted_content["html"] = await self._convert_markdown_to_html(content.text)

        # Add language information
        if hasattr(content, 'language') and content.language:
            formatted_content["language"] = content.language

        # Add formatting hints
        formatted_content["formatting_hints"] = {
            "preserve_whitespace": True,
            "auto_link": self.supported_features["link_previews"],
            "emoji_support": self.supported_features["emojis"]
        }

        return formatted_content

    async def _format_media_content(self, content: MessageContent) -> Dict[str, Any]:
        """Format media content for web display."""
        if not content.media:
            raise ValidationError(
                field="media",
                value="None",
                validation_rule="Media content is required for media messages"
            )

        # Check file size limits
        max_size = self.max_file_size_mb * 1024 * 1024
        if content.media.size_bytes > max_size:
            raise ValidationError(
                field="media_size",
                value=content.media.size_bytes,
                validation_rule=f"File too large for web upload",
                expected_format=f"Max {self.max_file_size_mb}MB"
            )

        formatted_content = {
            "url": content.media.url,
            "type": content.media.type,
            "size_bytes": content.media.size_bytes,
            "alt_text": getattr(content.media, 'alt_text', ''),
            "filename": getattr(content.media, 'filename', ''),
            "thumbnail_url": getattr(content.media, 'thumbnail_url', ''),
            "download_enabled": True,
            "preview_enabled": content.type in [MessageType.IMAGE, MessageType.VIDEO]
        }

        # Add caption if present
        if content.text:
            formatted_content["caption"] = content.text

        # Add web-specific display properties
        if content.type == MessageType.IMAGE:
            formatted_content["display_properties"] = {
                "max_width": "100%",
                "max_height": "400px",
                "responsive": True
            }
        elif content.type == MessageType.VIDEO:
            formatted_content["display_properties"] = {
                "controls": True,
                "autoplay": False,
                "muted": True,
                "max_width": "100%"
            }
        elif content.type == MessageType.AUDIO:
            formatted_content["display_properties"] = {
                "controls": True,
                "autoplay": False
            }

        return formatted_content

    async def _format_location_content(self, content: MessageContent) -> Dict[str, Any]:
        """Format location content for web display."""
        if not content.location:
            raise ValidationError(
                field="location",
                value="None",
                validation_rule="Location content is required for location messages"
            )

        formatted_content = {
            "latitude": content.location.latitude,
            "longitude": content.location.longitude,
            "address": getattr(content.location, 'address', ''),
            "map_enabled": True,
            "map_provider": "google",  # or "openstreetmap", configurable
            "zoom_level": 15
        }

        # Add map display properties
        formatted_content["display_properties"] = {
            "width": "100%",
            "height": "200px",
            "interactive": True,
            "show_marker": True
        }

        return formatted_content

    async def _format_buttons(self, buttons: List[Any]) -> List[Dict[str, Any]]:
        """Format buttons for web display."""
        formatted_buttons = []

        for button in buttons[:10]:  # Limit to 10 buttons for UI reasons
            formatted_button = {
                "type": button.type,
                "title": button.title,
                "payload": getattr(button, 'payload', ''),
                "url": getattr(button, 'url', ''),
                "style": "primary" if button.type == "postback" else "secondary"
            }

            # Add web-specific styling
            if button.type == "url":
                formatted_button["target"] = "_blank"
                formatted_button["rel"] = "noopener noreferrer"

            formatted_buttons.append(formatted_button)

        return formatted_buttons

    async def _format_quick_replies(self, quick_replies: List[Any]) -> List[Dict[str, Any]]:
        """Format quick replies for web display."""
        formatted_replies = []

        for qr in quick_replies[:15]:  # Limit for UI reasons
            formatted_reply = {
                "title": qr.title,
                "payload": qr.payload,
                "content_type": getattr(qr, 'content_type', 'text'),
                "style": "outline"
            }
            formatted_replies.append(formatted_reply)

        return formatted_replies

    async def _convert_markdown_to_html(self, text: str) -> str:
        """Convert markdown text to HTML (basic implementation)."""
        try:
            # Basic markdown conversion - use a proper library in production
            import re

            html = text

            # Bold text
            html = re.sub(r'\*\*(.*?)\*\*', r'<strong>\1</strong>', html)
            html = re.sub(r'\*(.*?)\*', r'<em>\1</em>', html)

            # Links
            html = re.sub(r'\[([^\]]+)\]\(([^)]+)\)', r'<a href="\2" target="_blank">\1</a>', html)

            # Line breaks
            html = html.replace('\n', '<br>')

            return html

        except Exception as e:
            self.logger.warning(
                "Markdown to HTML conversion failed",
                error=str(e),
                text_length=len(text)
            )
            return text

    async def _deliver_via_websocket(
            self,
            recipient: str,
            message: Dict[str, Any],
            metadata: Dict[str, Any]
    ) -> bool:
        """Deliver message via WebSocket connection."""
        try:
            # Find active connections for recipient
            connection_ids = self._get_connections_for_recipient(recipient)

            if not connection_ids:
                self.logger.debug(
                    "No active WebSocket connections for recipient",
                    recipient=recipient
                )
                return False

            # Prepare WebSocket message
            ws_message = {
                "type": "message",
                "data": message,
                "metadata": metadata,
                "timestamp": datetime.utcnow().isoformat()
            }

            # Send to all active connections
            successful_deliveries = 0

            for connection_id in connection_ids:
                try:
                    # This would integrate with your WebSocket manager
                    # await self.websocket_manager.send_to_connection(connection_id, ws_message)
                    successful_deliveries += 1

                    self.logger.debug(
                        "Message sent via WebSocket",
                        connection_id=connection_id,
                        recipient=recipient
                    )

                except Exception as e:
                    self.logger.warning(
                        "WebSocket delivery failed for connection",
                        connection_id=connection_id,
                        error=str(e)
                    )
                    # Remove invalid connection
                    await self._remove_connection(connection_id)

            return successful_deliveries > 0

        except Exception as e:
            self.logger.error(
                "WebSocket delivery failed",
                recipient=recipient,
                error=str(e)
            )
            return False

    async def _deliver_via_http_callback(
            self,
            recipient: str,
            message: Dict[str, Any],
            metadata: Dict[str, Any]
    ) -> bool:
        """Deliver message via HTTP callback."""
        try:
            # This would implement HTTP callback delivery
            # For now, return True as a placeholder

            self.logger.debug(
                "HTTP callback delivery attempted",
                recipient=recipient
            )

            # In a real implementation, you would:
            # 1. Look up callback URL for recipient
            # 2. Send HTTP POST with message data
            # 3. Handle response and retries

            return True

        except Exception as e:
            self.logger.error(
                "HTTP callback delivery failed",
                recipient=recipient,
                error=str(e)
            )
            return False

    async def _store_message_for_retrieval(
            self,
            recipient: str,
            message_id: str,
            message: Dict[str, Any],
            metadata: Dict[str, Any]
    ) -> None:
        """Store message for later retrieval."""
        try:
            # This would store the message in a database or cache
            # for retrieval via API calls

            stored_message = {
                "message_id": message_id,
                "recipient": recipient,
                "message": message,
                "metadata": metadata,
                "stored_at": datetime.utcnow().isoformat()
            }

            # await self.message_store.store(stored_message)

            self.logger.debug(
                "Message stored for retrieval",
                message_id=message_id,
                recipient=recipient
            )

        except Exception as e:
            self.logger.warning(
                "Message storage failed",
                message_id=message_id,
                error=str(e)
            )

    def _get_connections_for_recipient(self, recipient: str) -> Set[str]:
        """Get active connection IDs for a recipient."""
        # This would query your connection management system
        # For now, return empty set as placeholder
        return set()

    async def _remove_connection(self, connection_id: str) -> None:
        """Remove invalid connection from tracking."""
        if connection_id in self.active_connections:
            connection = self.active_connections[connection_id]

            # Remove from user connections mapping
            if connection.user_id in self.user_connections:
                self.user_connections[connection.user_id].discard(connection_id)
                if not self.user_connections[connection.user_id]:
                    del self.user_connections[connection.user_id]

            # Remove from active connections
            del self.active_connections[connection_id]

            self.logger.debug(
                "Connection removed",
                connection_id=connection_id,
                user_id=connection.user_id
            )

    async def register_connection(
            self,
            connection_id: str,
            user_id: str,
            session_id: str,
            metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """Register a new WebSocket connection."""
        connection = WebSocketConnection(
            connection_id=connection_id,
            user_id=user_id,
            session_id=session_id,
            conversation_id=metadata.get('conversation_id') if metadata else None,
            connected_at=datetime.utcnow(),
            last_activity=datetime.utcnow(),
            metadata=metadata or {}
        )

        self.active_connections[connection_id] = connection

        # Track user connections
        if user_id not in self.user_connections:
            self.user_connections[user_id] = set()
        self.user_connections[user_id].add(connection_id)

        self.logger.info(
            "WebSocket connection registered",
            connection_id=connection_id,
            user_id=user_id,
            session_id=session_id
        )

    async def unregister_connection(self, connection_id: str) -> None:
        """Unregister a WebSocket connection."""
        await self._remove_connection(connection_id)

    def _calculate_processing_time(self, start_time: datetime) -> int:
        """Calculate processing time in milliseconds."""
        return int((datetime.utcnow() - start_time).total_seconds() * 1000)

    async def _channel_specific_health_check(self) -> bool:
        """Perform web-specific health checks."""
        try:
            # Check if WebSocket manager is responsive
            if self.websocket_enabled:
                # This would check your WebSocket infrastructure
                pass

            # Check message storage if enabled
            if self.message_history_enabled:
                # This would check your message storage system
                pass

            return True

        except Exception as e:
            self.logger.error(
                "Web channel health check failed",
                error=str(e)
            )
            return False