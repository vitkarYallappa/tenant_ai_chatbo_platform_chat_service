"""
WhatsApp Business API channel implementation.

This module handles WhatsApp-specific message sending and webhook processing
using the WhatsApp Cloud API.
"""

import re
import httpx
import hashlib
import hmac
from typing import Dict, Any, Optional, List
from datetime import datetime

from src.core.channels.base_channel import BaseChannel, ChannelConfig, ChannelResponse
from src.models.types import MessageContent, MessageType, ChannelType, DeliveryStatus, MediaContent
from src.core.exceptions import (
    ChannelError,
    ValidationError,
    ChannelConnectionError,
    ChannelRateLimitError,
    ChannelDeliveryError
)


class WhatsAppChannel(BaseChannel):
    """WhatsApp Business API channel implementation."""

    def __init__(self, config: ChannelConfig):
        super().__init__(config)

        # WhatsApp-specific configuration
        self.api_base_url = "https://graph.facebook.com/v18.0"
        self.phone_number_id = config.features.get("phone_number_id")
        self.business_account_id = config.features.get("business_account_id")
        self.verify_token = config.features.get("verify_token")

        if not self.phone_number_id:
            raise ValueError("WhatsApp phone_number_id is required in features config")

        if not config.api_token:
            raise ValueError("WhatsApp API token is required")

        # Configure HTTP client with connection pooling
        self.http_client = httpx.AsyncClient(
            timeout=config.timeout_seconds,
            limits=httpx.Limits(max_connections=10, max_keepalive_connections=5),
            headers={
                "Authorization": f"Bearer {config.api_token}",
                "Content-Type": "application/json",
                "User-Agent": "ChatbotPlatform/1.0 WhatsAppChannel"
            }
        )

        # WhatsApp-specific rate limits (Cloud API limits)
        self.rate_limits = {
            "messages_per_second": 80,
            "messages_per_day": 250000,  # Varies by tier
            "template_messages_per_day": 1000000  # Varies by tier
        }

        self.logger.info(
            "WhatsApp channel initialized",
            phone_number_id=self.phone_number_id,
            business_account_id=self.business_account_id
        )

    @property
    def channel_type(self) -> ChannelType:
        return ChannelType.WHATSAPP

    async def send_message(
            self,
            recipient: str,
            content: MessageContent,
            metadata: Optional[Dict[str, Any]] = None
    ) -> ChannelResponse:
        """Send message via WhatsApp Business API."""
        start_time = datetime.utcnow()
        processing_start = start_time

        try:
            # Check rate limits
            if not await self.check_rate_limit():
                return self._create_error_response(
                    "RATE_LIMIT_EXCEEDED",
                    "WhatsApp rate limit exceeded",
                    recipient,
                    is_retryable=True,
                    processing_time_ms=self._calculate_processing_time(processing_start)
                )

            # Validate recipient
            if not await self.validate_recipient(recipient):
                return self._create_error_response(
                    "INVALID_RECIPIENT",
                    f"Invalid WhatsApp phone number: {recipient}",
                    recipient,
                    is_retryable=False,
                    processing_time_ms=self._calculate_processing_time(processing_start)
                )

            # Validate content
            if not await self.validate_content(content):
                return self._create_error_response(
                    "INVALID_CONTENT",
                    "Message content is not valid for WhatsApp",
                    recipient,
                    is_retryable=False,
                    processing_time_ms=self._calculate_processing_time(processing_start)
                )

            # Format message for WhatsApp API
            message_payload = await self.format_message(content)
            message_payload["to"] = recipient

            # Add message context from metadata
            if metadata:
                if "reply_to_message_id" in metadata:
                    message_payload["context"] = {
                        "message_id": metadata["reply_to_message_id"]
                    }

            # Send message
            url = f"{self.api_base_url}/{self.phone_number_id}/messages"

            self.logger.debug(
                "Sending WhatsApp message",
                url=url,
                recipient=recipient,
                message_type=content.type.value
            )

            response = await self.http_client.post(url, json=message_payload)

            # Handle different HTTP status codes
            if response.status_code == 200:
                response_data = response.json()
                platform_message_id = None

                # Extract message ID
                if "messages" in response_data and response_data["messages"]:
                    platform_message_id = response_data["messages"][0].get("id")

                # Calculate processing time
                processing_time = self._calculate_processing_time(processing_start)

                # Update metrics
                self.update_metrics(True, processing_time)

                self.logger.info(
                    "WhatsApp message sent successfully",
                    recipient=recipient,
                    platform_message_id=platform_message_id,
                    processing_time_ms=processing_time
                )

                return ChannelResponse(
                    success=True,
                    channel_type=self.channel_type,
                    platform_message_id=platform_message_id,
                    delivery_status=DeliveryStatus.SENT,
                    recipient=recipient,
                    processing_time_ms=processing_time,
                    metadata={
                        "api_response": response_data,
                        "whatsapp_phone_number_id": self.phone_number_id
                    }
                )

            elif response.status_code == 429:
                # Rate limit exceeded
                retry_after = response.headers.get("Retry-After", "60")
                processing_time = self._calculate_processing_time(processing_start)
                self.update_metrics(False, processing_time, "RATE_LIMIT_EXCEEDED")

                raise ChannelRateLimitError(
                    channel="whatsapp",
                    limit=self.config.requests_per_minute,
                    window="minute",
                    retry_after=int(retry_after)
                )

            else:
                # Handle API errors
                error_message = f"WhatsApp API error: {response.status_code}"
                error_code = "API_ERROR"

                try:
                    error_data = response.json()
                    if "error" in error_data:
                        error_details = error_data["error"]
                        error_message = error_details.get("message", error_message)
                        error_code = error_details.get("code", error_code)

                        # Check for specific error types
                        if error_details.get("error_subcode") == 2388055:
                            error_code = "TEMPLATE_MESSAGE_REQUIRED"
                            error_message = "Template message required for this recipient"
                        elif error_details.get("error_subcode") == 131056:
                            error_code = "RECIPIENT_UNAVAILABLE"
                            error_message = "Recipient phone number is not available on WhatsApp"

                except Exception:
                    pass

                processing_time = self._calculate_processing_time(processing_start)
                self.update_metrics(False, processing_time, error_code)

                is_retryable = response.status_code >= 500 or response.status_code == 429

                self.logger.error(
                    "WhatsApp API request failed",
                    recipient=recipient,
                    status_code=response.status_code,
                    error_code=error_code,
                    error_message=error_message,
                    is_retryable=is_retryable
                )

                return self._create_error_response(
                    error_code,
                    error_message,
                    recipient,
                    is_retryable=is_retryable,
                    processing_time_ms=processing_time
                )

        except ChannelRateLimitError:
            # Re-raise rate limit errors
            raise

        except httpx.TimeoutException:
            processing_time = self._calculate_processing_time(processing_start)
            self.update_metrics(False, processing_time, "TIMEOUT")

            self.logger.error(
                "WhatsApp API timeout",
                recipient=recipient,
                timeout_seconds=self.config.timeout_seconds
            )

            return self._create_error_response(
                "TIMEOUT",
                f"WhatsApp API request timed out after {self.config.timeout_seconds}s",
                recipient,
                is_retryable=True,
                processing_time_ms=processing_time
            )

        except Exception as e:
            processing_time = self._calculate_processing_time(processing_start)
            self.update_metrics(False, processing_time, "SEND_FAILED")

            self.logger.error(
                "WhatsApp message send failed",
                recipient=recipient,
                error=str(e),
                error_type=type(e).__name__
            )

            return self._create_error_response(
                "SEND_FAILED",
                f"Failed to send WhatsApp message: {str(e)}",
                recipient,
                is_retryable=not isinstance(e, ValidationError),
                processing_time_ms=processing_time
            )

    async def validate_recipient(self, recipient: str) -> bool:
        """Validate WhatsApp phone number (E.164 format)."""
        try:
            # E.164 format: +[country][number] (max 15 digits total)
            pattern = r'^\+[1-9]\d{1,14}$'
            is_valid = bool(re.match(pattern, recipient))

            if not is_valid:
                self.logger.warning(
                    "Invalid WhatsApp phone number format",
                    recipient=recipient,
                    expected_format="E.164 (+1234567890)"
                )

            return is_valid

        except Exception as e:
            self.logger.error(
                "Recipient validation failed",
                recipient=recipient,
                error=str(e)
            )
            return False

    async def format_message(self, content: MessageContent) -> Dict[str, Any]:
        """Format message content for WhatsApp API."""
        try:
            if content.type == MessageType.TEXT:
                return await self._format_text_message(content)
            elif content.type == MessageType.IMAGE:
                return await self._format_media_message(content, "image")
            elif content.type == MessageType.AUDIO:
                return await self._format_media_message(content, "audio")
            elif content.type == MessageType.VIDEO:
                return await self._format_media_message(content, "video")
            elif content.type == MessageType.FILE:
                return await self._format_media_message(content, "document")
            elif content.type == MessageType.LOCATION:
                return await self._format_location_message(content)
            else:
                raise ValidationError(
                    field="message_type",
                    value=content.type.value,
                    validation_rule=f"Unsupported message type for WhatsApp: {content.type.value}",
                    expected_format="text, image, audio, video, document, location"
                )

        except Exception as e:
            self.logger.error(
                "Message formatting failed",
                message_type=content.type.value,
                error=str(e)
            )
            raise ChannelError(f"Failed to format WhatsApp message: {e}")

    async def _format_text_message(self, content: MessageContent) -> Dict[str, Any]:
        """Format text message for WhatsApp."""
        if not content.text:
            raise ValidationError(
                field="text",
                value="",
                validation_rule="Text content is required for text messages"
            )

        message = {
            "type": "text",
            "text": {"body": content.text}
        }

        # Add interactive elements if supported
        if content.buttons and len(content.buttons) <= 3:
            message["type"] = "interactive"
            message["interactive"] = {
                "type": "button",
                "body": {"text": content.text},
                "action": {
                    "buttons": [
                        {
                            "type": "reply",
                            "reply": {
                                "id": button.payload,
                                "title": button.title[:20]  # WhatsApp limit
                            }
                        }
                        for button in content.buttons[:3]  # WhatsApp limit
                    ]
                }
            }
            message.pop("text")

        # Add quick replies support
        elif content.quick_replies and len(content.quick_replies) <= 10:
            message["type"] = "interactive"
            message["interactive"] = {
                "type": "list",
                "body": {"text": content.text},
                "action": {
                    "button": "Options",
                    "sections": [
                        {
                            "title": "Quick Replies",
                            "rows": [
                                {
                                    "id": qr.payload,
                                    "title": qr.title[:24],  # WhatsApp limit
                                    "description": ""
                                }
                                for qr in content.quick_replies[:10]  # WhatsApp limit
                            ]
                        }
                    ]
                }
            }
            message.pop("text")

        return message

    async def _format_media_message(
            self,
            content: MessageContent,
            media_type: str
    ) -> Dict[str, Any]:
        """Format media message for WhatsApp."""
        if not content.media:
            raise ValidationError(
                field="media",
                value="None",
                validation_rule="Media content is required for media messages"
            )

        # Check file size limits
        size_limits = {
            "image": 5 * 1024 * 1024,  # 5MB
            "audio": 16 * 1024 * 1024,  # 16MB
            "video": 16 * 1024 * 1024,  # 16MB
            "document": 100 * 1024 * 1024  # 100MB
        }

        if content.media.size_bytes > size_limits.get(media_type, 100 * 1024 * 1024):
            raise ValidationError(
                field="media_size",
                value=content.media.size_bytes,
                validation_rule=f"Media too large for WhatsApp {media_type}",
                expected_format=f"Max {size_limits.get(media_type, 100 * 1024 * 1024)} bytes"
            )

        message = {
            "type": media_type,
            media_type: {
                "link": content.media.url
            }
        }

        # Add caption if text is provided (and media type supports it)
        if content.text and media_type in ["image", "video", "document"]:
            message[media_type]["caption"] = content.text[:1024]  # WhatsApp limit

        # Add filename for documents
        if media_type == "document" and hasattr(content.media, 'filename'):
            message[media_type]["filename"] = content.media.filename

        return message

    async def _format_location_message(self, content: MessageContent) -> Dict[str, Any]:
        """Format location message for WhatsApp."""
        if not content.location:
            raise ValidationError(
                field="location",
                value="None",
                validation_rule="Location content is required for location messages"
            )

        message = {
            "type": "location",
            "location": {
                "latitude": content.location.latitude,
                "longitude": content.location.longitude
            }
        }

        if content.location.address:
            message["location"]["address"] = content.location.address[:1000]  # WhatsApp limit

        if hasattr(content.location, 'name') and content.location.name:
            message["location"]["name"] = content.location.name[:1000]  # WhatsApp limit

        return message

    async def process_webhook(self, webhook_data: Dict[str, Any]) -> Dict[str, Any]:
        """Process WhatsApp webhook data."""
        try:
            processed_events = []

            # Handle verification challenge
            if "hub.challenge" in webhook_data:
                return await self._handle_verification_challenge(webhook_data)

            # Process webhook entries
            if "entry" in webhook_data:
                for entry in webhook_data["entry"]:
                    if "changes" in entry:
                        for change in entry["changes"]:
                            if change.get("field") == "messages":
                                events = await self._process_message_change(change["value"])
                                processed_events.extend(events)
                            elif change.get("field") == "message_template_status_update":
                                event = await self._process_template_status_update(change["value"])
                                if event:
                                    processed_events.append(event)

            self.logger.info(
                "Webhook processed successfully",
                events_count=len(processed_events),
                event_types=[e.get("type") for e in processed_events]
            )

            return {
                "status": "processed",
                "events_count": len(processed_events),
                "events": processed_events
            }

        except Exception as e:
            self.logger.error(
                "Webhook processing failed",
                error=str(e),
                webhook_data=webhook_data
            )
            raise ChannelError(f"Failed to process WhatsApp webhook: {e}")

    async def _handle_verification_challenge(self, webhook_data: Dict[str, Any]) -> Dict[str, Any]:
        """Handle WhatsApp webhook verification challenge."""
        mode = webhook_data.get("hub.mode")
        token = webhook_data.get("hub.verify_token")
        challenge = webhook_data.get("hub.challenge")

        if mode == "subscribe" and token == self.verify_token:
            self.logger.info("WhatsApp webhook verification successful")
            return {"challenge": challenge}
        else:
            self.logger.warning(
                "WhatsApp webhook verification failed",
                mode=mode,
                token_match=token == self.verify_token
            )
            raise ChannelError("Webhook verification failed")

    async def _process_message_change(self, change_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Process message change events from webhook."""
        events = []

        try:
            # Process incoming messages
            if "messages" in change_data:
                for message in change_data["messages"]:
                    event = {
                        "type": "message_received",
                        "platform_message_id": message.get("id"),
                        "from": message.get("from"),
                        "timestamp": message.get("timestamp"),
                        "message_type": message.get("type"),
                        "content": await self._extract_message_content(message),
                        "context": message.get("context"),
                        "metadata": {
                            "whatsapp_phone_number_id": self.phone_number_id,
                            "business_account_id": self.business_account_id
                        }
                    }
                    events.append(event)

            # Process message status updates
            if "statuses" in change_data:
                for status in change_data["statuses"]:
                    event = {
                        "type": "message_status_update",
                        "platform_message_id": status.get("id"),
                        "recipient_id": status.get("recipient_id"),
                        "status": status.get("status"),
                        "timestamp": status.get("timestamp"),
                        "errors": status.get("errors", []),
                        "metadata": {
                            "whatsapp_phone_number_id": self.phone_number_id
                        }
                    }
                    events.append(event)

            return events

        except Exception as e:
            self.logger.error(
                "Message change processing failed",
                error=str(e),
                change_data=change_data
            )
            return []

    async def _extract_message_content(self, message: Dict[str, Any]) -> Dict[str, Any]:
        """Extract content from WhatsApp message."""
        message_type = message.get("type", "text")
        content = {"type": message_type}

        if message_type == "text":
            content["text"] = message.get("text", {}).get("body", "")

        elif message_type == "image":
            image_data = message.get("image", {})
            content["media"] = {
                "id": image_data.get("id"),
                "mime_type": image_data.get("mime_type"),
                "sha256": image_data.get("sha256"),
                "caption": image_data.get("caption", "")
            }

        elif message_type == "document":
            doc_data = message.get("document", {})
            content["media"] = {
                "id": doc_data.get("id"),
                "filename": doc_data.get("filename"),
                "mime_type": doc_data.get("mime_type"),
                "sha256": doc_data.get("sha256"),
                "caption": doc_data.get("caption", "")
            }

        elif message_type == "audio":
            audio_data = message.get("audio", {})
            content["media"] = {
                "id": audio_data.get("id"),
                "mime_type": audio_data.get("mime_type"),
                "sha256": audio_data.get("sha256")
            }

        elif message_type == "video":
            video_data = message.get("video", {})
            content["media"] = {
                "id": video_data.get("id"),
                "mime_type": video_data.get("mime_type"),
                "sha256": video_data.get("sha256"),
                "caption": video_data.get("caption", "")
            }

        elif message_type == "location":
            location_data = message.get("location", {})
            content["location"] = {
                "latitude": location_data.get("latitude"),
                "longitude": location_data.get("longitude"),
                "name": location_data.get("name"),
                "address": location_data.get("address")
            }

        elif message_type == "interactive":
            interactive_data = message.get("interactive", {})
            content["interactive"] = {
                "type": interactive_data.get("type"),
                "button_reply": interactive_data.get("button_reply"),
                "list_reply": interactive_data.get("list_reply")
            }

        return content

    async def _process_template_status_update(self, status_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Process template status update from webhook."""
        return {
            "type": "template_status_update",
            "message_template_id": status_data.get("message_template_id"),
            "message_template_name": status_data.get("message_template_name"),
            "message_template_language": status_data.get("message_template_language"),
            "event": status_data.get("event"),
            "reason": status_data.get("reason"),
            "metadata": {
                "whatsapp_phone_number_id": self.phone_number_id
            }
        }

    async def verify_webhook_signature(
            self,
            payload: str,
            signature: str
    ) -> bool:
        """Verify webhook signature for security."""
        if not self.config.webhook_secret:
            self.logger.warning("Webhook secret not configured - skipping signature verification")
            return True

        try:
            expected_signature = hmac.new(
                self.config.webhook_secret.encode('utf-8'),
                payload.encode('utf-8'),
                hashlib.sha256
            ).hexdigest()

            # WhatsApp sends signature as sha256=<hash>
            if signature.startswith('sha256='):
                signature = signature[7:]

            return hmac.compare_digest(expected_signature, signature)

        except Exception as e:
            self.logger.error(
                "Webhook signature verification failed",
                error=str(e)
            )
            return False

    async def _channel_specific_health_check(self) -> bool:
        """Perform WhatsApp-specific health checks."""
        try:
            # Check if we can reach the WhatsApp API
            url = f"{self.api_base_url}/{self.phone_number_id}"
            response = await self.http_client.get(url, timeout=10)

            if response.status_code == 200:
                return True
            elif response.status_code == 401:
                self.logger.error("WhatsApp API authentication failed")
                return False
            else:
                self.logger.warning(
                    "WhatsApp API health check returned unexpected status",
                    status_code=response.status_code
                )
                return False

        except Exception as e:
            self.logger.error(
                "WhatsApp health check failed",
                error=str(e)
            )
            return False

    def _calculate_processing_time(self, start_time: datetime) -> int:
        """Calculate processing time in milliseconds."""
        return int((datetime.utcnow() - start_time).total_seconds() * 1000)

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Cleanup HTTP client on exit."""
        await self.http_client.aclose()