"""
Message normalizer for consistent data format across all channels.

This module provides normalization of complete message structures to ensure
consistent data representation regardless of the input channel.
"""

import re
import html
from datetime import datetime
from typing import Dict, Any, Optional, List
import structlog

from src.models.types import MessageContent, MessageType, ChannelType
from src.core.normalizers.content_normalizer import ContentNormalizer
from src.core.normalizers.metadata_normalizer import MetadataNormalizer
from src.core.exceptions import ContentNormalizationError

logger = structlog.get_logger(__name__)


class MessageNormalizer:
    """Normalizes complete message structures for consistent processing."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.logger = structlog.get_logger(self.__class__.__name__)

        # Initialize sub-normalizers
        self.content_normalizer = ContentNormalizer(self.config.get("content", {}))
        self.metadata_normalizer = MetadataNormalizer(self.config.get("metadata", {}))

        # Normalization settings
        self.preserve_original = self.config.get("preserve_original", True)
        self.strict_validation = self.config.get("strict_validation", False)
        self.auto_detect_language = self.config.get("auto_detect_language", True)
        self.normalize_unicode = self.config.get("normalize_unicode", True)
        self.remove_pii = self.config.get("remove_pii", False)

        # Channel-specific settings
        self.channel_specific_rules = self.config.get("channel_specific_rules", {})

        # Message ID generation
        self.generate_message_ids = self.config.get("generate_message_ids", True)

        self.logger.info(
            "Message normalizer initialized",
            preserve_original=self.preserve_original,
            strict_validation=self.strict_validation,
            auto_detect_language=self.auto_detect_language,
            remove_pii=self.remove_pii
        )

    async def normalize_message(
            self,
            message_data: Dict[str, Any],
            channel: ChannelType,
            context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Normalize a complete message structure.

        Args:
            message_data: Raw message data from channel
            channel: Source channel type
            context: Optional processing context

        Returns:
            Normalized message structure

        Raises:
            ContentNormalizationError: When normalization fails
        """
        try:
            normalized = {
                "original_data": message_data if self.preserve_original else None,
                "normalization_timestamp": datetime.utcnow().isoformat(),
                "channel": channel.value,
                "normalized_content": {},
                "normalized_metadata": {},
                "validation_results": {},
                "processing_notes": []
            }

            # Step 1: Extract and normalize message content
            content_data = self._extract_content_data(message_data, channel)
            normalized_content = await self.content_normalizer.normalize_content(
                content_data,
                channel,
                context
            )
            normalized["normalized_content"] = normalized_content

            # Step 2: Extract and normalize metadata
            metadata = self._extract_metadata(message_data, channel)
            normalized_metadata = await self.metadata_normalizer.normalize_metadata(
                metadata,
                channel,
                context
            )
            normalized["normalized_metadata"] = normalized_metadata

            # Step 3: Apply channel-specific normalization rules
            if channel.value in self.channel_specific_rules:
                normalized = await self._apply_channel_specific_rules(
                    normalized,
                    channel,
                    context
                )

            # Step 4: Validate normalized structure
            validation_results = await self._validate_normalized_message(normalized, context)
            normalized["validation_results"] = validation_results

            # Step 5: Apply post-processing
            normalized = await self._post_process_normalized_message(normalized, context)

            # Step 6: Generate message identifiers if needed
            if self.generate_message_ids:
                normalized = await self._generate_message_identifiers(normalized, context)

            self.logger.debug(
                "Message normalization completed",
                channel=channel.value,
                content_type=normalized_content.get("type"),
                validation_passed=validation_results.get("valid", False),
                processing_notes_count=len(normalized.get("processing_notes", []))
            )

            return normalized

        except Exception as e:
            self.logger.error(
                "Message normalization failed",
                channel=channel.value,
                error=str(e),
                error_type=type(e).__name__
            )

            raise ContentNormalizationError(
                normalizer="MessageNormalizer",
                content_type="complete_message",
                error_details=f"Failed to normalize message: {str(e)}"
            )

    def _extract_content_data(
            self,
            message_data: Dict[str, Any],
            channel: ChannelType
    ) -> Dict[str, Any]:
        """Extract content data from raw message based on channel format."""
        try:
            # Channel-specific content extraction
            if channel == ChannelType.WHATSAPP:
                return self._extract_whatsapp_content(message_data)
            elif channel == ChannelType.WEB:
                return self._extract_web_content(message_data)
            elif channel == ChannelType.MESSENGER:
                return self._extract_messenger_content(message_data)
            elif channel == ChannelType.SLACK:
                return self._extract_slack_content(message_data)
            elif channel == ChannelType.TEAMS:
                return self._extract_teams_content(message_data)
            else:
                return self._extract_generic_content(message_data)

        except Exception as e:
            self.logger.error(
                "Content extraction failed",
                channel=channel.value,
                error=str(e)
            )
            return self._extract_generic_content(message_data)

    def _extract_whatsapp_content(self, message_data: Dict[str, Any]) -> Dict[str, Any]:
        """Extract content from WhatsApp message format."""
        content = {}

        # Text content
        if "text" in message_data:
            content["type"] = "text"
            content["text"] = message_data["text"].get("body", "")

        # Media content
        elif "image" in message_data:
            content["type"] = "image"
            content["media"] = {
                "url": message_data["image"].get("link", ""),
                "type": "image/jpeg",  # Default for WhatsApp
                "caption": message_data["image"].get("caption", ""),
                "id": message_data["image"].get("id", "")
            }

        elif "audio" in message_data:
            content["type"] = "audio"
            content["media"] = {
                "url": message_data["audio"].get("link", ""),
                "type": "audio/ogg",  # WhatsApp default
                "id": message_data["audio"].get("id", "")
            }

        elif "video" in message_data:
            content["type"] = "video"
            content["media"] = {
                "url": message_data["video"].get("link", ""),
                "type": "video/mp4",  # WhatsApp default
                "caption": message_data["video"].get("caption", ""),
                "id": message_data["video"].get("id", "")
            }

        elif "document" in message_data:
            content["type"] = "file"
            content["media"] = {
                "url": message_data["document"].get("link", ""),
                "type": message_data["document"].get("mime_type", "application/octet-stream"),
                "filename": message_data["document"].get("filename", ""),
                "caption": message_data["document"].get("caption", ""),
                "id": message_data["document"].get("id", "")
            }

        elif "location" in message_data:
            content["type"] = "location"
            loc_data = message_data["location"]
            content["location"] = {
                "latitude": loc_data.get("latitude"),
                "longitude": loc_data.get("longitude"),
                "address": loc_data.get("address", ""),
                "name": loc_data.get("name", "")
            }

        elif "interactive" in message_data:
            # Handle interactive messages (buttons, lists)
            interactive = message_data["interactive"]
            if interactive.get("type") == "button_reply":
                content["type"] = "text"
                content["text"] = interactive.get("button_reply", {}).get("title", "")
                content["payload"] = interactive.get("button_reply", {}).get("id", "")
            elif interactive.get("type") == "list_reply":
                content["type"] = "text"
                content["text"] = interactive.get("list_reply", {}).get("title", "")
                content["payload"] = interactive.get("list_reply", {}).get("id", "")

        return content

    def _extract_web_content(self, message_data: Dict[str, Any]) -> Dict[str, Any]:
        """Extract content from web message format."""
        content = {}

        # Web messages are typically simpler
        content["type"] = message_data.get("type", "text")

        if "text" in message_data:
            content["text"] = message_data["text"]

        if "media" in message_data:
            content["media"] = message_data["media"]

        if "location" in message_data:
            content["location"] = message_data["location"]

        if "buttons" in message_data:
            content["buttons"] = message_data["buttons"]

        if "quick_replies" in message_data:
            content["quick_replies"] = message_data["quick_replies"]

        return content

    def _extract_messenger_content(self, message_data: Dict[str, Any]) -> Dict[str, Any]:
        """Extract content from Facebook Messenger format."""
        content = {}

        # Messenger message structure
        if "text" in message_data:
            content["type"] = "text"
            content["text"] = message_data["text"]

        elif "attachments" in message_data and message_data["attachments"]:
            attachment = message_data["attachments"][0]  # Handle first attachment
            attachment_type = attachment.get("type", "")

            if attachment_type == "image":
                content["type"] = "image"
                content["media"] = {
                    "url": attachment.get("payload", {}).get("url", ""),
                    "type": "image/jpeg"
                }
            elif attachment_type == "audio":
                content["type"] = "audio"
                content["media"] = {
                    "url": attachment.get("payload", {}).get("url", ""),
                    "type": "audio/mpeg"
                }
            elif attachment_type == "video":
                content["type"] = "video"
                content["media"] = {
                    "url": attachment.get("payload", {}).get("url", ""),
                    "type": "video/mp4"
                }
            elif attachment_type == "file":
                content["type"] = "file"
                content["media"] = {
                    "url": attachment.get("payload", {}).get("url", ""),
                    "type": "application/octet-stream"
                }
            elif attachment_type == "location":
                content["type"] = "location"
                payload = attachment.get("payload", {})
                content["location"] = {
                    "latitude": payload.get("coordinates", {}).get("lat"),
                    "longitude": payload.get("coordinates", {}).get("long")
                }

        return content

    def _extract_slack_content(self, message_data: Dict[str, Any]) -> Dict[str, Any]:
        """Extract content from Slack message format."""
        content = {}

        # Slack message structure
        content["type"] = "text"  # Most Slack messages are text
        content["text"] = message_data.get("text", "")

        # Handle Slack attachments
        if "files" in message_data and message_data["files"]:
            file_data = message_data["files"][0]  # Handle first file
            content["type"] = "file"
            content["media"] = {
                "url": file_data.get("url_private", ""),
                "type": file_data.get("mimetype", "application/octet-stream"),
                "filename": file_data.get("name", ""),
                "size_bytes": file_data.get("size", 0)
            }

        return content

    def _extract_teams_content(self, message_data: Dict[str, Any]) -> Dict[str, Any]:
        """Extract content from Microsoft Teams format."""
        content = {}

        # Teams message structure
        content["type"] = "text"
        content["text"] = message_data.get("text", "")

        # Handle Teams attachments
        if "attachments" in message_data and message_data["attachments"]:
            attachment = message_data["attachments"][0]
            content["type"] = "file"
            content["media"] = {
                "url": attachment.get("contentUrl", ""),
                "type": attachment.get("contentType", "application/octet-stream"),
                "filename": attachment.get("name", "")
            }

        return content

    def _extract_generic_content(self, message_data: Dict[str, Any]) -> Dict[str, Any]:
        """Extract content using generic format."""
        content = {
            "type": message_data.get("type", "text"),
            "text": message_data.get("text", ""),
        }

        if "media" in message_data:
            content["media"] = message_data["media"]

        if "location" in message_data:
            content["location"] = message_data["location"]

        return content

    def _extract_metadata(
            self,
            message_data: Dict[str, Any],
            channel: ChannelType
    ) -> Dict[str, Any]:
        """Extract metadata from message data."""
        metadata = {}

        # Common metadata fields
        metadata["timestamp"] = message_data.get("timestamp", datetime.utcnow().isoformat())
        metadata["message_id"] = message_data.get("id", message_data.get("message_id"))
        metadata["from"] = message_data.get("from", message_data.get("sender"))
        metadata["to"] = message_data.get("to", message_data.get("recipient"))

        # Channel-specific metadata
        if channel == ChannelType.WHATSAPP:
            metadata["wa_id"] = message_data.get("from")
            metadata["profile_name"] = message_data.get("profile", {}).get("name")
            metadata["context"] = message_data.get("context")

        elif channel == ChannelType.WEB:
            metadata["session_id"] = message_data.get("session_id")
            metadata["user_agent"] = message_data.get("user_agent")
            metadata["ip_address"] = message_data.get("ip_address")

        elif channel == ChannelType.SLACK:
            metadata["user"] = message_data.get("user")
            metadata["team"] = message_data.get("team")
            metadata["channel_id"] = message_data.get("channel")
            metadata["thread_ts"] = message_data.get("thread_ts")

        elif channel == ChannelType.TEAMS:
            metadata["conversation"] = message_data.get("conversation")
            metadata["from_id"] = message_data.get("from", {}).get("id")
            metadata["recipient_id"] = message_data.get("recipient", {}).get("id")

        # Remove None values
        metadata = {k: v for k, v in metadata.items() if v is not None}

        return metadata

    async def _apply_channel_specific_rules(
            self,
            normalized: Dict[str, Any],
            channel: ChannelType,
            context: Optional[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Apply channel-specific normalization rules."""
        try:
            rules = self.channel_specific_rules.get(channel.value, {})

            for rule_name, rule_config in rules.items():
                if rule_config.get("enabled", True):
                    normalized = await self._apply_normalization_rule(
                        normalized,
                        rule_name,
                        rule_config,
                        context
                    )

            return normalized

        except Exception as e:
            self.logger.error(
                "Channel-specific rule application failed",
                channel=channel.value,
                error=str(e)
            )
            return normalized

    async def _apply_normalization_rule(
            self,
            normalized: Dict[str, Any],
            rule_name: str,
            rule_config: Dict[str, Any],
            context: Optional[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Apply a specific normalization rule."""
        try:
            if rule_name == "remove_html_tags":
                # Remove HTML tags from text content
                content = normalized.get("normalized_content", {})
                if content.get("text"):
                    content["text"] = re.sub(r'<[^>]+>', '', content["text"])

            elif rule_name == "decode_html_entities":
                # Decode HTML entities
                content = normalized.get("normalized_content", {})
                if content.get("text"):
                    content["text"] = html.unescape(content["text"])

            elif rule_name == "normalize_whitespace":
                # Normalize whitespace
                content = normalized.get("normalized_content", {})
                if content.get("text"):
                    content["text"] = re.sub(r'\s+', ' ', content["text"]).strip()

            elif rule_name == "convert_emojis":
                # Convert emoji representations (if needed)
                content = normalized.get("normalized_content", {})
                if content.get("text"):
                    # This would convert platform-specific emoji formats
                    pass

            elif rule_name == "validate_urls":
                # Validate and normalize URLs in content
                content = normalized.get("normalized_content", {})
                if content.get("media", {}).get("url"):
                    url = content["media"]["url"]
                    if not url.startswith(('http://', 'https://')):
                        content["media"]["url"] = None
                        normalized["processing_notes"].append("Invalid media URL removed")

            return normalized

        except Exception as e:
            self.logger.error(
                "Normalization rule application failed",
                rule_name=rule_name,
                error=str(e)
            )
            return normalized

    async def _validate_normalized_message(
            self,
            normalized: Dict[str, Any],
            context: Optional[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Validate the normalized message structure."""
        validation = {
            "valid": True,
            "errors": [],
            "warnings": [],
            "checks_performed": []
        }

        try:
            content = normalized.get("normalized_content", {})
            metadata = normalized.get("normalized_metadata", {})

            # Check required fields
            if not content.get("type"):
                validation["errors"].append("Missing content type")
                validation["valid"] = False

            validation["checks_performed"].append("content_type")

            # Validate content based on type
            content_type = content.get("type")

            if content_type == "text":
                if not content.get("text"):
                    validation["warnings"].append("Empty text content")
                validation["checks_performed"].append("text_content")

            elif content_type in ["image", "audio", "video", "file"]:
                if not content.get("media", {}).get("url"):
                    validation["errors"].append(f"Missing media URL for {content_type}")
                    validation["valid"] = False
                validation["checks_performed"].append("media_content")

            elif content_type == "location":
                location = content.get("location", {})
                if not (location.get("latitude") and location.get("longitude")):
                    validation["errors"].append("Missing location coordinates")
                    validation["valid"] = False
                validation["checks_performed"].append("location_content")

            # Validate metadata
            if not metadata.get("timestamp"):
                validation["warnings"].append("Missing timestamp")

            validation["checks_performed"].append("metadata")

            # Strict validation if enabled
            if self.strict_validation:
                if not metadata.get("message_id"):
                    validation["errors"].append("Missing message ID")
                    validation["valid"] = False

                if not metadata.get("from"):
                    validation["errors"].append("Missing sender information")
                    validation["valid"] = False

            return validation

        except Exception as e:
            self.logger.error(
                "Message validation failed",
                error=str(e)
            )

            validation["valid"] = False
            validation["errors"].append(f"Validation error: {str(e)}")
            return validation

    async def _post_process_normalized_message(
            self,
            normalized: Dict[str, Any],
            context: Optional[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Apply post-processing to normalized message."""
        try:
            # Add processing metadata
            normalized["processing_metadata"] = {
                "normalization_version": "1.0.0",
                "processed_at": datetime.utcnow().isoformat(),
                "processing_duration_ms": 0,  # Would be calculated
                "rules_applied": len(normalized.get("processing_notes", [])),
                "validation_passed": normalized.get("validation_results", {}).get("valid", False)
            }

            # Clean up temporary fields if needed
            if not self.preserve_original and "original_data" in normalized:
                del normalized["original_data"]

            return normalized

        except Exception as e:
            self.logger.error(
                "Post-processing failed",
                error=str(e)
            )
            return normalized

    async def _generate_message_identifiers(
            self,
            normalized: Dict[str, Any],
            context: Optional[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Generate message identifiers if missing."""
        try:
            import uuid

            metadata = normalized.get("normalized_metadata", {})

            # Generate message ID if missing
            if not metadata.get("message_id"):
                metadata["message_id"] = str(uuid.uuid4())
                normalized["processing_notes"].append("Generated message ID")

            # Generate correlation ID for tracking
            if not metadata.get("correlation_id"):
                metadata["correlation_id"] = str(uuid.uuid4())

            return normalized

        except Exception as e:
            self.logger.error(
                "Identifier generation failed",
                error=str(e)
            )
            return normalized

    async def normalize_batch_messages(
            self,
            messages: List[Dict[str, Any]],
            channel: ChannelType,
            context: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """
        Normalize a batch of messages efficiently.

        Args:
            messages: List of raw message data
            channel: Source channel type
            context: Optional processing context

        Returns:
            List of normalized messages
        """
        normalized_messages = []

        for i, message_data in enumerate(messages):
            try:
                normalized = await self.normalize_message(message_data, channel, context)
                normalized_messages.append(normalized)

            except Exception as e:
                self.logger.error(
                    "Batch message normalization failed",
                    message_index=i,
                    error=str(e)
                )

                # Add error placeholder
                normalized_messages.append({
                    "error": True,
                    "error_message": str(e),
                    "original_data": message_data if self.preserve_original else None,
                    "message_index": i
                })

        self.logger.info(
            "Batch normalization completed",
            total_messages=len(messages),
            successful=len([m for m in normalized_messages if not m.get("error")]),
            failed=len([m for m in normalized_messages if m.get("error")])
        )

        return normalized_messages