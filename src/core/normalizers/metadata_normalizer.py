"""
Metadata normalizer for consistent metadata format across all channels.

This module handles normalization of message metadata including timestamps,
user information, channel-specific data, and request context.
"""

import re
from datetime import datetime, timezone
from typing import Dict, Any, Optional, List
from urllib.parse import urlparse
import structlog

from src.models.types import ChannelType
from src.core.exceptions import ContentNormalizationError

logger = structlog.get_logger(__name__)


class MetadataNormalizer:
    """Normalizes message metadata for consistent processing."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.logger = structlog.get_logger(self.__class__.__name__)

        # Timestamp normalization settings
        self.normalize_timestamps = self.config.get("normalize_timestamps", True)
        self.require_utc_timestamps = self.config.get("require_utc_timestamps", True)
        self.timestamp_tolerance_hours = self.config.get("timestamp_tolerance_hours", 24)

        # ID normalization settings
        self.normalize_ids = self.config.get("normalize_ids", True)
        self.generate_missing_ids = self.config.get("generate_missing_ids", True)
        self.validate_id_formats = self.config.get("validate_id_formats", True)

        # User information normalization
        self.normalize_user_info = self.config.get("normalize_user_info", True)
        self.anonymize_sensitive_data = self.config.get("anonymize_sensitive_data", False)
        self.validate_user_data = self.config.get("validate_user_data", True)

        # Channel metadata normalization
        self.normalize_channel_metadata = self.config.get("normalize_channel_metadata", True)
        self.extract_platform_info = self.config.get("extract_platform_info", True)

        # Security and privacy settings
        self.remove_ip_addresses = self.config.get("remove_ip_addresses", False)
        self.hash_sensitive_ids = self.config.get("hash_sensitive_ids", False)
        self.validate_origins = self.config.get("validate_origins", True)

        # Validation patterns
        self.uuid_pattern = re.compile(
            r'^[0-9a-f]{8}-[0-9a-f]{4}-[1-5][0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$',
            re.IGNORECASE
        )
        self.phone_pattern = re.compile(r'^\+?[1-9]\d{1,14}$')
        self.email_pattern = re.compile(r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$')

        self.logger.info(
            "Metadata normalizer initialized",
            timestamp_settings={
                "normalize_timestamps": self.normalize_timestamps,
                "require_utc": self.require_utc_timestamps,
                "tolerance_hours": self.timestamp_tolerance_hours
            },
            id_settings={
                "normalize_ids": self.normalize_ids,
                "generate_missing": self.generate_missing_ids,
                "validate_formats": self.validate_id_formats
            },
            privacy_settings={
                "anonymize_sensitive": self.anonymize_sensitive_data,
                "remove_ips": self.remove_ip_addresses,
                "hash_ids": self.hash_sensitive_ids
            }
        )

    async def normalize_metadata(
            self,
            metadata: Dict[str, Any],
            channel: ChannelType,
            context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Normalize message metadata based on channel and context.

        Args:
            metadata: Raw metadata from message
            channel: Source channel type
            context: Optional processing context

        Returns:
            Normalized metadata structure

        Raises:
            ContentNormalizationError: When normalization fails
        """
        try:
            normalized = {
                "channel": channel.value,
                "normalization_applied": [],
                "validation_warnings": [],
                "processing_notes": [],
                "original_metadata": metadata.copy() if metadata else {}
            }

            # Normalize timestamps
            if self.normalize_timestamps:
                timestamp_result = await self._normalize_timestamps(metadata, channel)
                normalized.update(timestamp_result.get("normalized", {}))
                normalized["normalization_applied"].extend(timestamp_result.get("applied", []))
                normalized["validation_warnings"].extend(timestamp_result.get("warnings", []))

            # Normalize identifiers
            if self.normalize_ids:
                id_result = await self._normalize_identifiers(metadata, channel)
                normalized.update(id_result.get("normalized", {}))
                normalized["normalization_applied"].extend(id_result.get("applied", []))
                normalized["validation_warnings"].extend(id_result.get("warnings", []))

            # Normalize user information
            if self.normalize_user_info:
                user_result = await self._normalize_user_information(metadata, channel)
                normalized.update(user_result.get("normalized", {}))
                normalized["normalization_applied"].extend(user_result.get("applied", []))
                normalized["validation_warnings"].extend(user_result.get("warnings", []))

            # Normalize channel-specific metadata
            if self.normalize_channel_metadata:
                channel_result = await self._normalize_channel_specific_metadata(metadata, channel)
                normalized.update(channel_result.get("normalized", {}))
                normalized["normalization_applied"].extend(channel_result.get("applied", []))
                normalized["validation_warnings"].extend(channel_result.get("warnings", []))

            # Apply security and privacy normalizations
            security_result = await self._apply_security_normalizations(normalized, context)
            normalized.update(security_result.get("normalized", {}))
            normalized["normalization_applied"].extend(security_result.get("applied", []))

            # Validate normalized metadata
            validation_result = await self._validate_normalized_metadata(normalized, context)
            normalized["validation_result"] = validation_result

            # Clean up internal fields
            normalized = await self._finalize_metadata(normalized, context)

            self.logger.debug(
                "Metadata normalization completed",
                channel=channel.value,
                normalizations_applied=len(normalized.get("normalization_applied", [])),
                warnings=len(normalized.get("validation_warnings", []))
            )

            return normalized

        except Exception as e:
            self.logger.error(
                "Metadata normalization failed",
                channel=channel.value,
                error=str(e),
                error_type=type(e).__name__
            )

            raise ContentNormalizationError(
                normalizer="MetadataNormalizer",
                content_type="metadata",
                error_details=f"Failed to normalize metadata: {str(e)}"
            )

    async def _normalize_timestamps(
            self,
            metadata: Dict[str, Any],
            channel: ChannelType
    ) -> Dict[str, Any]:
        """Normalize timestamp fields."""
        result = {
            "normalized": {},
            "applied": [],
            "warnings": []
        }

        try:
            # Get timestamp from various possible fields
            timestamp_fields = ["timestamp", "created_at", "sent_at", "received_at", "time"]
            raw_timestamp = None
            timestamp_field = None

            for field in timestamp_fields:
                if field in metadata and metadata[field]:
                    raw_timestamp = metadata[field]
                    timestamp_field = field
                    break

            if raw_timestamp:
                normalized_timestamp = await self._parse_and_normalize_timestamp(raw_timestamp)
                if normalized_timestamp:
                    result["normalized"]["timestamp"] = normalized_timestamp.isoformat()
                    result["normalized"]["timestamp_unix"] = int(normalized_timestamp.timestamp())
                    result["applied"].append("timestamp_normalization")

                    # Validate timestamp is reasonable
                    now = datetime.now(timezone.utc)
                    time_diff = abs((normalized_timestamp - now).total_seconds() / 3600)

                    if time_diff > self.timestamp_tolerance_hours:
                        result["warnings"].append(
                            f"Timestamp is {time_diff:.1f} hours from current time"
                        )
                else:
                    result["warnings"].append(f"Could not parse timestamp: {raw_timestamp}")
                    # Use current timestamp as fallback
                    result["normalized"]["timestamp"] = datetime.now(timezone.utc).isoformat()
                    result["applied"].append("timestamp_fallback")
            else:
                # Generate timestamp if missing
                result["normalized"]["timestamp"] = datetime.now(timezone.utc).isoformat()
                result["applied"].append("timestamp_generated")

            # Add processing timestamp
            result["normalized"]["processed_at"] = datetime.now(timezone.utc).isoformat()

            return result

        except Exception as e:
            self.logger.error("Timestamp normalization failed", error=str(e))
            result["warnings"].append(f"Timestamp normalization error: {str(e)}")
            return result

    async def _parse_and_normalize_timestamp(self, timestamp: Any) -> Optional[datetime]:
        """Parse various timestamp formats and normalize to UTC."""
        try:
            if isinstance(timestamp, datetime):
                # Ensure timezone awareness
                if timestamp.tzinfo is None:
                    timestamp = timestamp.replace(tzinfo=timezone.utc)
                return timestamp.astimezone(timezone.utc)

            elif isinstance(timestamp, (int, float)):
                # Unix timestamp
                if timestamp > 1e10:  # Milliseconds
                    timestamp = timestamp / 1000
                return datetime.fromtimestamp(timestamp, timezone.utc)

            elif isinstance(timestamp, str):
                # Try various string formats
                formats = [
                    "%Y-%m-%dT%H:%M:%S.%fZ",  # ISO 8601 with microseconds
                    "%Y-%m-%dT%H:%M:%SZ",  # ISO 8601 basic
                    "%Y-%m-%dT%H:%M:%S.%f%z",  # ISO 8601 with timezone
                    "%Y-%m-%dT%H:%M:%S%z",  # ISO 8601 with timezone, no microseconds
                    "%Y-%m-%d %H:%M:%S",  # Simple format
                    "%Y-%m-%d %H:%M:%S.%f",  # Simple with microseconds
                    "%d/%m/%Y %H:%M:%S",  # DD/MM/YYYY format
                    "%m/%d/%Y %H:%M:%S",  # MM/DD/YYYY format
                ]

                for fmt in formats:
                    try:
                        dt = datetime.strptime(timestamp, fmt)
                        if dt.tzinfo is None:
                            dt = dt.replace(tzinfo=timezone.utc)
                        return dt.astimezone(timezone.utc)
                    except ValueError:
                        continue

                # Try parsing as Unix timestamp string
                try:
                    unix_time = float(timestamp)
                    if unix_time > 1e10:  # Milliseconds
                        unix_time = unix_time / 1000
                    return datetime.fromtimestamp(unix_time, timezone.utc)
                except ValueError:
                    pass

            return None

        except Exception as e:
            self.logger.error("Timestamp parsing failed", timestamp=timestamp, error=str(e))
            return None

    async def _normalize_identifiers(
            self,
            metadata: Dict[str, Any],
            channel: ChannelType
    ) -> Dict[str, Any]:
        """Normalize various identifier fields."""
        result = {
            "normalized": {},
            "applied": [],
            "warnings": []
        }

        try:
            # Message ID normalization
            message_id = metadata.get("message_id") or metadata.get("id")
            if message_id:
                normalized_id = await self._normalize_id(message_id, "message_id")
                if normalized_id:
                    result["normalized"]["message_id"] = normalized_id
                    result["applied"].append("message_id_normalization")
                else:
                    result["warnings"].append(f"Invalid message ID format: {message_id}")
            elif self.generate_missing_ids:
                result["normalized"]["message_id"] = self._generate_uuid()
                result["applied"].append("message_id_generated")

            # Conversation ID normalization
            conversation_id = metadata.get("conversation_id") or metadata.get("thread_id")
            if conversation_id:
                normalized_conv_id = await self._normalize_id(conversation_id, "conversation_id")
                if normalized_conv_id:
                    result["normalized"]["conversation_id"] = normalized_conv_id
                    result["applied"].append("conversation_id_normalization")

            # User ID normalization
            user_id = metadata.get("from") or metadata.get("user_id") or metadata.get("sender")
            if user_id:
                normalized_user_id = await self._normalize_user_id(user_id, channel)
                if normalized_user_id:
                    result["normalized"]["user_id"] = normalized_user_id
                    result["applied"].append("user_id_normalization")

            # Session ID normalization
            session_id = metadata.get("session_id")
            if session_id:
                normalized_session_id = await self._normalize_id(session_id, "session_id")
                if normalized_session_id:
                    result["normalized"]["session_id"] = normalized_session_id
                    result["applied"].append("session_id_normalization")

            return result

        except Exception as e:
            self.logger.error("ID normalization failed", error=str(e))
            result["warnings"].append(f"ID normalization error: {str(e)}")
            return result

    async def _normalize_id(self, id_value: Any, id_type: str) -> Optional[str]:
        """Normalize a generic ID value."""
        try:
            if not id_value:
                return None

            id_str = str(id_value).strip()

            # Validate format if required
            if self.validate_id_formats:
                if id_type in ["message_id", "conversation_id", "session_id"]:
                    # Expect UUID format for internal IDs
                    if self.uuid_pattern.match(id_str):
                        return id_str.lower()
                    else:
                        # Platform-specific IDs might not be UUIDs
                        if len(id_str) > 0 and len(id_str) <= 255:
                            return id_str
                        return None

            return id_str

        except Exception:
            return None

    async def _normalize_user_id(self, user_id: Any, channel: ChannelType) -> Optional[str]:
        """Normalize user ID based on channel type."""
        try:
            if not user_id:
                return None

            user_id_str = str(user_id).strip()

            # Channel-specific user ID validation
            if channel == ChannelType.WHATSAPP:
                # WhatsApp uses phone numbers
                if self.phone_pattern.match(user_id_str):
                    return user_id_str
                elif user_id_str.isdigit() and len(user_id_str) >= 10:
                    # Add + prefix if missing
                    return f"+{user_id_str}"

            elif channel == ChannelType.WEB:
                # Web might use email or UUID
                if self.email_pattern.match(user_id_str):
                    return user_id_str.lower()
                elif self.uuid_pattern.match(user_id_str):
                    return user_id_str.lower()

            elif channel in [ChannelType.SLACK, ChannelType.TEAMS]:
                # Platform-specific user IDs
                if len(user_id_str) > 0 and len(user_id_str) <= 255:
                    return user_id_str

            # Generic validation
            if len(user_id_str) > 0 and len(user_id_str) <= 255:
                return user_id_str

            return None

        except Exception:
            return None

    async def _normalize_user_information(
            self,
            metadata: Dict[str, Any],
            channel: ChannelType
    ) -> Dict[str, Any]:
        """Normalize user-related metadata."""
        result = {
            "normalized": {},
            "applied": [],
            "warnings": []
        }

        try:
            user_info = {}

            # Extract user profile information
            if "profile" in metadata:
                profile = metadata["profile"]
                if isinstance(profile, dict):
                    user_info.update(await self._normalize_user_profile(profile, channel))
                    result["applied"].append("user_profile_normalization")

            # Extract user name information
            name_fields = ["name", "display_name", "username", "profile_name"]
            for field in name_fields:
                if field in metadata and metadata[field]:
                    normalized_name = await self._normalize_user_name(metadata[field])
                    if normalized_name:
                        user_info["display_name"] = normalized_name
                        result["applied"].append("user_name_normalization")
                        break

            # Extract contact information
            if "email" in metadata:
                email = await self._normalize_email(metadata["email"])
                if email:
                    if self.anonymize_sensitive_data:
                        user_info["email_hash"] = self._hash_value(email)
                    else:
                        user_info["email"] = email
                    result["applied"].append("email_normalization")

            if "phone" in metadata:
                phone = await self._normalize_phone(metadata["phone"])
                if phone:
                    if self.anonymize_sensitive_data:
                        user_info["phone_hash"] = self._hash_value(phone)
                    else:
                        user_info["phone"] = phone
                    result["applied"].append("phone_normalization")

            # Language and locale information
            if "language" in metadata:
                lang = await self._normalize_language_code(metadata["language"])
                if lang:
                    user_info["language"] = lang
                    result["applied"].append("language_normalization")

            if "locale" in metadata:
                locale = await self._normalize_locale(metadata["locale"])
                if locale:
                    user_info["locale"] = locale
                    result["applied"].append("locale_normalization")

            if user_info:
                result["normalized"]["user_info"] = user_info

            return result

        except Exception as e:
            self.logger.error("User information normalization failed", error=str(e))
            result["warnings"].append(f"User info normalization error: {str(e)}")
            return result

    async def _normalize_user_profile(self, profile: Dict[str, Any], channel: ChannelType) -> Dict[str, Any]:
        """Normalize user profile data."""
        normalized_profile = {}

        # Common profile fields
        if "name" in profile:
            normalized_profile["name"] = await self._normalize_user_name(profile["name"])

        if "avatar" in profile or "picture" in profile:
            avatar_url = profile.get("avatar") or profile.get("picture")
            if avatar_url and isinstance(avatar_url, str):
                normalized_profile["avatar_url"] = avatar_url

        # Channel-specific profile normalization
        if channel == ChannelType.WHATSAPP:
            if "wa_id" in profile:
                normalized_profile["platform_id"] = profile["wa_id"]

        elif channel == ChannelType.SLACK:
            if "real_name" in profile:
                normalized_profile["real_name"] = profile["real_name"]
            if "team_id" in profile:
                normalized_profile["team_id"] = profile["team_id"]

        return normalized_profile

    async def _normalize_channel_specific_metadata(
            self,
            metadata: Dict[str, Any],
            channel: ChannelType
    ) -> Dict[str, Any]:
        """Normalize channel-specific metadata."""
        result = {
            "normalized": {},
            "applied": [],
            "warnings": []
        }

        try:
            channel_metadata = {}

            if channel == ChannelType.WHATSAPP:
                channel_metadata.update(await self._normalize_whatsapp_metadata(metadata))
                result["applied"].append("whatsapp_metadata_normalization")

            elif channel == ChannelType.WEB:
                channel_metadata.update(await self._normalize_web_metadata(metadata))
                result["applied"].append("web_metadata_normalization")

            elif channel == ChannelType.MESSENGER:
                channel_metadata.update(await self._normalize_messenger_metadata(metadata))
                result["applied"].append("messenger_metadata_normalization")

            elif channel == ChannelType.SLACK:
                channel_metadata.update(await self._normalize_slack_metadata(metadata))
                result["applied"].append("slack_metadata_normalization")

            elif channel == ChannelType.TEAMS:
                channel_metadata.update(await self._normalize_teams_metadata(metadata))
                result["applied"].append("teams_metadata_normalization")

            if channel_metadata:
                result["normalized"]["channel_metadata"] = channel_metadata

            return result

        except Exception as e:
            self.logger.error("Channel metadata normalization failed", error=str(e))
            result["warnings"].append(f"Channel metadata normalization error: {str(e)}")
            return result

    async def _normalize_whatsapp_metadata(self, metadata: Dict[str, Any]) -> Dict[str, Any]:
        """Normalize WhatsApp-specific metadata."""
        whatsapp_meta = {}

        if "wa_id" in metadata:
            whatsapp_meta["wa_id"] = metadata["wa_id"]

        if "profile" in metadata:
            profile = metadata["profile"]
            if isinstance(profile, dict) and "name" in profile:
                whatsapp_meta["profile_name"] = profile["name"]

        if "context" in metadata:
            context = metadata["context"]
            if isinstance(context, dict):
                whatsapp_meta["context"] = {
                    "from": context.get("from"),
                    "id": context.get("id"),
                    "referred_product": context.get("referred_product")
                }

        return whatsapp_meta

    async def _normalize_web_metadata(self, metadata: Dict[str, Any]) -> Dict[str, Any]:
        """Normalize web-specific metadata."""
        web_meta = {}

        if "user_agent" in metadata:
            web_meta["user_agent"] = metadata["user_agent"]

        if "ip_address" in metadata and not self.remove_ip_addresses:
            ip_addr = metadata["ip_address"]
            if self.anonymize_sensitive_data:
                web_meta["ip_hash"] = self._hash_value(ip_addr)
            else:
                web_meta["ip_address"] = ip_addr

        if "referrer" in metadata:
            referrer = metadata["referrer"]
            if self.validate_origins and await self._validate_url(referrer):
                web_meta["referrer"] = referrer

        if "session_id" in metadata:
            web_meta["session_id"] = metadata["session_id"]

        return web_meta

    async def _normalize_messenger_metadata(self, metadata: Dict[str, Any]) -> Dict[str, Any]:
        """Normalize Facebook Messenger metadata."""
        messenger_meta = {}

        if "sender" in metadata:
            sender = metadata["sender"]
            if isinstance(sender, dict) and "id" in sender:
                messenger_meta["sender_id"] = sender["id"]

        if "recipient" in metadata:
            recipient = metadata["recipient"]
            if isinstance(recipient, dict) and "id" in recipient:
                messenger_meta["page_id"] = recipient["id"]

        return messenger_meta

    async def _normalize_slack_metadata(self, metadata: Dict[str, Any]) -> Dict[str, Any]:
        """Normalize Slack-specific metadata."""
        slack_meta = {}

        if "team" in metadata:
            slack_meta["team_id"] = metadata["team"]

        if "channel" in metadata:
            slack_meta["channel_id"] = metadata["channel"]

        if "user" in metadata:
            slack_meta["user_id"] = metadata["user"]

        if "thread_ts" in metadata:
            slack_meta["thread_timestamp"] = metadata["thread_ts"]

        if "bot_id" in metadata:
            slack_meta["bot_id"] = metadata["bot_id"]

        return slack_meta

    async def _normalize_teams_metadata(self, metadata: Dict[str, Any]) -> Dict[str, Any]:
        """Normalize Microsoft Teams metadata."""
        teams_meta = {}

        if "conversation" in metadata:
            conversation = metadata["conversation"]
            if isinstance(conversation, dict):
                teams_meta["conversation_id"] = conversation.get("id")
                teams_meta["conversation_type"] = conversation.get("conversationType")

        if "from" in metadata:
            from_info = metadata["from"]
            if isinstance(from_info, dict):
                teams_meta["from_id"] = from_info.get("id")
                teams_meta["from_name"] = from_info.get("name")

        if "channelData" in metadata:
            channel_data = metadata["channelData"]
            if isinstance(channel_data, dict):
                teams_meta["tenant_id"] = channel_data.get("tenant", {}).get("id")

        return teams_meta

    async def _apply_security_normalizations(
            self,
            normalized: Dict[str, Any],
            context: Optional[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Apply security and privacy normalizations."""
        result = {
            "normalized": normalized.copy(),
            "applied": []
        }

        try:
            # Hash sensitive IDs if required
            if self.hash_sensitive_ids:
                for field in ["user_id", "phone", "email"]:
                    if field in result["normalized"] and result["normalized"][field]:
                        original_value = result["normalized"][field]
                        result["normalized"][f"{field}_hash"] = self._hash_value(original_value)
                        if self.anonymize_sensitive_data:
                            del result["normalized"][field]
                        result["applied"].append(f"{field}_hashing")

            # Remove IP addresses if required
            if self.remove_ip_addresses:
                if "channel_metadata" in result["normalized"]:
                    channel_meta = result["normalized"]["channel_metadata"]
                    if "ip_address" in channel_meta:
                        del channel_meta["ip_address"]
                        result["applied"].append("ip_address_removal")

            # Add security metadata
            result["normalized"]["security_metadata"] = {
                "anonymization_applied": self.anonymize_sensitive_data,
                "ip_addresses_removed": self.remove_ip_addresses,
                "ids_hashed": self.hash_sensitive_ids,
                "processing_timestamp": datetime.now(timezone.utc).isoformat()
            }

            return result

        except Exception as e:
            self.logger.error("Security normalization failed", error=str(e))
            return result

    async def _validate_normalized_metadata(
            self,
            normalized: Dict[str, Any],
            context: Optional[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Validate the normalized metadata structure."""
        validation = {
            "valid": True,
            "errors": [],
            "warnings": normalized.get("validation_warnings", []),
            "score": 1.0
        }

        try:
            # Check required fields
            required_fields = ["timestamp", "channel"]
            for field in required_fields:
                if field not in normalized:
                    validation["errors"].append(f"Missing required field: {field}")
                    validation["valid"] = False

            # Validate timestamp format
            if "timestamp" in normalized:
                try:
                    datetime.fromisoformat(normalized["timestamp"].replace('Z', '+00:00'))
                except ValueError:
                    validation["errors"].append("Invalid timestamp format")
                    validation["valid"] = False

            # Validate IDs if present
            for id_field in ["message_id", "conversation_id", "session_id"]:
                if id_field in normalized:
                    id_value = normalized[id_field]
                    if not id_value or not isinstance(id_value, str):
                        validation["warnings"].append(f"Invalid {id_field} format")
                        validation["score"] -= 0.1

            # Adjust score based on warnings
            warning_count = len(validation["warnings"])
            if warning_count > 0:
                validation["score"] = max(0.1, validation["score"] - (warning_count * 0.05))

            return validation

        except Exception as e:
            self.logger.error("Metadata validation failed", error=str(e))
            validation["valid"] = False
            validation["errors"].append(f"Validation error: {str(e)}")
            validation["score"] = 0.0
            return validation

    async def _finalize_metadata(
            self,
            normalized: Dict[str, Any],
            context: Optional[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Finalize normalized metadata by cleaning up internal fields."""
        # Remove internal processing fields if not needed for debugging
        if not context or not context.get("debug_mode", False):
            normalized.pop("original_metadata", None)
            if not normalized.get("validation_warnings"):
                normalized.pop("validation_warnings", None)
            if not normalized.get("processing_notes"):
                normalized.pop("processing_notes", None)

        return normalized

    # Helper methods

    async def _normalize_user_name(self, name: str) -> Optional[str]:
        """Normalize user name."""
        if not name or not isinstance(name, str):
            return None

        # Basic name cleanup
        name = name.strip()
        if len(name) == 0:
            return None

        # Remove extra whitespace
        name = re.sub(r'\s+', ' ', name)

        # Limit length
        if len(name) > 255:
            name = name[:255]

        return name

    async def _normalize_email(self, email: str) -> Optional[str]:
        """Normalize email address."""
        if not email or not isinstance(email, str):
            return None

        email = email.strip().lower()
        if self.email_pattern.match(email):
            return email

        return None

    async def _normalize_phone(self, phone: str) -> Optional[str]:
        """Normalize phone number."""
        if not phone or not isinstance(phone, str):
            return None

        # Remove all non-digit characters except +
        phone = re.sub(r'[^\d+]', '', phone)

        if self.phone_pattern.match(phone):
            return phone

        return None

    async def _normalize_language_code(self, language: str) -> Optional[str]:
        """Normalize language code to ISO 639-1 format."""
        if not language or not isinstance(language, str):
            return None

        language = language.lower().strip()

        # Common language code mappings
        language_mappings = {
            "english": "en",
            "spanish": "es",
            "french": "fr",
            "german": "de",
            "italian": "it",
            "portuguese": "pt",
            "chinese": "zh",
            "japanese": "ja",
            "korean": "ko",
            "russian": "ru",
            "arabic": "ar",
            "hindi": "hi"
        }

        if language in language_mappings:
            return language_mappings[language]

        # Validate 2-letter code
        if len(language) == 2 and language.isalpha():
            return language

        # Extract language from locale (e.g., en-US -> en)
        if '-' in language:
            lang_part = language.split('-')[0]
            if len(lang_part) == 2 and lang_part.isalpha():
                return lang_part

        return None

    async def _normalize_locale(self, locale: str) -> Optional[str]:
        """Normalize locale code."""
        if not locale or not isinstance(locale, str):
            return None

        locale = locale.strip()

        # Validate locale format (e.g., en-US, fr-FR)
        if re.match(r'^[a-z]{2}-[A-Z]{2}$', locale):
            return locale

        # Try to fix common formats
        if '_' in locale:
            locale = locale.replace('_', '-')

        parts = locale.split('-')
        if len(parts) == 2:
            lang = parts[0].lower()
            country = parts[1].upper()
            if len(lang) == 2 and len(country) == 2:
                return f"{lang}-{country}"

        return None

    async def _validate_url(self, url: str) -> bool:
        """Validate URL format."""
        try:
            result = urlparse(url)
            return all([result.scheme, result.netloc]) and result.scheme in ['http', 'https']
        except Exception:
            return False

    def _generate_uuid(self) -> str:
        """Generate a new UUID."""
        import uuid
        return str(uuid.uuid4())

    def _hash_value(self, value: str) -> str:
        """Hash a sensitive value."""
        import hashlib
        return hashlib.sha256(value.encode()).hexdigest()[:16]  # First 16 chars of hash