"""
Content normalizer for message content standardization.

This module handles normalization of different content types (text, media, location)
to ensure consistent format and quality across all channels.
"""

import re
import html
import unicodedata
from datetime import datetime
from typing import Dict, Any, Optional, List, Tuple
from urllib.parse import urlparse, parse_qs
import structlog

from src.models.types import ChannelType
from src.core.exceptions import ContentNormalizationError

logger = structlog.get_logger(__name__)


class ContentNormalizer:
    """Normalizes message content for consistent processing."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.logger = structlog.get_logger(self.__class__.__name__)

        # Text normalization settings
        self.normalize_unicode = self.config.get("normalize_unicode", True)
        self.remove_html_tags = self.config.get("remove_html_tags", True)
        self.decode_html_entities = self.config.get("decode_html_entities", True)
        self.normalize_whitespace = self.config.get("normalize_whitespace", True)
        self.preserve_formatting = self.config.get("preserve_formatting", False)

        # Content limits
        self.max_text_length = self.config.get("max_text_length", 10000)
        self.max_media_url_length = self.config.get("max_media_url_length", 2048)

        # Content filtering
        self.filter_profanity = self.config.get("filter_profanity", False)
        self.remove_personal_info = self.config.get("remove_personal_info", False)
        self.validate_urls = self.config.get("validate_urls", True)

        # Language settings
        self.default_language = self.config.get("default_language", "en")
        self.auto_detect_language = self.config.get("auto_detect_language", True)

        # Media normalization
        self.normalize_media_urls = self.config.get("normalize_media_urls", True)
        self.validate_media_types = self.config.get("validate_media_types", True)
        self.extract_media_metadata = self.config.get("extract_media_metadata", True)

        # Location normalization
        self.normalize_coordinates = self.config.get("normalize_coordinates", True)
        self.validate_coordinate_bounds = self.config.get("validate_coordinate_bounds", True)

        # Content patterns for detection and filtering
        self.url_pattern = re.compile(
            r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
        )
        self.email_pattern = re.compile(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b')
        self.phone_pattern = re.compile(r'(\+?[\d\s\-\(\)]{10,})')

        self.logger.info(
            "Content normalizer initialized",
            text_settings={
                "normalize_unicode": self.normalize_unicode,
                "remove_html": self.remove_html_tags,
                "normalize_whitespace": self.normalize_whitespace
            },
            content_limits={
                "max_text_length": self.max_text_length,
                "max_url_length": self.max_media_url_length
            },
            filtering={
                "filter_profanity": self.filter_profanity,
                "remove_personal_info": self.remove_personal_info,
                "validate_urls": self.validate_urls
            }
        )

    async def normalize_content(
            self,
            content_data: Dict[str, Any],
            channel: ChannelType,
            context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Normalize message content based on type and channel.

        Args:
            content_data: Raw content data
            channel: Source channel type
            context: Optional processing context

        Returns:
            Normalized content structure

        Raises:
            ContentNormalizationError: When normalization fails
        """
        try:
            content_type = content_data.get("type", "text")

            normalized = {
                "type": content_type,
                "original_type": content_data.get("type"),
                "channel": channel.value,
                "normalization_applied": [],
                "validation_warnings": [],
                "processing_notes": []
            }

            # Normalize based on content type
            if content_type == "text":
                normalized.update(await self._normalize_text_content(content_data, channel, context))

            elif content_type in ["image", "audio", "video", "file"]:
                normalized.update(await self._normalize_media_content(content_data, channel, context))

            elif content_type == "location":
                normalized.update(await self._normalize_location_content(content_data, channel, context))

            else:
                # Handle unknown content types
                normalized.update(await self._normalize_generic_content(content_data, channel, context))

            # Apply common normalizations
            normalized = await self._apply_common_normalizations(normalized, context)

            # Validate normalized content
            validation_result = await self._validate_normalized_content(normalized, context)
            normalized["validation_result"] = validation_result

            self.logger.debug(
                "Content normalization completed",
                content_type=content_type,
                channel=channel.value,
                normalizations_applied=len(normalized.get("normalization_applied", [])),
                warnings=len(normalized.get("validation_warnings", []))
            )

            return normalized

        except Exception as e:
            self.logger.error(
                "Content normalization failed",
                content_type=content_data.get("type", "unknown"),
                channel=channel.value,
                error=str(e)
            )

            raise ContentNormalizationError(
                normalizer="ContentNormalizer",
                content_type=content_data.get("type", "unknown"),
                error_details=f"Failed to normalize content: {str(e)}"
            )

    async def _normalize_text_content(
            self,
            content_data: Dict[str, Any],
            channel: ChannelType,
            context: Optional[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Normalize text content."""
        text = content_data.get("text", "")
        normalized = {"text": text}

        if not text:
            normalized["validation_warnings"] = ["Empty text content"]
            return normalized

        # Unicode normalization
        if self.normalize_unicode:
            text = unicodedata.normalize('NFKC', text)
            normalized["normalization_applied"].append("unicode_normalization")

        # HTML entity decoding
        if self.decode_html_entities:
            original_text = text
            text = html.unescape(text)
            if text != original_text:
                normalized["normalization_applied"].append("html_entity_decoding")

        # HTML tag removal
        if self.remove_html_tags:
            original_text = text
            text = re.sub(r'<[^>]+>', '', text)
            if text != original_text:
                normalized["normalization_applied"].append("html_tag_removal")

        # Whitespace normalization
        if self.normalize_whitespace:
            original_text = text
            if self.preserve_formatting:
                # Preserve line breaks but normalize spaces
                lines = text.split('\n')
                lines = [re.sub(r'[ \t]+', ' ', line).strip() for line in lines]
                text = '\n'.join(lines)
            else:
                # Aggressive whitespace cleanup
                text = re.sub(r'\s+', ' ', text).strip()

            if text != original_text:
                normalized["normalization_applied"].append("whitespace_normalization")

        # Length validation and truncation
        if len(text) > self.max_text_length:
            text = text[:self.max_text_length]
            normalized["validation_warnings"].append(f"Text truncated to {self.max_text_length} characters")
            normalized["normalization_applied"].append("length_truncation")

        # Content filtering
        if self.filter_profanity:
            filtered_text = await self._filter_profanity(text)
            if filtered_text != text:
                text = filtered_text
                normalized["normalization_applied"].append("profanity_filtering")

        # Personal information removal
        if self.remove_personal_info:
            cleaned_text = await self._remove_personal_info(text)
            if cleaned_text != text:
                text = cleaned_text
                normalized["normalization_applied"].append("personal_info_removal")

        # Language detection
        if self.auto_detect_language:
            detected_language = await self._detect_language(text)
            if detected_language:
                normalized["language"] = detected_language
                normalized["normalization_applied"].append("language_detection")
            else:
                normalized["language"] = self.default_language

        # Extract and validate URLs
        urls = self.url_pattern.findall(text)
        if urls:
            validated_urls = []
            for url in urls:
                if await self._validate_url(url):
                    validated_urls.append(url)
                else:
                    normalized["validation_warnings"].append(f"Invalid URL detected: {url}")

            if validated_urls:
                normalized["extracted_urls"] = validated_urls

        # Extract entities for reference
        entities = await self._extract_text_entities(text)
        if entities:
            normalized["extracted_entities"] = entities

        normalized["text"] = text
        return normalized

    async def _normalize_media_content(
            self,
            content_data: Dict[str, Any],
            channel: ChannelType,
            context: Optional[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Normalize media content."""
        media_data = content_data.get("media", {})
        normalized = {"media": media_data.copy() if media_data else {}}

        if not media_data:
            normalized["validation_warnings"] = ["Missing media data"]
            return normalized

        # Normalize media URL
        url = media_data.get("url", "")
        if url:
            if self.normalize_media_urls:
                normalized_url = await self._normalize_url(url)
                if normalized_url != url:
                    normalized["media"]["url"] = normalized_url
                    normalized["normalization_applied"].append("url_normalization")

            # Validate URL
            if self.validate_urls:
                if not await self._validate_url(url):
                    normalized["validation_warnings"].append("Invalid media URL")
                elif len(url) > self.max_media_url_length:
                    normalized["validation_warnings"].append("Media URL too long")
        else:
            normalized["validation_warnings"].append("Missing media URL")

        # Normalize MIME type
        mime_type = media_data.get("type", "")
        if mime_type:
            normalized_mime = await self._normalize_mime_type(mime_type, content_data.get("type"))
            if normalized_mime != mime_type:
                normalized["media"]["type"] = normalized_mime
                normalized["normalization_applied"].append("mime_type_normalization")

        # Validate media type compatibility
        if self.validate_media_types:
            content_type = content_data.get("type")
            if not await self._validate_media_type_compatibility(mime_type, content_type):
                normalized["validation_warnings"].append(
                    f"MIME type {mime_type} incompatible with content type {content_type}")

        # Normalize file size
        size_bytes = media_data.get("size_bytes")
        if size_bytes is not None:
            try:
                size_bytes = int(size_bytes)
                normalized["media"]["size_bytes"] = size_bytes

                # Add human-readable size
                normalized["media"]["size_human"] = self._format_file_size(size_bytes)
            except (ValueError, TypeError):
                normalized["validation_warnings"].append("Invalid file size format")

        # Normalize filename
        filename = media_data.get("filename", "")
        if filename:
            normalized_filename = await self._normalize_filename(filename)
            if normalized_filename != filename:
                normalized["media"]["filename"] = normalized_filename
                normalized["normalization_applied"].append("filename_normalization")

        # Extract metadata if enabled
        if self.extract_media_metadata:
            metadata = await self._extract_media_metadata(media_data, content_data.get("type"))
            if metadata:
                normalized["media"]["extracted_metadata"] = metadata

        # Handle caption/alt text
        caption = content_data.get("text", "") or media_data.get("caption", "")
        if caption:
            # Apply text normalization to caption
            caption_normalized = await self._normalize_text_content(
                {"text": caption}, channel, context
            )
            normalized["text"] = caption_normalized.get("text", "")
            if "normalization_applied" in caption_normalized:
                normalized["normalization_applied"].extend(
                    [f"caption_{norm}" for norm in caption_normalized["normalization_applied"]]
                )

        return normalized

    async def _normalize_location_content(
            self,
            content_data: Dict[str, Any],
            channel: ChannelType,
            context: Optional[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Normalize location content."""
        location_data = content_data.get("location", {})
        normalized = {"location": location_data.copy() if location_data else {}}

        if not location_data:
            normalized["validation_warnings"] = ["Missing location data"]
            return normalized

        # Normalize coordinates
        if self.normalize_coordinates:
            latitude = location_data.get("latitude")
            longitude = location_data.get("longitude")

            if latitude is not None and longitude is not None:
                try:
                    # Convert to float and round to reasonable precision
                    lat = round(float(latitude), 6)  # ~10cm precision
                    lon = round(float(longitude), 6)

                    normalized["location"]["latitude"] = lat
                    normalized["location"]["longitude"] = lon
                    normalized["normalization_applied"].append("coordinate_normalization")

                    # Validate coordinate bounds
                    if self.validate_coordinate_bounds:
                        if not (-90 <= lat <= 90):
                            normalized["validation_warnings"].append(f"Invalid latitude: {lat}")
                        if not (-180 <= lon <= 180):
                            normalized["validation_warnings"].append(f"Invalid longitude: {lon}")

                except (ValueError, TypeError):
                    normalized["validation_warnings"].append("Invalid coordinate format")
            else:
                normalized["validation_warnings"].append("Missing coordinates")

        # Normalize accuracy
        accuracy = location_data.get("accuracy_meters")
        if accuracy is not None:
            try:
                accuracy = round(float(accuracy), 2)
                normalized["location"]["accuracy_meters"] = accuracy
            except (ValueError, TypeError):
                normalized["validation_warnings"].append("Invalid accuracy format")

        # Normalize address
        address = location_data.get("address", "")
        if address:
            # Apply text normalization to address
            address_normalized = await self._normalize_text_content(
                {"text": address}, channel, context
            )
            normalized["location"]["address"] = address_normalized.get("text", "")
            if "normalization_applied" in address_normalized:
                normalized["normalization_applied"].extend(
                    [f"address_{norm}" for norm in address_normalized["normalization_applied"]]
                )

        # Normalize place name
        place_name = location_data.get("name", "")
        if place_name:
            place_normalized = await self._normalize_text_content(
                {"text": place_name}, channel, context
            )
            normalized["location"]["name"] = place_normalized.get("text", "")

        return normalized

    async def _normalize_generic_content(
            self,
            content_data: Dict[str, Any],
            channel: ChannelType,
            context: Optional[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Normalize unknown/generic content types."""
        normalized = {}

        # Apply basic normalization to any text fields
        for key, value in content_data.items():
            if isinstance(value, str) and key != "type":
                text_normalized = await self._normalize_text_content(
                    {"text": value}, channel, context
                )
                normalized[key] = text_normalized.get("text", value)
            else:
                normalized[key] = value

        normalized["processing_notes"] = ["Generic content normalization applied"]
        return normalized

    async def _apply_common_normalizations(
            self,
            normalized: Dict[str, Any],
            context: Optional[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Apply normalizations common to all content types."""

        # Add timestamp if missing
        if "timestamp" not in normalized:
            normalized["timestamp"] = datetime.utcnow().isoformat()
            normalized["normalization_applied"].append("timestamp_added")

        # Ensure required fields exist
        if "normalization_applied" not in normalized:
            normalized["normalization_applied"] = []

        if "validation_warnings" not in normalized:
            normalized["validation_warnings"] = []

        if "processing_notes" not in normalized:
            normalized["processing_notes"] = []

        return normalized

    async def _validate_normalized_content(
            self,
            normalized: Dict[str, Any],
            context: Optional[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Validate the normalized content structure."""
        validation = {
            "valid": True,
            "errors": [],
            "warnings": normalized.get("validation_warnings", []),
            "score": 1.0
        }

        try:
            content_type = normalized.get("type")

            # Type-specific validation
            if content_type == "text":
                if not normalized.get("text"):
                    validation["warnings"].append("Empty text content")
                    validation["score"] -= 0.2

            elif content_type in ["image", "audio", "video", "file"]:
                media = normalized.get("media", {})
                if not media.get("url"):
                    validation["errors"].append("Missing media URL")
                    validation["valid"] = False
                    validation["score"] = 0.0

            elif content_type == "location":
                location = normalized.get("location", {})
                if not (location.get("latitude") and location.get("longitude")):
                    validation["errors"].append("Missing location coordinates")
                    validation["valid"] = False
                    validation["score"] = 0.0

            # Adjust score based on warnings
            warning_count = len(validation["warnings"])
            if warning_count > 0:
                validation["score"] = max(0.1, validation["score"] - (warning_count * 0.1))

            return validation

        except Exception as e:
            self.logger.error("Content validation failed", error=str(e))
            return {
                "valid": False,
                "errors": [f"Validation error: {str(e)}"],
                "warnings": [],
                "score": 0.0
            }

    # Helper methods for specific normalization tasks

    async def _filter_profanity(self, text: str) -> str:
        """Filter profanity from text (basic implementation)."""
        # In production, use a proper profanity filter
        profane_words = ["badword1", "badword2"]  # Replace with actual list

        for word in profane_words:
            text = re.sub(re.escape(word), "*" * len(word), text, flags=re.IGNORECASE)

        return text

    async def _remove_personal_info(self, text: str) -> str:
        """Remove personal information from text."""
        # Remove email addresses
        text = self.email_pattern.sub("[EMAIL]", text)

        # Remove phone numbers (basic)
        text = self.phone_pattern.sub("[PHONE]", text)

        # Remove SSN patterns
        text = re.sub(r'\b\d{3}-\d{2}-\d{4}\b', "[SSN]", text)

        # Remove credit card patterns
        text = re.sub(r'\b\d{4}[-\s]?\d{4}[-\s]?\d{4}[-\s]?\d{4}\b', "[CARD]", text)

        return text

    async def _detect_language(self, text: str) -> Optional[str]:
        """Detect language of text (placeholder implementation)."""
        # In production, use proper language detection
        if len(text) < 10:
            return None

        # Very basic detection based on character patterns
        if re.search(r'[àáâãäåæçèéêëìíîïðñòóôõöøùúûüýþÿ]', text, re.IGNORECASE):
            return "es"  # Spanish/French/European languages
        elif re.search(r'[αβγδεζηθικλμνξοπρστυφχψω]', text, re.IGNORECASE):
            return "el"  # Greek
        elif re.search(r'[а-яё]', text, re.IGNORECASE):
            return "ru"  # Russian
        elif re.search(r'[一-龯]', text):
            return "zh"  # Chinese
        elif re.search(r'[ひらがなカタカナ]', text):
            return "ja"  # Japanese
        else:
            return "en"  # Default to English

    async def _validate_url(self, url: str) -> bool:
        """Validate URL format."""
        try:
            result = urlparse(url)
            return all([result.scheme, result.netloc]) and result.scheme in ['http', 'https']
        except Exception:
            return False

    async def _normalize_url(self, url: str) -> str:
        """Normalize URL format."""
        try:
            # Basic URL cleanup
            url = url.strip()

            # Remove tracking parameters
            parsed = urlparse(url)
            query_params = parse_qs(parsed.query)

            # Remove common tracking parameters
            tracking_params = ['utm_source', 'utm_medium', 'utm_campaign', 'fbclid', 'gclid']
            for param in tracking_params:
                query_params.pop(param, None)

            # Reconstruct URL without tracking params
            from urllib.parse import urlencode, urlunparse
            clean_query = urlencode(query_params, doseq=True)
            clean_url = urlunparse((
                parsed.scheme, parsed.netloc, parsed.path,
                parsed.params, clean_query, parsed.fragment
            ))

            return clean_url

        except Exception:
            return url

    async def _normalize_mime_type(self, mime_type: str, content_type: str) -> str:
        """Normalize MIME type format."""
        mime_type = mime_type.lower().strip()

        # Common MIME type corrections
        mime_corrections = {
            "jpeg": "image/jpeg",
            "jpg": "image/jpeg",
            "png": "image/png",
            "gif": "image/gif",
            "mp4": "video/mp4",
            "mp3": "audio/mpeg",
            "pdf": "application/pdf"
        }

        if mime_type in mime_corrections:
            return mime_corrections[mime_type]

        # Ensure proper format
        if '/' not in mime_type:
            if content_type == "image":
                return f"image/{mime_type}"
            elif content_type == "audio":
                return f"audio/{mime_type}"
            elif content_type == "video":
                return f"video/{mime_type}"
            else:
                return f"application/{mime_type}"

        return mime_type

    async def _validate_media_type_compatibility(self, mime_type: str, content_type: str) -> bool:
        """Validate MIME type compatibility with content type."""
        if not mime_type or not content_type:
            return False

        type_prefixes = {
            "image": "image/",
            "audio": "audio/",
            "video": "video/",
            "file": ""  # Files can have any MIME type
        }

        expected_prefix = type_prefixes.get(content_type)
        if expected_prefix is None:
            return True  # Unknown content type, allow any MIME type

        if expected_prefix == "":
            return True  # File type allows any MIME type

        return mime_type.startswith(expected_prefix)

    async def _normalize_filename(self, filename: str) -> str:
        """Normalize filename."""
        # Remove potentially dangerous characters
        filename = re.sub(r'[<>:"/\\|?*]', '_', filename)

        # Limit length
        if len(filename) > 255:
            name, ext = filename.rsplit('.', 1) if '.' in filename else (filename, '')
            name = name[:255 - len(ext) - 1]
            filename = f"{name}.{ext}" if ext else name

        return filename

    async def _extract_media_metadata(self, media_data: Dict[str, Any], content_type: str) -> Dict[str, Any]:
        """Extract additional metadata from media data."""
        metadata = {}

        # File extension from filename or URL
        filename = media_data.get("filename", "")
        url = media_data.get("url", "")

        if filename:
            ext = filename.split('.')[-1].lower() if '.' in filename else ""
            if ext:
                metadata["file_extension"] = ext
        elif url:
            ext = url.split('.')[-1].split('?')[0].lower()
            if ext and len(ext) <= 4:
                metadata["file_extension"] = ext

        # Estimated download time for different connection speeds
        size_bytes = media_data.get("size_bytes")
        if size_bytes:
            # Estimate download times (in seconds)
            speeds = {
                "slow_3g": 50 * 1024,  # 50 KB/s
                "fast_3g": 200 * 1024,  # 200 KB/s
                "wifi": 1024 * 1024  # 1 MB/s
            }

            download_estimates = {}
            for speed_name, speed_bps in speeds.items():
                download_estimates[speed_name] = round(size_bytes / speed_bps, 1)

            metadata["estimated_download_time"] = download_estimates

        return metadata

    async def _extract_text_entities(self, text: str) -> Dict[str, List[str]]:
        """Extract entities from text content."""
        entities = {}

        # URLs
        urls = self.url_pattern.findall(text)
        if urls:
            entities["urls"] = urls

        # Email addresses
        emails = self.email_pattern.findall(text)
        if emails:
            entities["emails"] = emails

        # Phone numbers
        phones = self.phone_pattern.findall(text)
        if phones:
            entities["phones"] = [p.strip() for p in phones]

        # Hashtags
        hashtags = re.findall(r'#\w+', text)
        if hashtags:
            entities["hashtags"] = hashtags

        # Mentions
        mentions = re.findall(r'@\w+', text)
        if mentions:
            entities["mentions"] = mentions

        return entities

    def _format_file_size(self, size_bytes: int) -> str:
        """Format file size in human-readable format."""
        for unit in ['B', 'KB', 'MB', 'GB']:
            if size_bytes < 1024:
                return f"{size_bytes:.1f} {unit}"
            size_bytes /= 1024
        return f"{size_bytes:.1f} TB"