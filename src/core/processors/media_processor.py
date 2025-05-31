"""
Specialized processor for media message content (images, audio, video, files).

This module handles media-specific processing including metadata extraction,
content analysis, format validation, and security scanning.
"""

import re
import mimetypes
from datetime import datetime
from typing import List, Dict, Any, Optional, Tuple
from urllib.parse import urlparse
import asyncio

from src.core.processors.base_processor import (
    BaseProcessor,
    ProcessingContext,
    ProcessingResult
)
from src.models.types import MessageContent, MessageType, MediaContent
from src.core.exceptions import ProcessingError, ProcessingValidationError


class MediaProcessor(BaseProcessor):
    """Processor for media message content."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)

        # Media processing configuration
        self.max_file_size_mb = self.config.get("max_file_size_mb", 50)
        self.max_file_size_bytes = self.max_file_size_mb * 1024 * 1024

        # Supported formats
        self.supported_image_formats = self.config.get(
            "supported_image_formats",
            ["image/jpeg", "image/png", "image/gif", "image/webp", "image/bmp"]
        )
        self.supported_audio_formats = self.config.get(
            "supported_audio_formats",
            ["audio/mpeg", "audio/wav", "audio/ogg", "audio/mp4", "audio/webm"]
        )
        self.supported_video_formats = self.config.get(
            "supported_video_formats",
            ["video/mp4", "video/webm", "video/avi", "video/mov", "video/wmv"]
        )
        self.supported_document_formats = self.config.get(
            "supported_document_formats",
            ["application/pdf", "text/plain", "application/msword",
             "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
             "application/vnd.ms-excel",
             "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"]
        )

        # Feature flags
        self.enable_metadata_extraction = self.config.get("enable_metadata_extraction", True)
        self.enable_content_analysis = self.config.get("enable_content_analysis", True)
        self.enable_virus_scanning = self.config.get("enable_virus_scanning", False)
        self.enable_ocr = self.config.get("enable_ocr", False)
        self.enable_transcription = self.config.get("enable_transcription", False)

        # Security settings
        self.quarantine_suspicious_files = self.config.get("quarantine_suspicious_files", True)
        self.block_executable_files = self.config.get("block_executable_files", True)

        # Analysis timeouts
        self.analysis_timeout_seconds = self.config.get("analysis_timeout_seconds", 30)

        # Dangerous file extensions
        self.dangerous_extensions = {
            ".exe", ".bat", ".cmd", ".com", ".pif", ".scr", ".vbs", ".js", ".jar",
            ".msi", ".dll", ".sys", ".drv", ".bin", ".run", ".deb", ".rpm"
        }

        # Suspicious MIME types
        self.suspicious_mime_types = {
            "application/x-executable", "application/x-msdownload",
            "application/x-dosexec", "application/x-winexe"
        }

        self.logger.info(
            "Media processor initialized",
            max_file_size_mb=self.max_file_size_mb,
            supported_formats={
                "images": len(self.supported_image_formats),
                "audio": len(self.supported_audio_formats),
                "video": len(self.supported_video_formats),
                "documents": len(self.supported_document_formats)
            },
            features_enabled={
                "metadata_extraction": self.enable_metadata_extraction,
                "content_analysis": self.enable_content_analysis,
                "virus_scanning": self.enable_virus_scanning,
                "ocr": self.enable_ocr,
                "transcription": self.enable_transcription
            }
        )

    @property
    def supported_message_types(self) -> List[MessageType]:
        return [MessageType.IMAGE, MessageType.AUDIO, MessageType.VIDEO, MessageType.FILE]

    @property
    def processor_name(self) -> str:
        return "MediaProcessor"

    async def process(
            self,
            content: MessageContent,
            context: ProcessingContext
    ) -> ProcessingResult:
        """Process media message content."""
        start_time = datetime.utcnow()

        try:
            # Validate input
            if not await self.validate_input(content, context):
                processing_time = self._measure_processing_time(start_time)
                self.update_metrics(False, processing_time, content.type.value, "VALIDATION_FAILED")

                return self._create_result(
                    content,
                    processing_time,
                    success=False,
                    errors=["Input validation failed"]
                )

            media = content.media
            if not media:
                processing_time = self._measure_processing_time(start_time)
                self.update_metrics(False, processing_time, content.type.value, "NO_MEDIA_CONTENT")

                return self._create_result(
                    content,
                    processing_time,
                    success=False,
                    errors=["No media content to process"]
                )

            # Initialize result tracking
            processing_results = {}
            warnings = []

            # Step 1: Basic media validation and metadata extraction
            media_info = await self.extract_media_metadata(media, content.type)
            processing_results["media_info"] = media_info

            # Step 2: Security analysis
            security_analysis = await self.analyze_media_security(media, content.type, context)
            processing_results["security_analysis"] = security_analysis

            if not security_analysis.get("safe", True):
                warnings.extend(security_analysis.get("warnings", []))

            # Step 3: Content analysis based on media type
            content_analysis = {}

            if content.type == MessageType.IMAGE:
                content_analysis = await self.analyze_image_content(media, context)
            elif content.type == MessageType.AUDIO:
                content_analysis = await self.analyze_audio_content(media, context)
            elif content.type == MessageType.VIDEO:
                content_analysis = await self.analyze_video_content(media, context)
            elif content.type == MessageType.FILE:
                content_analysis = await self.analyze_file_content(media, context)

            processing_results["content_analysis"] = content_analysis

            # Step 4: Extract entities and metadata
            entities = await self.extract_entities(content, context)

            # Step 5: Quality assessment
            quality_assessment = await self.assess_media_quality(media, content.type, context)
            processing_results["quality_assessment"] = quality_assessment

            # Step 6: Generate thumbnails or previews if applicable
            preview_info = await self.generate_preview_info(media, content.type)
            processing_results["preview_info"] = preview_info

            # Create processed content with enhanced metadata
            processed_media = MediaContent(
                url=media.url,
                type=media.type,
                size_bytes=getattr(media, 'size_bytes', 0),
                alt_text=getattr(media, 'alt_text', ''),
                thumbnail_url=getattr(media, 'thumbnail_url', ''),
                filename=getattr(media, 'filename', '')
            )

            processed_content = MessageContent(
                type=content.type,
                text=content.text,
                media=processed_media,
                location=content.location,
                quick_replies=content.quick_replies,
                buttons=content.buttons
            )

            # Determine content categories
            categories = await self.categorize_media_content(content.type, content_analysis, context)

            # Calculate processing time
            processing_time = self._measure_processing_time(start_time)

            # Update metrics
            self.update_metrics(True, processing_time, content.type.value)

            # Create comprehensive result
            result = ProcessingResult(
                success=True,
                original_content=content,
                processed_content=processed_content,

                entities=entities,
                extracted_data={
                    "media_metadata": media_info,
                    "content_analysis": content_analysis,
                    "quality_assessment": quality_assessment,
                    "preview_info": preview_info,
                    **processing_results
                },

                content_categories=categories,
                content_tags=self._generate_content_tags(content.type, content_analysis),

                quality_score=quality_assessment.get("overall_score", 0.5),
                safety_flags=security_analysis.get("flags", []),
                moderation_required=not security_analysis.get("safe", True),

                processing_time_ms=processing_time,
                processor_version=self.processor_version,
                warnings=warnings,

                metadata={
                    "media_type": content.type.value,
                    "mime_type": media.type,
                    "file_size_bytes": getattr(media, 'size_bytes', 0),
                    "processing_features": {
                        "metadata_extracted": self.enable_metadata_extraction,
                        "content_analyzed": self.enable_content_analysis,
                        "security_scanned": True,
                        "preview_generated": bool(preview_info)
                    },
                    "security_analysis": security_analysis
                }
            )

            self.logger.info(
                "Media processing completed",
                media_type=content.type.value,
                mime_type=media.type,
                file_size=getattr(media, 'size_bytes', 0),
                categories=categories,
                quality_score=quality_assessment.get("overall_score"),
                security_flags=len(security_analysis.get("flags", [])),
                processing_time_ms=processing_time
            )

            return result

        except Exception as e:
            processing_time = self._measure_processing_time(start_time)
            self.update_metrics(False, processing_time, content.type.value, "PROCESSING_ERROR")

            self.logger.error(
                "Media processing failed",
                error=str(e),
                media_type=content.type.value,
                error_type=type(e).__name__
            )

            return self._create_result(
                content,
                processing_time,
                success=False,
                errors=[f"Processing failed: {str(e)}"]
            )

    async def _validate_type_specific(
            self,
            content: MessageContent,
            context: ProcessingContext
    ) -> bool:
        """Validate media-specific content."""
        if not content.media:
            return False

        media = content.media

        # Check file size
        if hasattr(media, 'size_bytes') and media.size_bytes > self.max_file_size_bytes:
            self.logger.warning(
                "Media file too large",
                size_bytes=media.size_bytes,
                max_size_bytes=self.max_file_size_bytes
            )
            return False

        # Check MIME type support
        if not self._is_mime_type_supported(media.type, content.type):
            self.logger.warning(
                "Unsupported media format",
                mime_type=media.type,
                message_type=content.type.value
            )
            return False

        # Check URL format
        if not self._is_valid_url(media.url):
            self.logger.warning(
                "Invalid media URL",
                url=media.url
            )
            return False

        return True

    def _is_mime_type_supported(self, mime_type: str, message_type: MessageType) -> bool:
        """Check if MIME type is supported for the message type."""
        if message_type == MessageType.IMAGE:
            return mime_type in self.supported_image_formats
        elif message_type == MessageType.AUDIO:
            return mime_type in self.supported_audio_formats
        elif message_type == MessageType.VIDEO:
            return mime_type in self.supported_video_formats
        elif message_type == MessageType.FILE:
            return mime_type in self.supported_document_formats
        return False

    def _is_valid_url(self, url: str) -> bool:
        """Validate URL format."""
        try:
            result = urlparse(url)
            return all([result.scheme, result.netloc])
        except Exception:
            return False

    async def extract_media_metadata(
            self,
            media: MediaContent,
            message_type: MessageType
    ) -> Dict[str, Any]:
        """Extract metadata from media content."""
        if not self.enable_metadata_extraction:
            return {}

        try:
            metadata = {
                "url": media.url,
                "mime_type": media.type,
                "size_bytes": getattr(media, 'size_bytes', 0),
                "filename": getattr(media, 'filename', ''),
                "alt_text": getattr(media, 'alt_text', ''),
                "message_type": message_type.value
            }

            # Extract file extension
            if hasattr(media, 'filename') and media.filename:
                metadata["file_extension"] = self._get_file_extension(media.filename)
            else:
                # Try to get extension from URL
                parsed_url = urlparse(media.url)
                if parsed_url.path:
                    metadata["file_extension"] = self._get_file_extension(parsed_url.path)

            # Add format-specific metadata
            if message_type == MessageType.IMAGE:
                metadata.update(await self._extract_image_metadata(media))
            elif message_type == MessageType.AUDIO:
                metadata.update(await self._extract_audio_metadata(media))
            elif message_type == MessageType.VIDEO:
                metadata.update(await self._extract_video_metadata(media))
            elif message_type == MessageType.FILE:
                metadata.update(await self._extract_file_metadata(media))

            return metadata

        except Exception as e:
            self.logger.error(
                "Metadata extraction failed",
                error=str(e),
                media_url=media.url
            )
            return {}

    def _get_file_extension(self, filename: str) -> str:
        """Extract file extension from filename."""
        import os
        return os.path.splitext(filename)[1].lower()

    async def _extract_image_metadata(self, media: MediaContent) -> Dict[str, Any]:
        """Extract image-specific metadata."""
        metadata = {}

        # Basic image info
        metadata["format"] = "image"

        # Try to get image dimensions from URL or other sources
        # In production, you would fetch and analyze the actual image
        if hasattr(media, 'width') and hasattr(media, 'height'):
            metadata["dimensions"] = {
                "width": media.width,
                "height": media.height
            }

        return metadata

    async def _extract_audio_metadata(self, media: MediaContent) -> Dict[str, Any]:
        """Extract audio-specific metadata."""
        metadata = {}

        metadata["format"] = "audio"

        # Try to get audio duration
        if hasattr(media, 'duration_seconds'):
            metadata["duration_seconds"] = media.duration_seconds

        return metadata

    async def _extract_video_metadata(self, media: MediaContent) -> Dict[str, Any]:
        """Extract video-specific metadata."""
        metadata = {}

        metadata["format"] = "video"

        # Try to get video info
        if hasattr(media, 'duration_seconds'):
            metadata["duration_seconds"] = media.duration_seconds

        if hasattr(media, 'width') and hasattr(media, 'height'):
            metadata["dimensions"] = {
                "width": media.width,
                "height": media.height
            }

        return metadata

    async def _extract_file_metadata(self, media: MediaContent) -> Dict[str, Any]:
        """Extract file-specific metadata."""
        metadata = {}

        metadata["format"] = "document"

        # Detect document type from MIME type
        if "pdf" in media.type:
            metadata["document_type"] = "pdf"
        elif "word" in media.type:
            metadata["document_type"] = "word"
        elif "excel" in media.type or "spreadsheet" in media.type:
            metadata["document_type"] = "spreadsheet"
        elif "text" in media.type:
            metadata["document_type"] = "text"
        else:
            metadata["document_type"] = "unknown"

        return metadata

    async def analyze_media_security(
            self,
            media: MediaContent,
            message_type: MessageType,
            context: ProcessingContext
    ) -> Dict[str, Any]:
        """Analyze media for security issues."""
        security_result = {
            "safe": True,
            "flags": [],
            "warnings": [],
            "score": 1.0
        }

        try:
            # Check file extension
            if hasattr(media, 'filename') and media.filename:
                ext = self._get_file_extension(media.filename)
                if ext in self.dangerous_extensions:
                    security_result["safe"] = False
                    security_result["flags"].append("dangerous_file_extension")
                    security_result["warnings"].append(f"Potentially dangerous file extension: {ext}")
                    security_result["score"] -= 0.5

            # Check MIME type
            if media.type in self.suspicious_mime_types:
                security_result["safe"] = False
                security_result["flags"].append("suspicious_mime_type")
                security_result["warnings"].append(f"Suspicious MIME type: {media.type}")
                security_result["score"] -= 0.4

            # Check file size (very large files might be suspicious)
            if hasattr(media, 'size_bytes') and media.size_bytes > self.max_file_size_bytes:
                security_result["flags"].append("file_too_large")
                security_result["warnings"].append("File exceeds maximum allowed size")
                security_result["score"] -= 0.2

            # Check URL for suspicious patterns
            url_security = await self._analyze_url_security(media.url)
            if not url_security["safe"]:
                security_result["safe"] = False
                security_result["flags"].extend(url_security["flags"])
                security_result["warnings"].extend(url_security["warnings"])
                security_result["score"] = min(security_result["score"], url_security["score"])

            # Virus scanning (if enabled)
            if self.enable_virus_scanning:
                virus_scan_result = await self._perform_virus_scan(media)
                if not virus_scan_result["clean"]:
                    security_result["safe"] = False
                    security_result["flags"].append("virus_detected")
                    security_result["warnings"].append("Virus or malware detected")
                    security_result["score"] = 0.0

            security_result["score"] = max(0.0, security_result["score"])

            return security_result

        except Exception as e:
            self.logger.error(
                "Media security analysis failed",
                error=str(e),
                media_url=media.url
            )
            return {
                "safe": True,  # Default to safe on error
                "flags": ["analysis_error"],
                "warnings": [f"Security analysis failed: {str(e)}"],
                "score": 0.5
            }

    async def _analyze_url_security(self, url: str) -> Dict[str, Any]:
        """Analyze URL for security issues."""
        result = {
            "safe": True,
            "flags": [],
            "warnings": [],
            "score": 1.0
        }

        try:
            parsed = urlparse(url)

            # Check for suspicious domains
            suspicious_tlds = [".tk", ".ml", ".ga", ".cf"]
            if any(parsed.netloc.endswith(tld) for tld in suspicious_tlds):
                result["flags"].append("suspicious_domain")
                result["warnings"].append("Domain uses suspicious TLD")
                result["score"] -= 0.2

            # Check for IP addresses instead of domains
            import ipaddress
            try:
                ipaddress.ip_address(parsed.netloc.split(':')[0])
                result["flags"].append("ip_address_url")
                result["warnings"].append("URL uses IP address instead of domain")
                result["score"] -= 0.3
            except ValueError:
                pass  # Not an IP address, which is good

            # Check for URL shorteners
            shortener_domains = ["bit.ly", "tinyurl.com", "t.co", "goo.gl", "ow.ly"]
            if any(domain in parsed.netloc for domain in shortener_domains):
                result["flags"].append("url_shortener")
                result["warnings"].append("URL uses shortening service")
                result["score"] -= 0.1

            # Check for suspicious patterns in path
            if re.search(r'(\.\./|%2e%2e/)', url, re.IGNORECASE):
                result["flags"].append("path_traversal")
                result["warnings"].append("URL contains path traversal patterns")
                result["score"] -= 0.4
                result["safe"] = False

            return result

        except Exception as e:
            self.logger.error("URL security analysis failed", error=str(e))
            return result

    async def _perform_virus_scan(self, media: MediaContent) -> Dict[str, Any]:
        """Perform virus scanning (placeholder implementation)."""
        # In production, integrate with actual antivirus APIs
        return {
            "clean": True,
            "scan_performed": False,
            "reason": "Virus scanning not implemented"
        }

    async def analyze_image_content(
            self,
            media: MediaContent,
            context: ProcessingContext
    ) -> Dict[str, Any]:
        """Analyze image content."""
        analysis = {
            "content_type": "image",
            "analysis_performed": self.enable_content_analysis
        }

        if not self.enable_content_analysis:
            return analysis

        try:
            # OCR text extraction (if enabled)
            if self.enable_ocr:
                ocr_result = await self._extract_text_from_image(media)
                analysis["ocr"] = ocr_result

            # Basic image analysis (placeholder)
            analysis["features"] = {
                "has_text": self.enable_ocr and bool(analysis.get("ocr", {}).get("text")),
                "estimated_complexity": "medium",  # Would be determined by actual analysis
                "estimated_quality": "good"  # Would be determined by actual analysis
            }

            return analysis

        except Exception as e:
            self.logger.error("Image content analysis failed", error=str(e))
            return {"content_type": "image", "error": str(e)}

    async def _extract_text_from_image(self, media: MediaContent) -> Dict[str, Any]:
        """Extract text from image using OCR (placeholder)."""
        # In production, integrate with OCR services like Tesseract, Google Vision, etc.
        return {
            "text": "",
            "confidence": 0.0,
            "language": "en",
            "performed": False,
            "reason": "OCR not implemented"
        }

    async def analyze_audio_content(
            self,
            media: MediaContent,
            context: ProcessingContext
    ) -> Dict[str, Any]:
        """Analyze audio content."""
        analysis = {
            "content_type": "audio",
            "analysis_performed": self.enable_content_analysis
        }

        if not self.enable_content_analysis:
            return analysis

        try:
            # Speech-to-text transcription (if enabled)
            if self.enable_transcription:
                transcription_result = await self._transcribe_audio(media)
                analysis["transcription"] = transcription_result

            # Basic audio analysis
            analysis["features"] = {
                "has_speech": self.enable_transcription and bool(analysis.get("transcription", {}).get("text")),
                "estimated_duration": getattr(media, 'duration_seconds', 0),
                "estimated_quality": "good"
            }

            return analysis

        except Exception as e:
            self.logger.error("Audio content analysis failed", error=str(e))
            return {"content_type": "audio", "error": str(e)}

    async def _transcribe_audio(self, media: MediaContent) -> Dict[str, Any]:
        """Transcribe audio to text (placeholder)."""
        # In production, integrate with speech-to-text services
        return {
            "text": "",
            "confidence": 0.0,
            "language": "en",
            "performed": False,
            "reason": "Transcription not implemented"
        }

    async def analyze_video_content(
            self,
            media: MediaContent,
            context: ProcessingContext
    ) -> Dict[str, Any]:
        """Analyze video content."""
        analysis = {
            "content_type": "video",
            "analysis_performed": self.enable_content_analysis
        }

        if not self.enable_content_analysis:
            return analysis

        try:
            # Video analysis (placeholder)
            analysis["features"] = {
                "estimated_duration": getattr(media, 'duration_seconds', 0),
                "has_audio": True,  # Would be determined by actual analysis
                "estimated_quality": "good"
            }

            return analysis

        except Exception as e:
            self.logger.error("Video content analysis failed", error=str(e))
            return {"content_type": "video", "error": str(e)}

    async def analyze_file_content(
            self,
            media: MediaContent,
            context: ProcessingContext
    ) -> Dict[str, Any]:
        """Analyze file content."""
        analysis = {
            "content_type": "file",
            "analysis_performed": self.enable_content_analysis
        }

        if not self.enable_content_analysis:
            return analysis

        try:
            # File analysis based on type
            if "pdf" in media.type:
                analysis.update(await self._analyze_pdf_content(media))
            elif "text" in media.type:
                analysis.update(await self._analyze_text_file_content(media))
            elif "word" in media.type:
                analysis.update(await self._analyze_word_document(media))
            elif "excel" in media.type or "spreadsheet" in media.type:
                analysis.update(await self._analyze_spreadsheet(media))

            return analysis

        except Exception as e:
            self.logger.error("File content analysis failed", error=str(e))
            return {"content_type": "file", "error": str(e)}

    async def _analyze_pdf_content(self, media: MediaContent) -> Dict[str, Any]:
        """Analyze PDF content (placeholder)."""
        return {
            "document_type": "pdf",
            "estimated_pages": 0,
            "has_text": False,
            "has_images": False,
            "analysis_note": "PDF analysis not implemented"
        }

    async def _analyze_text_file_content(self, media: MediaContent) -> Dict[str, Any]:
        """Analyze text file content (placeholder)."""
        return {
            "document_type": "text",
            "estimated_lines": 0,
            "encoding": "utf-8",
            "analysis_note": "Text file analysis not implemented"
        }

    async def _analyze_word_document(self, media: MediaContent) -> Dict[str, Any]:
        """Analyze Word document (placeholder)."""
        return {
            "document_type": "word",
            "estimated_pages": 0,
            "has_images": False,
            "analysis_note": "Word document analysis not implemented"
        }

    async def _analyze_spreadsheet(self, media: MediaContent) -> Dict[str, Any]:
        """Analyze spreadsheet content (placeholder)."""
        return {
            "document_type": "spreadsheet",
            "estimated_sheets": 0,
            "estimated_rows": 0,
            "analysis_note": "Spreadsheet analysis not implemented"
        }

    async def assess_media_quality(
            self,
            media: MediaContent,
            message_type: MessageType,
            context: ProcessingContext
    ) -> Dict[str, Any]:
        """Assess the quality of media content."""
        try:
            assessment = {
                "overall_score": 0.5,
                "dimensions": {},
                "issues": [],
                "recommendations": []
            }

            # File size assessment
            size_score = await self._assess_file_size_quality(media)
            assessment["dimensions"]["file_size"] = size_score

            # Format appropriateness
            format_score = await self._assess_format_quality(media, message_type)
            assessment["dimensions"]["format"] = format_score

            # Accessibility
            accessibility_score = await self._assess_accessibility(media, message_type)
            assessment["dimensions"]["accessibility"] = accessibility_score

            # Calculate overall score
            scores = [size_score, format_score, accessibility_score]
            assessment["overall_score"] = sum(scores) / len(scores)

            # Generate recommendations
            if size_score < 0.6:
                assessment["issues"].append("file_size_issues")
                if hasattr(media, 'size_bytes') and media.size_bytes > 10 * 1024 * 1024:
                    assessment["recommendations"].append("Consider compressing large files")

            if format_score < 0.6:
                assessment["issues"].append("format_issues")
                assessment["recommendations"].append("Use more widely supported formats")

            if accessibility_score < 0.6:
                assessment["issues"].append("accessibility_issues")
                assessment["recommendations"].append("Add alt text and descriptions")

            return assessment

        except Exception as e:
            self.logger.error("Media quality assessment failed", error=str(e))
            return {"overall_score": 0.5, "error": str(e)}

    async def _assess_file_size_quality(self, media: MediaContent) -> float:
        """Assess file size appropriateness."""
        if not hasattr(media, 'size_bytes') or not media.size_bytes:
            return 0.7  # Unknown size gets neutral score

        size_mb = media.size_bytes / (1024 * 1024)

        # Optimal sizes vary by media type
        if size_mb < 1:
            return 1.0  # Small files are always good
        elif size_mb < 5:
            return 0.9
        elif size_mb < 10:
            return 0.7
        elif size_mb < 25:
            return 0.5
        else:
            return 0.3  # Very large files

    async def _assess_format_quality(self, media: MediaContent, message_type: MessageType) -> float:
        """Assess format appropriateness."""
        # Modern, widely supported formats get higher scores
        format_scores = {
            # Images
            "image/jpeg": 0.9,
            "image/png": 0.9,
            "image/webp": 1.0,
            "image/gif": 0.8,
            "image/bmp": 0.6,

            # Audio
            "audio/mpeg": 0.9,
            "audio/mp4": 0.9,
            "audio/ogg": 0.8,
            "audio/wav": 0.7,

            # Video
            "video/mp4": 1.0,
            "video/webm": 0.9,
            "video/avi": 0.6,
            "video/mov": 0.7,

            # Documents
            "application/pdf": 1.0,
            "text/plain": 0.9,
            "application/vnd.openxmlformats-officedocument.wordprocessingml.document": 0.8,
            "application/msword": 0.6
        }

        return format_scores.get(media.type, 0.5)

    async def _assess_accessibility(self, media: MediaContent, message_type: MessageType) -> float:
        """Assess accessibility features."""
        score = 0.5  # Base score

        # Check for alt text
        if hasattr(media, 'alt_text') and media.alt_text:
            score += 0.3

        # Check for descriptive filename
        if hasattr(media, 'filename') and media.filename:
            if not re.match(r'^[A-Z0-9_-]+\.[a-z0-9]+$', media.filename, re.IGNORECASE):
                score += 0.2  # Descriptive filename

        return min(1.0, score)

    async def generate_preview_info(
            self,
            media: MediaContent,
            message_type: MessageType
    ) -> Dict[str, Any]:
        """Generate preview information for media."""
        preview_info = {
            "preview_available": False,
            "preview_type": None,
            "preview_url": None
        }

        try:
            if message_type == MessageType.IMAGE:
                # For images, the original can serve as preview
                preview_info.update({
                    "preview_available": True,
                    "preview_type": "image",
                    "preview_url": media.url,
                    "thumbnail_available": bool(getattr(media, 'thumbnail_url', None))
                })

            elif message_type == MessageType.VIDEO:
                # Video might have thumbnail
                if hasattr(media, 'thumbnail_url') and media.thumbnail_url:
                    preview_info.update({
                        "preview_available": True,
                        "preview_type": "image",
                        "preview_url": media.thumbnail_url
                    })

            elif message_type == MessageType.FILE:
                # Files might have icon based on type
                preview_info.update({
                    "preview_available": True,
                    "preview_type": "icon",
                    "icon_type": self._get_file_icon_type(media.type)
                })

            return preview_info

        except Exception as e:
            self.logger.error("Preview generation failed", error=str(e))
            return preview_info

    def _get_file_icon_type(self, mime_type: str) -> str:
        """Get appropriate icon type for file."""
        if "pdf" in mime_type:
            return "pdf"
        elif "word" in mime_type:
            return "document"
        elif "excel" in mime_type or "spreadsheet" in mime_type:
            return "spreadsheet"
        elif "text" in mime_type:
            return "text"
        else:
            return "file"

    async def categorize_media_content(
            self,
            message_type: MessageType,
            content_analysis: Dict[str, Any],
            context: ProcessingContext
    ) -> List[str]:
        """Categorize media content based on analysis."""
        categories = [message_type.value]  # Always include the media type

        try:
            # Add categories based on content analysis
            if content_analysis.get("ocr", {}).get("text"):
                categories.append("contains_text")

            if content_analysis.get("transcription", {}).get("text"):
                categories.append("contains_speech")

            # Add context-based categories
            if context.channel:
                categories.append(f"{context.channel}_media")

            # Add business categories
            if message_type in [MessageType.FILE]:
                categories.append("document_sharing")
            elif message_type in [MessageType.IMAGE]:
                categories.append("visual_content")
            elif message_type in [MessageType.AUDIO, MessageType.VIDEO]:
                categories.append("multimedia_content")

            return list(set(categories))

        except Exception as e:
            self.logger.error("Media categorization failed", error=str(e))
            return [message_type.value]

    def _generate_content_tags(
            self,
            message_type: MessageType,
            content_analysis: Dict[str, Any]
    ) -> List[str]:
        """Generate tags for media content."""
        tags = [message_type.value]

        try:
            # Add format-specific tags
            if message_type == MessageType.IMAGE:
                tags.extend(["visual", "image"])
            elif message_type == MessageType.AUDIO:
                tags.extend(["audio", "sound"])
            elif message_type == MessageType.VIDEO:
                tags.extend(["video", "multimedia"])
            elif message_type == MessageType.FILE:
                tags.extend(["document", "file"])

            # Add analysis-based tags
            if content_analysis.get("ocr", {}).get("text"):
                tags.append("text_content")

            if content_analysis.get("transcription", {}).get("text"):
                tags.append("speech_content")

            return tags[:10]  # Limit to 10 tags

        except Exception as e:
            self.logger.error("Tag generation failed", error=str(e))
            return [message_type.value]