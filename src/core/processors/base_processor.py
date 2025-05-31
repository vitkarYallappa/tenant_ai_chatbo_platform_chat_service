"""
Abstract base class for message processors with common processing interface.

This module provides the foundation for all message type processors with standardized
processing workflows, entity extraction, and result formatting.
"""
import re
from abc import ABC, abstractmethod
from datetime import datetime
from typing import Dict, Any, List, Optional, Union
from pydantic import BaseModel, Field
import structlog

from src.models.types import MessageContent, MessageType, TenantId, UserId

logger = structlog.get_logger()


class ProcessingContext(BaseModel):
    """Context data for message processing."""

    tenant_id: TenantId
    user_id: UserId
    conversation_id: Optional[str] = None
    session_id: Optional[str] = None

    # Channel information
    channel: str
    channel_metadata: Dict[str, Any] = Field(default_factory=dict)

    # User and conversation context
    user_profile: Dict[str, Any] = Field(default_factory=dict)
    conversation_context: Dict[str, Any] = Field(default_factory=dict)
    conversation_history: List[Dict[str, Any]] = Field(default_factory=list)

    # Processing hints and preferences
    processing_hints: Dict[str, Any] = Field(default_factory=dict)
    language: str = "en"
    timezone: str = "UTC"
    locale: str = "en_US"

    # Request metadata
    request_id: Optional[str] = None
    timestamp: datetime = Field(default_factory=datetime.utcnow)

    # Security and compliance context
    security_context: Dict[str, Any] = Field(default_factory=dict)
    compliance_requirements: List[str] = Field(default_factory=list)

    # Performance and debugging
    debug_mode: bool = False
    performance_tracking: bool = True

    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }


class ProcessingResult(BaseModel):
    """Result of message processing."""

    success: bool = True

    # Processed content
    original_content: MessageContent
    processed_content: Optional[MessageContent] = None
    normalized_content: Optional[Dict[str, Any]] = None

    # Language and content analysis
    detected_language: Optional[str] = None
    language_confidence: Optional[float] = None
    content_encoding: Optional[str] = None

    # Entity extraction and NLP results
    entities: Dict[str, Any] = Field(default_factory=dict)
    extracted_data: Dict[str, Any] = Field(default_factory=dict)

    # Sentiment and intent analysis (placeholder for future AI integration)
    sentiment: Optional[Dict[str, Any]] = None
    intent: Optional[Dict[str, Any]] = None
    confidence_scores: Dict[str, float] = Field(default_factory=dict)

    # Content categorization and tagging
    content_categories: List[str] = Field(default_factory=list)
    content_tags: List[str] = Field(default_factory=list)
    content_topics: List[str] = Field(default_factory=list)

    # Quality and safety assessment
    quality_score: Optional[float] = None
    safety_flags: List[str] = Field(default_factory=list)
    moderation_required: bool = False
    content_warnings: List[str] = Field(default_factory=list)

    # Privacy and compliance
    pii_detected: bool = False
    pii_types: List[str] = Field(default_factory=list)
    compliance_status: str = "compliant"  # compliant, warning, violation

    # Processing metadata
    processing_time_ms: Optional[int] = None
    processor_version: Optional[str] = None
    pipeline_stage: Optional[str] = None

    # Issues and diagnostics
    warnings: List[str] = Field(default_factory=list)
    errors: List[str] = Field(default_factory=list)
    debug_info: Dict[str, Any] = Field(default_factory=dict)

    # Additional metadata
    metadata: Dict[str, Any] = Field(default_factory=dict)

    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }


class ProcessorMetrics(BaseModel):
    """Metrics for processor performance tracking."""

    total_processed: int = 0
    successful_processed: int = 0
    failed_processed: int = 0
    average_processing_time_ms: float = 0.0

    # Content type breakdown
    content_type_counts: Dict[str, int] = Field(default_factory=dict)

    # Performance percentiles
    processing_time_p50: float = 0.0
    processing_time_p95: float = 0.0
    processing_time_p99: float = 0.0

    # Error tracking
    error_types: Dict[str, int] = Field(default_factory=dict)
    last_error: Optional[str] = None
    last_error_at: Optional[datetime] = None

    # Health indicators
    health_status: str = "healthy"  # healthy, degraded, unhealthy
    last_health_check: Optional[datetime] = None


class BaseProcessor(ABC):
    """Abstract base class for message processors."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.logger = structlog.get_logger(self.__class__.__name__)
        self.processor_version = "1.0.0"
        self.metrics = ProcessorMetrics()

        # Processing configuration
        self.max_processing_time_ms = self.config.get("max_processing_time_ms", 30000)
        self.enable_caching = self.config.get("enable_caching", True)
        self.enable_debug = self.config.get("enable_debug", False)

        # Feature flags
        self.enable_entity_extraction = self.config.get("enable_entity_extraction", True)
        self.enable_content_analysis = self.config.get("enable_content_analysis", True)
        self.enable_safety_checks = self.config.get("enable_safety_checks", True)

        self.logger.info(
            "Processor initialized",
            processor=self.processor_name,
            version=self.processor_version,
            supported_types=[t.value for t in self.supported_message_types]
        )

    @property
    @abstractmethod
    def supported_message_types(self) -> List[MessageType]:
        """Return list of supported message types."""
        pass

    @property
    @abstractmethod
    def processor_name(self) -> str:
        """Return processor name."""
        pass

    @abstractmethod
    async def process(
            self,
            content: MessageContent,
            context: ProcessingContext
    ) -> ProcessingResult:
        """
        Process message content.

        Args:
            content: Message content to process
            context: Processing context with conversation and user info

        Returns:
            ProcessingResult with analysis and processed content

        Raises:
            ProcessingError: When processing fails
            ValidationError: When input validation fails
        """
        pass

    async def validate_input(
            self,
            content: MessageContent,
            context: ProcessingContext
    ) -> bool:
        """
        Validate input content for processing.

        Args:
            content: Message content to validate
            context: Processing context

        Returns:
            True if content is valid for this processor
        """
        try:
            # Check if message type is supported
            if content.type not in self.supported_message_types:
                self.logger.warning(
                    "Unsupported message type",
                    processor=self.processor_name,
                    message_type=content.type.value,
                    supported_types=[t.value for t in self.supported_message_types]
                )
                return False

            # Check basic content requirements
            if not self._has_required_content(content):
                self.logger.warning(
                    "Missing required content",
                    processor=self.processor_name,
                    message_type=content.type.value
                )
                return False

            # Perform type-specific validation
            return await self._validate_type_specific(content, context)

        except Exception as e:
            self.logger.error(
                "Input validation failed",
                processor=self.processor_name,
                error=str(e)
            )
            return False

    async def _validate_type_specific(
            self,
            content: MessageContent,
            context: ProcessingContext
    ) -> bool:
        """Override in subclasses for type-specific validation."""
        return True

    def _has_required_content(self, content: MessageContent) -> bool:
        """Check if content has required fields for its type."""
        if content.type == MessageType.TEXT:
            return bool(content.text and content.text.strip())
        elif content.type in [MessageType.IMAGE, MessageType.AUDIO, MessageType.VIDEO, MessageType.FILE]:
            return bool(content.media)
        elif content.type == MessageType.LOCATION:
            return bool(content.location)
        else:
            return True

    async def extract_entities(
            self,
            content: MessageContent,
            context: ProcessingContext
    ) -> Dict[str, Any]:
        """
        Extract entities from message content.

        Args:
            content: Message content
            context: Processing context

        Returns:
            Dictionary of extracted entities
        """
        if not self.enable_entity_extraction:
            return {}

        entities = {}

        try:
            # Basic entity extraction (override in subclasses for advanced extraction)
            if content.text:
                entities.update(await self._extract_text_entities(content.text, context))

            if content.location:
                entities["location"] = {
                    "latitude": content.location.latitude,
                    "longitude": content.location.longitude,
                    "address": getattr(content.location, 'address', None),
                    "accuracy": getattr(content.location, 'accuracy_meters', None)
                }

            if content.media:
                entities["media"] = {
                    "type": content.media.type,
                    "url": content.media.url,
                    "size_bytes": getattr(content.media, 'size_bytes', None),
                    "filename": getattr(content.media, 'filename', None)
                }

            # Add context-based entities
            if context.user_profile:
                entities["user_context"] = {
                    "language": context.language,
                    "timezone": context.timezone,
                    "channel": context.channel
                }

            return entities

        except Exception as e:
            self.logger.error(
                "Entity extraction failed",
                processor=self.processor_name,
                error=str(e)
            )
            return {}

    async def _extract_text_entities(
            self,
            text: str,
            context: ProcessingContext
    ) -> Dict[str, Any]:
        """Extract entities from text (basic implementation)."""
        entities = {}

        try:
            # Basic patterns (extend with NER models in production)
            import re

            # Email extraction
            email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
            emails = re.findall(email_pattern, text)
            if emails:
                entities["emails"] = emails

            # Phone number extraction (simple)
            phone_pattern = r'(\+?[\d\s\-\(\)]{10,})'
            phones = re.findall(phone_pattern, text)
            if phones:
                entities["phones"] = [p.strip() for p in phones if len(re.sub(r'[^\d]', '', p)) >= 10]

            # URL extraction
            url_pattern = r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
            urls = re.findall(url_pattern, text)
            if urls:
                entities["urls"] = urls

            # Date patterns (basic)
            date_patterns = [
                r'\d{1,2}[/-]\d{1,2}[/-]\d{2,4}',  # MM/DD/YYYY or similar
                r'\d{4}[/-]\d{1,2}[/-]\d{1,2}',  # YYYY/MM/DD or similar
            ]
            dates = []
            for pattern in date_patterns:
                dates.extend(re.findall(pattern, text))
            if dates:
                entities["dates"] = dates

            # Money amounts
            money_pattern = r'[\$£€¥]\s*\d+(?:,\d{3})*(?:\.\d{2})?|\d+(?:,\d{3})*(?:\.\d{2})?\s*(?:dollars?|pounds?|euros?|yen)'
            money = re.findall(money_pattern, text, re.IGNORECASE)
            if money:
                entities["money"] = money

            # Numbers
            number_pattern = r'\b\d+(?:,\d{3})*(?:\.\d+)?\b'
            numbers = re.findall(number_pattern, text)
            if numbers:
                entities["numbers"] = numbers

            return entities

        except Exception as e:
            self.logger.error(
                "Text entity extraction failed",
                error=str(e),
                text_length=len(text)
            )
            return {}

    async def analyze_content_safety(
            self,
            content: MessageContent,
            context: ProcessingContext
    ) -> Dict[str, Any]:
        """
        Analyze content for safety issues.

        Args:
            content: Message content to analyze
            context: Processing context

        Returns:
            Dictionary with safety analysis results
        """
        if not self.enable_safety_checks:
            return {"safe": True, "flags": [], "score": 1.0}

        safety_result = {
            "safe": True,
            "flags": [],
            "score": 1.0,
            "categories": {}
        }

        try:
            if content.text:
                text_safety = await self._analyze_text_safety(content.text, context)
                safety_result.update(text_safety)

            if content.media:
                media_safety = await self._analyze_media_safety(content.media, context)
                safety_result["categories"]["media"] = media_safety

            # Overall safety determination
            safety_result["safe"] = len(safety_result["flags"]) == 0

            return safety_result

        except Exception as e:
            self.logger.error(
                "Content safety analysis failed",
                processor=self.processor_name,
                error=str(e)
            )
            return {"safe": True, "flags": [], "score": 1.0, "error": str(e)}

    async def _analyze_text_safety(
            self,
            text: str,
            context: ProcessingContext
    ) -> Dict[str, Any]:
        """Analyze text content for safety issues."""
        flags = []
        score = 1.0

        try:
            text_lower = text.lower()

            # Basic safety checks (extend with proper content moderation)
            if any(word in text_lower for word in ["spam", "scam", "fraud", "phishing"]):
                flags.append("potential_spam")
                score -= 0.3

            if len(re.findall(r'[A-Z]{5,}', text)) > 3:
                flags.append("excessive_caps")
                score -= 0.1

            if len(re.findall(r'[!]{3,}', text)) > 0:
                flags.append("excessive_punctuation")
                score -= 0.1

            # Check for potential PII (basic patterns)
            if re.search(r'\b\d{3}-\d{2}-\d{4}\b', text):  # SSN pattern
                flags.append("potential_pii_ssn")
                score -= 0.2

            if re.search(r'\b\d{16}\b', text):  # Credit card pattern
                flags.append("potential_pii_credit_card")
                score -= 0.2

            # Profanity detection (basic word list)
            profane_words = ["badword1", "badword2"]  # Replace with actual list
            if any(word in text_lower for word in profane_words):
                flags.append("profanity")
                score -= 0.4

            return {
                "flags": flags,
                "score": max(0.0, score),
                "categories": {
                    "text": {
                        "safe": len(flags) == 0,
                        "flags": flags,
                        "score": max(0.0, score)
                    }
                }
            }

        except Exception as e:
            self.logger.error(
                "Text safety analysis failed",
                error=str(e)
            )
            return {"flags": [], "score": 1.0, "categories": {"text": {"error": str(e)}}}

    async def _analyze_media_safety(
            self,
            media: Any,
            context: ProcessingContext
    ) -> Dict[str, Any]:
        """Analyze media content for safety issues."""
        # Placeholder for media safety analysis
        return {
            "safe": True,
            "flags": [],
            "score": 1.0,
            "analyzed": False,
            "reason": "Media safety analysis not implemented"
        }

    def update_metrics(
            self,
            success: bool,
            processing_time_ms: int,
            content_type: str,
            error_type: Optional[str] = None
    ) -> None:
        """Update processor metrics."""
        self.metrics.total_processed += 1

        if success:
            self.metrics.successful_processed += 1
        else:
            self.metrics.failed_processed += 1
            if error_type:
                if error_type not in self.metrics.error_types:
                    self.metrics.error_types[error_type] = 0
                self.metrics.error_types[error_type] += 1
                self.metrics.last_error = error_type
                self.metrics.last_error_at = datetime.utcnow()

        # Update content type counts
        if content_type not in self.metrics.content_type_counts:
            self.metrics.content_type_counts[content_type] = 0
        self.metrics.content_type_counts[content_type] += 1

        # Update average processing time (simple moving average)
        if self.metrics.average_processing_time_ms == 0:
            self.metrics.average_processing_time_ms = processing_time_ms
        else:
            self.metrics.average_processing_time_ms = (
                    self.metrics.average_processing_time_ms * 0.9 + processing_time_ms * 0.1
            )

        # Update health status
        total = self.metrics.total_processed
        success_rate = self.metrics.successful_processed / total if total > 0 else 1.0

        if success_rate >= 0.95:
            self.metrics.health_status = "healthy"
        elif success_rate >= 0.8:
            self.metrics.health_status = "degraded"
        else:
            self.metrics.health_status = "unhealthy"

    def get_metrics(self) -> ProcessorMetrics:
        """Get current processor metrics."""
        return self.metrics.copy()

    def _create_result(
            self,
            content: MessageContent,
            processing_time_ms: int,
            success: bool = True,
            **kwargs
    ) -> ProcessingResult:
        """Create processing result with common fields."""
        return ProcessingResult(
            success=success,
            original_content=content,
            processing_time_ms=processing_time_ms,
            processor_version=self.processor_version,
            **kwargs
        )

    def _measure_processing_time(self, start_time: datetime) -> int:
        """Calculate processing time in milliseconds."""
        return int((datetime.utcnow() - start_time).total_seconds() * 1000)

    async def health_check(self) -> bool:
        """Perform health check for this processor."""
        try:
            # Basic health indicators
            if self.metrics.health_status == "unhealthy":
                return False

            # Check recent error rate
            if self.metrics.total_processed > 10:
                error_rate = self.metrics.failed_processed / self.metrics.total_processed
                if error_rate > 0.2:  # More than 20% error rate
                    return False

            # Check processing time
            if self.metrics.average_processing_time_ms > self.max_processing_time_ms:
                return False

            return True

        except Exception as e:
            self.logger.error(
                "Health check failed",
                processor=self.processor_name,
                error=str(e)
            )
            return False

    def reset_metrics(self) -> None:
        """Reset processor metrics."""
        self.metrics = ProcessorMetrics()
        self.logger.info(
            "Processor metrics reset",
            processor=self.processor_name
        )

    def __str__(self) -> str:
        """String representation of processor."""
        return f"{self.processor_name}(version={self.processor_version})"

    def __repr__(self) -> str:
        """Detailed string representation of processor."""
        return (
            f"{self.__class__.__name__}("
            f"name={self.processor_name}, "
            f"version={self.processor_version}, "
            f"types={[t.value for t in self.supported_message_types]}, "
            f"health={self.metrics.health_status}"
            f")"
        )