"""
Specialized processor for text message content analysis and transformation.

This module handles text-specific processing like language detection, content analysis,
sentiment analysis, and text normalization.
"""

import re
from datetime import datetime
from typing import List, Dict, Any, Tuple, Optional
from collections import Counter

from src.core.processors.base_processor import (
    BaseProcessor,
    ProcessingContext,
    ProcessingResult
)
from src.models.types import MessageContent, MessageType
from src.core.exceptions import ProcessingError, ProcessingValidationError

# Language detection - in production, use proper libraries like langdetect
try:
    from langdetect import detect, LangDetectError

    LANGDETECT_AVAILABLE = True
except ImportError:
    LANGDETECT_AVAILABLE = False


class TextProcessor(BaseProcessor):
    """Processor for text message content."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)

        # Text processing configuration
        self.max_text_length = self.config.get("max_text_length", 10000)
        self.min_text_length = self.config.get("min_text_length", 1)
        self.supported_languages = self.config.get(
            "supported_languages",
            ["en", "es", "fr", "de", "it", "pt", "ja", "ko", "zh", "ar", "hi", "ru"]
        )

        # Feature flags
        self.enable_language_detection = self.config.get("enable_language_detection", True)
        self.enable_sentiment_analysis = self.config.get("enable_sentiment_analysis", True)
        self.enable_keyword_extraction = self.config.get("enable_keyword_extraction", True)
        self.enable_topic_modeling = self.config.get("enable_topic_modeling", False)
        self.enable_text_normalization = self.config.get("enable_text_normalization", True)

        # Quality thresholds
        self.min_language_confidence = self.config.get("min_language_confidence", 0.8)
        self.max_special_char_ratio = self.config.get("max_special_char_ratio", 0.3)

        # Preprocessing settings
        self.normalize_unicode = self.config.get("normalize_unicode", True)
        self.remove_extra_whitespace = self.config.get("remove_extra_whitespace", True)
        self.preserve_formatting = self.config.get("preserve_formatting", True)

        # Stop words for different languages
        self.stop_words = self._load_stop_words()

        # Content categories and patterns
        self.content_patterns = self._load_content_patterns()

        self.logger.info(
            "Text processor initialized",
            max_length=self.max_text_length,
            supported_languages=len(self.supported_languages),
            features_enabled={
                "language_detection": self.enable_language_detection,
                "sentiment_analysis": self.enable_sentiment_analysis,
                "keyword_extraction": self.enable_keyword_extraction,
                "text_normalization": self.enable_text_normalization
            }
        )

    @property
    def supported_message_types(self) -> List[MessageType]:
        return [MessageType.TEXT]

    @property
    def processor_name(self) -> str:
        return "TextProcessor"

    async def process(
            self,
            content: MessageContent,
            context: ProcessingContext
    ) -> ProcessingResult:
        """Process text message content."""
        start_time = datetime.utcnow()

        try:
            # Validate input
            if not await self.validate_input(content, context):
                processing_time = self._measure_processing_time(start_time)
                self.update_metrics(False, processing_time, "text", "VALIDATION_FAILED")

                return self._create_result(
                    content,
                    processing_time,
                    success=False,
                    errors=["Input validation failed"]
                )

            text = content.text
            if not text:
                processing_time = self._measure_processing_time(start_time)
                self.update_metrics(False, processing_time, "text", "NO_TEXT_CONTENT")

                return self._create_result(
                    content,
                    processing_time,
                    success=False,
                    errors=["No text content to process"]
                )

            # Initialize result tracking
            processing_results = {}
            warnings = []

            # Step 1: Text normalization
            if self.enable_text_normalization:
                normalized_text = await self.normalize_text(text)
                processing_results["normalized_text"] = normalized_text
            else:
                normalized_text = text

            # Step 2: Language detection
            detected_language = None
            language_confidence = None
            if self.enable_language_detection:
                detected_language, language_confidence = await self.detect_language(normalized_text)
                processing_results["language_detection"] = {
                    "detected_language": detected_language,
                    "confidence": language_confidence
                }

                if language_confidence and language_confidence < self.min_language_confidence:
                    warnings.append(f"Low language detection confidence: {language_confidence:.2f}")

            # Step 3: Entity extraction
            entities = await self.extract_entities(content, context)

            # Step 4: Keyword extraction
            keywords = []
            if self.enable_keyword_extraction:
                keywords = await self.extract_keywords(normalized_text, detected_language or context.language)
                processing_results["keywords"] = keywords

            # Step 5: Content analysis
            content_analysis = await self.analyze_content(normalized_text, context)
            processing_results["content_analysis"] = content_analysis

            # Step 6: Sentiment analysis
            sentiment = None
            if self.enable_sentiment_analysis:
                sentiment = await self.analyze_sentiment(normalized_text, detected_language or context.language)
                processing_results["sentiment"] = sentiment

            # Step 7: Safety and quality checks
            safety_result = await self.analyze_content_safety(content, context)
            quality_assessment = await self.assess_text_quality(normalized_text, context)

            # Step 8: Topic extraction and categorization
            topics = await self.extract_topics(normalized_text, context)
            categories = await self.categorize_content(normalized_text, entities, context)

            # Create processed content
            processed_content = MessageContent(
                type=content.type,
                text=normalized_text,
                language=detected_language or context.language,
                media=content.media,
                location=content.location,
                quick_replies=content.quick_replies,
                buttons=content.buttons
            )

            # Calculate processing time
            processing_time = self._measure_processing_time(start_time)

            # Update metrics
            self.update_metrics(True, processing_time, "text")

            # Create comprehensive result
            result = ProcessingResult(
                success=True,
                original_content=content,
                processed_content=processed_content,
                detected_language=detected_language,
                language_confidence=language_confidence,

                entities=entities,
                extracted_data={
                    "keywords": keywords,
                    "normalized_text": normalized_text,
                    "content_analysis": content_analysis,
                    "topics": topics,
                    "quality_assessment": quality_assessment,
                    **processing_results
                },

                sentiment=sentiment,
                content_categories=categories,
                content_tags=keywords[:10],  # Limit tags
                content_topics=topics,

                quality_score=quality_assessment.get("overall_score", 0.5),
                safety_flags=safety_result.get("flags", []),
                moderation_required=not safety_result.get("safe", True),

                pii_detected=any("pii" in flag for flag in safety_result.get("flags", [])),
                pii_types=[flag.replace("potential_pii_", "") for flag in safety_result.get("flags", []) if
                           "pii" in flag],

                processing_time_ms=processing_time,
                processor_version=self.processor_version,
                warnings=warnings,

                metadata={
                    "text_stats": {
                        "character_count": len(text),
                        "word_count": len(text.split()),
                        "sentence_count": len(re.split(r'[.!?]+', text)),
                        "paragraph_count": len([p for p in text.split('\n\n') if p.strip()])
                    },
                    "processing_features": {
                        "normalization_applied": self.enable_text_normalization,
                        "language_detected": detected_language is not None,
                        "sentiment_analyzed": sentiment is not None,
                        "keywords_extracted": len(keywords) > 0
                    },
                    "safety_analysis": safety_result
                }
            )

            self.logger.info(
                "Text processing completed",
                text_length=len(text),
                detected_language=detected_language,
                entities_count=len(entities),
                keywords_count=len(keywords),
                categories=categories,
                quality_score=quality_assessment.get("overall_score"),
                safety_flags=len(safety_result.get("flags", [])),
                processing_time_ms=processing_time
            )

            return result

        except Exception as e:
            processing_time = self._measure_processing_time(start_time)
            self.update_metrics(False, processing_time, "text", "PROCESSING_ERROR")

            self.logger.error(
                "Text processing failed",
                error=str(e),
                text_length=len(content.text) if content.text else 0,
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
        """Validate text-specific content."""
        if not content.text:
            return False

        text_length = len(content.text)

        if text_length < self.min_text_length:
            self.logger.warning(
                "Text too short for processing",
                text_length=text_length,
                min_length=self.min_text_length
            )
            return False

        if text_length > self.max_text_length:
            self.logger.warning(
                "Text too long for processing",
                text_length=text_length,
                max_length=self.max_text_length
            )
            return False

        return True

    async def detect_language(self, text: str) -> Tuple[Optional[str], Optional[float]]:
        """Detect language of text."""
        if not LANGDETECT_AVAILABLE:
            self.logger.debug("Language detection library not available")
            return None, None

        try:
            # Clean text for better detection
            clean_text = re.sub(r'[^\w\s]', ' ', text)
            clean_text = re.sub(r'\s+', ' ', clean_text).strip()

            if len(clean_text) < 3:
                return None, None

            detected_lang = detect(clean_text)

            # Simple confidence estimation based on text length and language
            confidence = min(0.95, 0.5 + (len(clean_text) / 200))

            # Adjust confidence based on supported languages
            if detected_lang in self.supported_languages:
                confidence = min(0.95, confidence + 0.1)
            else:
                confidence = max(0.3, confidence - 0.2)
                # Default to English for unsupported languages
                detected_lang = "en"

            return detected_lang, confidence

        except LangDetectError:
            self.logger.debug("Language detection failed - insufficient text")
            return None, None
        except Exception as e:
            self.logger.error("Language detection failed", error=str(e))
            return None, None

    async def normalize_text(self, text: str) -> str:
        """Normalize text formatting."""
        try:
            normalized = text

            # Unicode normalization
            if self.normalize_unicode:
                import unicodedata
                normalized = unicodedata.normalize('NFKC', normalized)

            # Remove excessive whitespace while preserving structure
            if self.remove_extra_whitespace:
                if self.preserve_formatting:
                    # Preserve line breaks but clean up spaces
                    lines = normalized.split('\n')
                    lines = [re.sub(r'[ \t]+', ' ', line).strip() for line in lines]
                    normalized = '\n'.join(lines)
                else:
                    # Aggressive whitespace cleanup
                    normalized = re.sub(r'\s+', ' ', normalized)

            # Remove leading/trailing whitespace
            normalized = normalized.strip()

            return normalized

        except Exception as e:
            self.logger.error("Text normalization failed", error=str(e))
            return text

    async def extract_keywords(self, text: str, language: str = "en") -> List[str]:
        """Extract keywords from text."""
        try:
            # Simple keyword extraction (use more sophisticated NLP in production)
            words = re.findall(r'\b[a-zA-Z]{3,}\b', text.lower())

            # Get stop words for language
            stop_words = self.stop_words.get(language, self.stop_words.get("en", set()))

            # Remove stop words and short words
            keywords = [word for word in words if word not in stop_words and len(word) > 2]

            # Count frequency and return top keywords
            word_counts = Counter(keywords)
            top_keywords = [word for word, count in word_counts.most_common(20)]

            return top_keywords

        except Exception as e:
            self.logger.error("Keyword extraction failed", error=str(e))
            return []

    async def analyze_content(self, text: str, context: ProcessingContext) -> Dict[str, Any]:
        """Analyze content characteristics."""
        try:
            # Basic text statistics
            words = text.split()
            sentences = re.split(r'[.!?]+', text)
            paragraphs = [p for p in text.split('\n\n') if p.strip()]

            analysis = {
                "statistics": {
                    "word_count": len(words),
                    "character_count": len(text),
                    "sentence_count": len([s for s in sentences if s.strip()]),
                    "paragraph_count": len(paragraphs),
                    "avg_word_length": sum(len(word) for word in words) / len(words) if words else 0,
                    "avg_sentence_length": len(words) / len(sentences) if sentences else 0
                },

                "characteristics": {
                    "has_questions": bool(re.search(r'\?', text)),
                    "has_exclamations": bool(re.search(r'!', text)),
                    "has_urls": bool(re.search(r'http[s]?://', text)),
                    "has_emails": bool(re.search(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', text)),
                    "has_phone_numbers": bool(re.search(r'(\+?[\d\s\-\(\)]{10,})', text)),
                    "has_numbers": bool(re.search(r'\d+', text)),
                    "has_caps": bool(re.search(r'[A-Z]{3,}', text))
                },

                "categories": [],
                "quality_indicators": {},
                "readability": "medium"
            }

            # Content categorization based on patterns
            categories = []

            # Business/commerce indicators
            business_keywords = ["order", "purchase", "buy", "sell", "price", "cost", "payment", "invoice", "product",
                                 "service"]
            if any(keyword in text.lower() for keyword in business_keywords):
                categories.append("business")

            # Support/help indicators
            support_keywords = ["help", "support", "problem", "issue", "error", "bug", "question", "assistance"]
            if any(keyword in text.lower() for keyword in support_keywords):
                categories.append("support")

            # Technical indicators
            tech_keywords = ["api", "code", "software", "application", "system", "database", "server", "technical"]
            if any(keyword in text.lower() for keyword in tech_keywords):
                categories.append("technical")

            # Personal/social indicators
            personal_keywords = ["i", "me", "my", "myself", "personal", "family", "friend"]
            if any(keyword in text.lower().split() for keyword in personal_keywords):
                categories.append("personal")

            # Urgent/priority indicators
            urgent_keywords = ["urgent", "emergency", "asap", "immediately", "critical", "important"]
            if any(keyword in text.lower() for keyword in urgent_keywords):
                categories.append("urgent")

            analysis["categories"] = categories

            # Quality scoring
            quality_score = 0.5  # Base score

            # Positive quality indicators
            if len(text) > 10:
                quality_score += 0.1
            if len(words) > 5:
                quality_score += 0.1
            if not re.search(r'[!@#$%^&*()]{3,}', text):  # Not too many special chars
                quality_score += 0.1
            if analysis["statistics"]["avg_word_length"] > 3:
                quality_score += 0.1
            if analysis["characteristics"]["has_questions"] or analysis["characteristics"]["has_exclamations"]:
                quality_score += 0.05

            # Negative quality indicators
            if len(re.findall(r'[A-Z]{5,}', text)) > 2:  # Too much caps
                quality_score -= 0.2
            if text.count('!') > 5:  # Too many exclamations
                quality_score -= 0.1
            if len(text) < 5:  # Too short
                quality_score -= 0.3

            analysis["overall_score"] = max(0.0, min(1.0, quality_score))

            # Readability assessment (simplified)
            avg_sentence_length = analysis["statistics"]["avg_sentence_length"]
            if avg_sentence_length < 10:
                analysis["readability"] = "easy"
            elif avg_sentence_length < 20:
                analysis["readability"] = "medium"
            else:
                analysis["readability"] = "difficult"

            return analysis

        except Exception as e:
            self.logger.error("Content analysis failed", error=str(e))
            return {"categories": [], "overall_score": 0.5, "error": str(e)}

    async def analyze_sentiment(self, text: str, language: str = "en") -> Optional[Dict[str, Any]]:
        """Analyze sentiment of text (basic implementation)."""
        try:
            # Simple rule-based sentiment analysis
            # In production, use proper sentiment analysis models

            positive_words = [
                "good", "great", "excellent", "amazing", "wonderful", "fantastic", "love", "like",
                "happy", "pleased", "satisfied", "thank", "thanks", "awesome", "perfect", "best"
            ]

            negative_words = [
                "bad", "terrible", "awful", "hate", "dislike", "angry", "frustrated", "annoyed",
                "disappointed", "unhappy", "worst", "horrible", "disgusting", "stupid", "useless"
            ]

            words = re.findall(r'\b\w+\b', text.lower())

            positive_count = sum(1 for word in words if word in positive_words)
            negative_count = sum(1 for word in words if word in negative_words)
            total_words = len(words)

            if total_words == 0:
                return None

            # Calculate sentiment score (-1 to 1)
            sentiment_score = (positive_count - negative_count) / total_words
            sentiment_score = max(-1.0, min(1.0, sentiment_score))

            # Determine sentiment label
            if sentiment_score > 0.1:
                label = "positive"
            elif sentiment_score < -0.1:
                label = "negative"
            else:
                label = "neutral"

            # Calculate confidence based on the strength of indicators
            confidence = min(0.9, abs(sentiment_score) * 2 + 0.1)

            return {
                "label": label,
                "score": sentiment_score,
                "confidence": confidence,
                "positive_indicators": positive_count,
                "negative_indicators": negative_count,
                "total_words": total_words
            }

        except Exception as e:
            self.logger.error("Sentiment analysis failed", error=str(e))
            return None

    async def extract_topics(self, text: str, context: ProcessingContext) -> List[str]:
        """Extract topics from text."""
        try:
            # Simple topic extraction based on keyword clustering
            keywords = await self.extract_keywords(text, context.language)

            # Group keywords by semantic similarity (simplified)
            topics = []

            # Predefined topic categories
            topic_keywords = {
                "technology": ["software", "computer", "internet", "digital", "tech", "system", "application"],
                "business": ["money", "sales", "market", "business", "company", "revenue", "profit"],
                "health": ["health", "medical", "doctor", "hospital", "medicine", "treatment", "care"],
                "education": ["school", "education", "learning", "student", "teacher", "course", "study"],
                "travel": ["travel", "trip", "vacation", "hotel", "flight", "destination", "tourism"],
                "food": ["food", "restaurant", "cooking", "recipe", "meal", "dining", "cuisine"],
                "sports": ["sports", "game", "team", "player", "match", "competition", "athletic"],
                "entertainment": ["movie", "music", "entertainment", "show", "concert", "performance"]
            }

            # Check which topics are present based on keywords
            for topic, topic_words in topic_keywords.items():
                if any(keyword in topic_words for keyword in keywords):
                    topics.append(topic)

            return topics[:5]  # Limit to top 5 topics

        except Exception as e:
            self.logger.error("Topic extraction failed", error=str(e))
            return []

    async def categorize_content(
            self,
            text: str,
            entities: Dict[str, Any],
            context: ProcessingContext
    ) -> List[str]:
        """Categorize content based on text analysis and entities."""
        try:
            categories = []
            text_lower = text.lower()

            # Use content patterns for categorization
            for category, patterns in self.content_patterns.items():
                if any(pattern in text_lower for pattern in patterns):
                    categories.append(category)

            # Entity-based categorization
            if entities.get("emails"):
                categories.append("contact_information")

            if entities.get("phones"):
                categories.append("contact_information")

            if entities.get("urls"):
                categories.append("web_content")

            if entities.get("money"):
                categories.append("financial")

            if entities.get("dates"):
                categories.append("scheduling")

            # Context-based categorization
            if context.channel == "whatsapp":
                categories.append("messaging")
            elif context.channel == "web":
                categories.append("web_chat")

            return list(set(categories))  # Remove duplicates

        except Exception as e:
            self.logger.error("Content categorization failed", error=str(e))
            return []

    async def assess_text_quality(self, text: str, context: ProcessingContext) -> Dict[str, Any]:
        """Assess the quality of text content."""
        try:
            assessment = {
                "overall_score": 0.5,
                "dimensions": {},
                "issues": [],
                "recommendations": []
            }

            # Grammar and spelling assessment (basic)
            grammar_score = await self._assess_grammar(text)
            assessment["dimensions"]["grammar"] = grammar_score

            # Clarity assessment
            clarity_score = await self._assess_clarity(text)
            assessment["dimensions"]["clarity"] = clarity_score

            # Completeness assessment
            completeness_score = await self._assess_completeness(text, context)
            assessment["dimensions"]["completeness"] = completeness_score

            # Coherence assessment
            coherence_score = await self._assess_coherence(text)
            assessment["dimensions"]["coherence"] = coherence_score

            # Calculate overall score
            scores = [grammar_score, clarity_score, completeness_score, coherence_score]
            assessment["overall_score"] = sum(scores) / len(scores)

            # Generate recommendations based on scores
            if grammar_score < 0.6:
                assessment["issues"].append("grammar_issues")
                assessment["recommendations"].append("Check spelling and grammar")

            if clarity_score < 0.6:
                assessment["issues"].append("clarity_issues")
                assessment["recommendations"].append("Use clearer language and structure")

            if completeness_score < 0.6:
                assessment["issues"].append("incomplete_information")
                assessment["recommendations"].append("Provide more complete information")

            return assessment

        except Exception as e:
            self.logger.error("Text quality assessment failed", error=str(e))
            return {"overall_score": 0.5, "error": str(e)}

    async def _assess_grammar(self, text: str) -> float:
        """Assess grammar quality (basic implementation)."""
        # Simple grammar checks
        score = 1.0

        # Check for basic punctuation
        if not re.search(r'[.!?]$', text.strip()):
            score -= 0.1

        # Check for proper capitalization
        sentences = re.split(r'[.!?]+', text)
        for sentence in sentences:
            sentence = sentence.strip()
            if sentence and not sentence[0].isupper():
                score -= 0.1
                break

        # Check for excessive punctuation
        if len(re.findall(r'[!?]{2,}', text)) > 0:
            score -= 0.2

        return max(0.0, score)

    async def _assess_clarity(self, text: str) -> float:
        """Assess text clarity."""
        # Simple clarity metrics
        words = text.split()
        sentences = re.split(r'[.!?]+', text)

        if not words or not sentences:
            return 0.0

        # Average sentence length (shorter is generally clearer)
        avg_sentence_length = len(words) / len([s for s in sentences if s.strip()])

        # Clarity score based on sentence length
        if avg_sentence_length <= 15:
            clarity_score = 1.0
        elif avg_sentence_length <= 25:
            clarity_score = 0.8
        else:
            clarity_score = 0.6

        # Adjust for readability indicators
        if re.search(r'\d+', text):  # Contains numbers (might be clearer)
            clarity_score += 0.1

        if len(re.findall(r'[A-Z]{3,}', text)) > 2:  # Too many caps
            clarity_score -= 0.2

        return max(0.0, min(1.0, clarity_score))

    async def _assess_completeness(self, text: str, context: ProcessingContext) -> float:
        """Assess information completeness."""
        # Basic completeness checks
        score = 0.5

        # Length-based assessment
        if len(text) > 50:
            score += 0.2

        if len(text.split()) > 10:
            score += 0.2

        # Content completeness indicators
        if re.search(r'\?', text):  # Contains questions (might indicate incomplete info)
            score -= 0.1

        if re.search(r'\b(what|when|where|why|how)\b', text.lower()):
            score += 0.1  # Contains question words (provides context)

        return max(0.0, min(1.0, score))

    async def _assess_coherence(self, text: str) -> float:
        """Assess text coherence."""
        # Simple coherence assessment
        sentences = [s.strip() for s in re.split(r'[.!?]+', text) if s.strip()]

        if len(sentences) <= 1:
            return 0.8  # Single sentence is coherent by default

        # Check for coherence indicators
        coherence_score = 0.7

        # Check for transition words
        transition_words = ["however", "therefore", "moreover", "furthermore", "also", "additionally", "but", "and",
                            "so"]
        if any(word in text.lower() for word in transition_words):
            coherence_score += 0.2

        # Check for pronoun references (basic indicator of coherence)
        pronouns = ["it", "this", "that", "they", "them", "these", "those"]
        if any(pronoun in text.lower().split() for pronoun in pronouns):
            coherence_score += 0.1

        return max(0.0, min(1.0, coherence_score))

    def _load_stop_words(self) -> Dict[str, set]:
        """Load stop words for different languages."""
        # Basic stop words - in production, load from comprehensive lists
        stop_words = {
            "en": {
                "the", "and", "or", "but", "in", "on", "at", "to", "for", "of", "with",
                "by", "from", "up", "about", "into", "through", "during", "before",
                "after", "above", "below", "out", "off", "down", "under", "again",
                "further", "then", "once", "here", "there", "when", "where", "why",
                "how", "all", "any", "both", "each", "few", "more", "most", "other",
                "some", "such", "no", "nor", "not", "only", "own", "same", "so",
                "than", "too", "very", "can", "will", "just", "should", "could",
                "would", "may", "might", "must", "shall", "ought", "need", "dare",
                "used", "able", "like", "well", "also", "back", "even", "still",
                "way", "take", "come", "good", "new", "first", "last", "long",
                "great", "little", "own", "other", "old", "right", "big", "high",
                "different", "small", "large", "next", "early", "young", "important",
                "few", "public", "bad", "same", "able", "is", "are", "was", "were",
                "be", "been", "being", "have", "has", "had", "do", "does", "did",
                "a", "an", "as", "i", "you", "he", "she", "we", "they", "me", "him",
                "her", "us", "them", "my", "your", "his", "its", "our", "their"
            },
            "es": {
                "el", "la", "de", "que", "y", "a", "en", "un", "ser", "se", "no",
                "te", "lo", "le", "da", "su", "por", "son", "con", "para", "es",
                "al", "una", "del", "los", "las", "me", "mi", "tu", "te", "yo",
                "él", "ella", "esto", "eso", "todo", "muy", "más", "pero", "como",
                "sin", "hasta", "desde", "cuando", "donde", "porque", "si", "bien"
            },
            "fr": {
                "le", "de", "et", "à", "un", "il", "être", "et", "en", "avoir",
                "que", "pour", "dans", "ce", "son", "une", "sur", "avec", "ne",
                "se", "pas", "tout", "plus", "par", "grand", "en", "me", "bien",
                "où", "ou", "si", "mais", "non", "des", "ces", "nos", "vos", "aux"
            }
        }

        return stop_words

    def _load_content_patterns(self) -> Dict[str, List[str]]:
        """Load content categorization patterns."""
        return {
            "greeting": ["hello", "hi", "hey", "good morning", "good afternoon", "good evening"],
            "farewell": ["goodbye", "bye", "see you", "talk to you later", "farewell"],
            "question": ["what", "when", "where", "why", "how", "can you", "could you"],
            "complaint": ["problem", "issue", "complaint", "not working", "broken", "error"],
            "compliment": ["great", "awesome", "excellent", "amazing", "wonderful", "perfect"],
            "request": ["please", "can you", "would you", "help me", "i need", "request"],
            "information": ["information", "details", "tell me", "explain", "describe"],
            "order": ["order", "purchase", "buy", "checkout", "payment", "invoice"],
            "support": ["help", "support", "assistance", "guide", "tutorial", "manual"],
            "technical": ["code", "api", "software", "system", "technical", "error", "bug"],
            "billing": ["bill", "billing", "payment", "charge", "cost", "price", "invoice"],
            "account": ["account", "profile", "settings", "login", "password", "user"]
        }