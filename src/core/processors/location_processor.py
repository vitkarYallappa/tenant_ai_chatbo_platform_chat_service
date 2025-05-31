"""
Specialized processor for location message content.

This module handles location-specific processing including coordinate validation,
geocoding, reverse geocoding, and location-based analysis.
"""

import re
import math
from datetime import datetime
from typing import List, Dict, Any, Optional, Tuple

from src.core.processors.base_processor import (
    BaseProcessor,
    ProcessingContext,
    ProcessingResult
)
from src.models.types import MessageContent, MessageType
from src.core.exceptions import ProcessingError, ProcessingValidationError


class LocationProcessor(BaseProcessor):
    """Processor for location message content."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)

        # Location processing configuration
        self.enable_geocoding = self.config.get("enable_geocoding", False)
        self.enable_reverse_geocoding = self.config.get("enable_reverse_geocoding", True)
        self.enable_nearby_search = self.config.get("enable_nearby_search", False)
        self.enable_location_validation = self.config.get("enable_location_validation", True)

        # Coordinate validation bounds
        self.min_latitude = self.config.get("min_latitude", -90.0)
        self.max_latitude = self.config.get("max_latitude", 90.0)
        self.min_longitude = self.config.get("min_longitude", -180.0)
        self.max_longitude = self.config.get("max_longitude", 180.0)

        # Accuracy thresholds (in meters)
        self.high_accuracy_threshold = self.config.get("high_accuracy_threshold", 10)
        self.medium_accuracy_threshold = self.config.get("medium_accuracy_threshold", 100)
        self.low_accuracy_threshold = self.config.get("low_accuracy_threshold", 1000)

        # Privacy and security settings
        self.enable_location_anonymization = self.config.get("enable_location_anonymization", False)
        self.anonymization_radius_meters = self.config.get("anonymization_radius_meters", 1000)
        self.store_precise_coordinates = self.config.get("store_precise_coordinates", True)

        # External service configuration
        self.geocoding_service_url = self.config.get("geocoding_service_url", None)
        self.geocoding_api_key = self.config.get("geocoding_api_key", None)

        # Place categories for classification
        self.place_categories = self._load_place_categories()

        # Coordinate formats for parsing
        self.coordinate_patterns = self._load_coordinate_patterns()

        self.logger.info(
            "Location processor initialized",
            features_enabled={
                "geocoding": self.enable_geocoding,
                "reverse_geocoding": self.enable_reverse_geocoding,
                "nearby_search": self.enable_nearby_search,
                "location_validation": self.enable_location_validation,
                "anonymization": self.enable_location_anonymization
            },
            accuracy_thresholds={
                "high": self.high_accuracy_threshold,
                "medium": self.medium_accuracy_threshold,
                "low": self.low_accuracy_threshold
            }
        )

    @property
    def supported_message_types(self) -> List[MessageType]:
        return [MessageType.LOCATION]

    @property
    def processor_name(self) -> str:
        return "LocationProcessor"

    async def process(
            self,
            content: MessageContent,
            context: ProcessingContext
    ) -> ProcessingResult:
        """Process location message content."""
        start_time = datetime.utcnow()

        try:
            # Validate input
            if not await self.validate_input(content, context):
                processing_time = self._measure_processing_time(start_time)
                self.update_metrics(False, processing_time, "location", "VALIDATION_FAILED")

                return self._create_result(
                    content,
                    processing_time,
                    success=False,
                    errors=["Input validation failed"]
                )

            location = content.location
            if not location:
                processing_time = self._measure_processing_time(start_time)
                self.update_metrics(False, processing_time, "location", "NO_LOCATION_CONTENT")

                return self._create_result(
                    content,
                    processing_time,
                    success=False,
                    errors=["No location content to process"]
                )

            # Initialize result tracking
            processing_results = {}
            warnings = []

            # Step 1: Validate and normalize coordinates
            coordinate_validation = await self.validate_coordinates(
                location.latitude,
                location.longitude
            )
            processing_results["coordinate_validation"] = coordinate_validation

            if not coordinate_validation["valid"]:
                warnings.extend(coordinate_validation.get("warnings", []))

            # Step 2: Assess location accuracy
            accuracy_assessment = await self.assess_location_accuracy(location, context)
            processing_results["accuracy_assessment"] = accuracy_assessment

            # Step 3: Reverse geocoding (if enabled and coordinates are valid)
            reverse_geocoding_result = {}
            if self.enable_reverse_geocoding and coordinate_validation["valid"]:
                reverse_geocoding_result = await self.reverse_geocode(
                    location.latitude,
                    location.longitude
                )
                processing_results["reverse_geocoding"] = reverse_geocoding_result

            # Step 4: Location classification and categorization
            location_classification = await self.classify_location(
                location,
                reverse_geocoding_result,
                context
            )
            processing_results["classification"] = location_classification

            # Step 5: Privacy and anonymization (if enabled)
            anonymized_location = location
            if self.enable_location_anonymization:
                anonymized_location = await self.anonymize_location(location, context)
                processing_results["anonymization"] = {
                    "applied": True,
                    "radius_meters": self.anonymization_radius_meters
                }

            # Step 6: Extract location-based entities
            entities = await self.extract_entities(content, context)

            # Step 7: Nearby points of interest (if enabled)
            nearby_places = {}
            if self.enable_nearby_search and coordinate_validation["valid"]:
                nearby_places = await self.find_nearby_places(
                    location.latitude,
                    location.longitude,
                    context
                )
                processing_results["nearby_places"] = nearby_places

            # Step 8: Location quality and safety assessment
            safety_assessment = await self.analyze_content_safety(content, context)
            quality_assessment = await self.assess_location_quality(location, context)

            # Create processed content
            processed_content = MessageContent(
                type=content.type,
                text=content.text,
                location=anonymized_location,
                media=content.media,
                quick_replies=content.quick_replies,
                buttons=content.buttons
            )

            # Determine content categories
            categories = await self.categorize_location_content(
                location,
                location_classification,
                context
            )

            # Generate location tags
            tags = self._generate_location_tags(
                location,
                reverse_geocoding_result,
                location_classification
            )

            # Calculate processing time
            processing_time = self._measure_processing_time(start_time)

            # Update metrics
            self.update_metrics(True, processing_time, "location")

            # Create comprehensive result
            result = ProcessingResult(
                success=True,
                original_content=content,
                processed_content=processed_content,

                entities=entities,
                extracted_data={
                    "coordinates": {
                        "latitude": location.latitude,
                        "longitude": location.longitude,
                        "accuracy_meters": getattr(location, 'accuracy_meters', None)
                    },
                    "address_components": reverse_geocoding_result.get("address_components", {}),
                    "place_info": reverse_geocoding_result.get("place_info", {}),
                    "nearby_places": nearby_places,
                    "quality_assessment": quality_assessment,
                    **processing_results
                },

                content_categories=categories,
                content_tags=tags,
                content_topics=location_classification.get("topics", []),

                quality_score=quality_assessment.get("overall_score", 0.5),
                safety_flags=safety_assessment.get("flags", []),
                moderation_required=not safety_assessment.get("safe", True),

                processing_time_ms=processing_time,
                processor_version=self.processor_version,
                warnings=warnings,

                metadata={
                    "location_stats": {
                        "coordinate_precision": coordinate_validation.get("precision", "unknown"),
                        "accuracy_level": accuracy_assessment.get("level", "unknown"),
                        "geocoding_confidence": reverse_geocoding_result.get("confidence", 0.0)
                    },
                    "processing_features": {
                        "coordinates_validated": coordinate_validation["valid"],
                        "reverse_geocoded": bool(reverse_geocoding_result),
                        "location_classified": bool(location_classification),
                        "privacy_applied": self.enable_location_anonymization
                    },
                    "privacy_info": {
                        "coordinates_anonymized": self.enable_location_anonymization,
                        "precise_location_stored": self.store_precise_coordinates
                    }
                }
            )

            self.logger.info(
                "Location processing completed",
                latitude=location.latitude,
                longitude=location.longitude,
                accuracy=getattr(location, 'accuracy_meters', None),
                address=getattr(location, 'address', None),
                categories=categories,
                quality_score=quality_assessment.get("overall_score"),
                processing_time_ms=processing_time
            )

            return result

        except Exception as e:
            processing_time = self._measure_processing_time(start_time)
            self.update_metrics(False, processing_time, "location", "PROCESSING_ERROR")

            self.logger.error(
                "Location processing failed",
                error=str(e),
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
        """Validate location-specific content."""
        if not content.location:
            return False

        location = content.location

        # Validate coordinate ranges
        if not (-90 <= location.latitude <= 90):
            self.logger.warning(
                "Invalid latitude",
                latitude=location.latitude
            )
            return False

        if not (-180 <= location.longitude <= 180):
            self.logger.warning(
                "Invalid longitude",
                longitude=location.longitude
            )
            return False

        return True

    async def validate_coordinates(
            self,
            latitude: float,
            longitude: float
    ) -> Dict[str, Any]:
        """Validate and assess coordinate quality."""
        validation_result = {
            "valid": False,
            "precision": "unknown",
            "warnings": [],
            "issues": []
        }

        try:
            # Basic range validation
            if not (-90 <= latitude <= 90):
                validation_result["issues"].append("latitude_out_of_range")
                validation_result["warnings"].append(f"Latitude {latitude} is outside valid range [-90, 90]")
                return validation_result

            if not (-180 <= longitude <= 180):
                validation_result["issues"].append("longitude_out_of_range")
                validation_result["warnings"].append(f"Longitude {longitude} is outside valid range [-180, 180]")
                return validation_result

            # Check for null island (0, 0)
            if abs(latitude) < 0.001 and abs(longitude) < 0.001:
                validation_result["issues"].append("null_island_coordinates")
                validation_result["warnings"].append("Coordinates are very close to (0, 0) - may be invalid")

            # Assess coordinate precision
            lat_precision = self._assess_coordinate_precision(latitude)
            lon_precision = self._assess_coordinate_precision(longitude)

            if lat_precision >= 5 and lon_precision >= 5:
                validation_result["precision"] = "high"  # ~1 meter precision
            elif lat_precision >= 4 and lon_precision >= 4:
                validation_result["precision"] = "medium"  # ~10 meter precision
            elif lat_precision >= 3 and lon_precision >= 3:
                validation_result["precision"] = "low"  # ~100 meter precision
            else:
                validation_result["precision"] = "very_low"  # >1km precision
                validation_result["warnings"].append("Very low coordinate precision detected")

            # Check for common invalid coordinates
            if self._is_obviously_invalid_location(latitude, longitude):
                validation_result["issues"].append("obviously_invalid")
                validation_result["warnings"].append("Coordinates appear to be invalid or test data")

            validation_result["valid"] = len(validation_result["issues"]) == 0

            return validation_result

        except Exception as e:
            self.logger.error("Coordinate validation failed", error=str(e))
            validation_result["issues"].append("validation_error")
            validation_result["warnings"].append(f"Validation error: {str(e)}")
            return validation_result

    def _assess_coordinate_precision(self, coordinate: float) -> int:
        """Assess the decimal precision of a coordinate."""
        coord_str = f"{coordinate:.10f}".rstrip('0')
        if '.' in coord_str:
            return len(coord_str.split('.')[1])
        return 0

    def _is_obviously_invalid_location(self, latitude: float, longitude: float) -> bool:
        """Check for obviously invalid or test coordinates."""
        # Common test coordinates
        test_coordinates = [
            (37.7749, -122.4194),  # San Francisco (very common test location)
            (40.7128, -74.0060),  # New York City
            (51.5074, -0.1278),  # London
        ]

        # Check if coordinates match common test locations exactly
        for test_lat, test_lon in test_coordinates:
            if abs(latitude - test_lat) < 0.0001 and abs(longitude - test_lon) < 0.0001:
                return True

        # Check for coordinates that are too perfect (e.g., exact integers)
        if latitude == int(latitude) and longitude == int(longitude):
            return True

        return False

    async def assess_location_accuracy(
            self,
            location: Any,
            context: ProcessingContext
    ) -> Dict[str, Any]:
        """Assess the accuracy of location data."""
        assessment = {
            "level": "unknown",
            "score": 0.5,
            "factors": {},
            "recommendations": []
        }

        try:
            accuracy_meters = getattr(location, 'accuracy_meters', None)

            if accuracy_meters is not None:
                # Assess based on reported accuracy
                if accuracy_meters <= self.high_accuracy_threshold:
                    assessment["level"] = "high"
                    assessment["score"] = 0.9
                elif accuracy_meters <= self.medium_accuracy_threshold:
                    assessment["level"] = "medium"
                    assessment["score"] = 0.7
                elif accuracy_meters <= self.low_accuracy_threshold:
                    assessment["level"] = "low"
                    assessment["score"] = 0.5
                else:
                    assessment["level"] = "very_low"
                    assessment["score"] = 0.3

                assessment["factors"]["reported_accuracy"] = accuracy_meters
            else:
                # Assess based on other factors
                assessment["level"] = "medium"
                assessment["score"] = 0.6
                assessment["factors"]["reported_accuracy"] = None
                assessment["recommendations"].append("No accuracy information provided")

            # Factor in coordinate precision
            lat_precision = self._assess_coordinate_precision(location.latitude)
            lon_precision = self._assess_coordinate_precision(location.longitude)
            min_precision = min(lat_precision, lon_precision)

            if min_precision >= 5:
                precision_score = 1.0
            elif min_precision >= 4:
                precision_score = 0.8
            elif min_precision >= 3:
                precision_score = 0.6
            else:
                precision_score = 0.4
                assessment["recommendations"].append("Low coordinate precision detected")

            assessment["factors"]["coordinate_precision"] = {
                "latitude_decimals": lat_precision,
                "longitude_decimals": lon_precision,
                "score": precision_score
            }

            # Adjust overall score based on precision
            assessment["score"] = (assessment["score"] + precision_score) / 2

            # Factor in channel reliability
            channel_reliability = self._get_channel_accuracy_factor(context.channel)
            assessment["factors"]["channel_reliability"] = channel_reliability
            assessment["score"] = assessment["score"] * channel_reliability

            # Final score bounds
            assessment["score"] = max(0.0, min(1.0, assessment["score"]))

            return assessment

        except Exception as e:
            self.logger.error("Location accuracy assessment failed", error=str(e))
            return assessment

    def _get_channel_accuracy_factor(self, channel: str) -> float:
        """Get accuracy factor based on channel type."""
        # Different channels have different typical accuracy
        channel_factors = {
            "whatsapp": 0.9,  # Mobile GPS typically good
            "messenger": 0.9,
            "web": 0.7,  # Browser location can be less accurate
            "slack": 0.6,  # Often desktop, may use IP location
            "teams": 0.6,
            "sms": 0.8,
            "voice": 0.8
        }

        return channel_factors.get(channel.lower(), 0.7)

    async def reverse_geocode(
            self,
            latitude: float,
            longitude: float
    ) -> Dict[str, Any]:
        """Perform reverse geocoding to get address from coordinates."""
        if not self.enable_reverse_geocoding:
            return {}

        try:
            # In production, integrate with actual geocoding service
            # For now, provide a structured placeholder response

            geocoding_result = {
                "success": False,
                "confidence": 0.0,
                "address_components": {},
                "formatted_address": "",
                "place_info": {},
                "service_used": "placeholder"
            }

            # Simulate geocoding based on coordinate ranges (very basic)
            geocoding_result.update(await self._simulate_reverse_geocoding(latitude, longitude))

            return geocoding_result

        except Exception as e:
            self.logger.error(
                "Reverse geocoding failed",
                error=str(e),
                latitude=latitude,
                longitude=longitude
            )
            return {"success": False, "error": str(e)}

    async def _simulate_reverse_geocoding(
            self,
            latitude: float,
            longitude: float
    ) -> Dict[str, Any]:
        """Simulate reverse geocoding for demonstration purposes."""
        # Very basic simulation based on coordinate ranges
        result = {
            "success": True,
            "confidence": 0.6,
            "address_components": {},
            "formatted_address": "",
            "place_info": {}
        }

        # Rough country detection based on coordinate ranges
        if 25 <= latitude <= 49 and -125 <= longitude <= -66:
            # Roughly USA
            result["address_components"] = {
                "country": "United States",
                "country_code": "US",
                "administrative_area_level_1": "Unknown State",
                "locality": "Unknown City"
            }
            result["formatted_address"] = f"Approximate location in United States"
        elif 49 <= latitude <= 83 and -141 <= longitude <= -52:
            # Roughly Canada
            result["address_components"] = {
                "country": "Canada",
                "country_code": "CA",
                "administrative_area_level_1": "Unknown Province",
                "locality": "Unknown City"
            }
            result["formatted_address"] = f"Approximate location in Canada"
        elif 35 <= latitude <= 71 and -10 <= longitude <= 40:
            # Roughly Europe
            result["address_components"] = {
                "country": "Unknown European Country",
                "country_code": "EU",
                "administrative_area_level_1": "Unknown Region",
                "locality": "Unknown City"
            }
            result["formatted_address"] = f"Approximate location in Europe"
        else:
            result["address_components"] = {
                "country": "Unknown",
                "country_code": "XX",
                "administrative_area_level_1": "Unknown",
                "locality": "Unknown"
            }
            result["formatted_address"] = f"Location at {latitude:.4f}, {longitude:.4f}"

        # Add place type estimation
        result["place_info"] = {
            "place_type": "unknown",
            "business_status": "unknown",
            "types": ["geographic_location"]
        }

        return result

    async def classify_location(
            self,
            location: Any,
            geocoding_result: Dict[str, Any],
            context: ProcessingContext
    ) -> Dict[str, Any]:
        """Classify location based on coordinates and geocoding results."""
        classification = {
            "primary_type": "geographic_location",
            "secondary_types": [],
            "confidence": 0.5,
            "topics": [],
            "business_context": "unknown"
        }

        try:
            # Use geocoding results if available
            if geocoding_result.get("place_info"):
                place_info = geocoding_result["place_info"]
                classification["primary_type"] = place_info.get("place_type", "geographic_location")
                classification["secondary_types"] = place_info.get("types", [])

            # Classify based on context
            if context.conversation_context:
                # Look for business-related keywords in conversation
                business_keywords = ["meeting", "office", "store", "restaurant", "hotel", "appointment"]
                conversation_text = str(context.conversation_context).lower()

                if any(keyword in conversation_text for keyword in business_keywords):
                    classification["business_context"] = "business_related"
                    classification["topics"].append("business")
                else:
                    classification["business_context"] = "personal"
                    classification["topics"].append("personal")

            # Classify based on accuracy and precision
            if hasattr(location, 'accuracy_meters'):
                if location.accuracy_meters <= 10:
                    classification["topics"].append("precise_location")
                elif location.accuracy_meters > 1000:
                    classification["topics"].append("approximate_location")

            # Add geographic topics based on coordinates
            geographic_topics = self._get_geographic_topics(location.latitude, location.longitude)
            classification["topics"].extend(geographic_topics)

            return classification

        except Exception as e:
            self.logger.error("Location classification failed", error=str(e))
            return classification

    def _get_geographic_topics(self, latitude: float, longitude: float) -> List[str]:
        """Get geographic topics based on coordinates."""
        topics = []

        # Hemisphere classification
        if latitude >= 0:
            topics.append("northern_hemisphere")
        else:
            topics.append("southern_hemisphere")

        if longitude >= 0:
            topics.append("eastern_hemisphere")
        else:
            topics.append("western_hemisphere")

        # Climate zone (very rough)
        if abs(latitude) <= 23.5:
            topics.append("tropical_zone")
        elif abs(latitude) <= 66.5:
            topics.append("temperate_zone")
        else:
            topics.append("polar_zone")

        return topics

    async def anonymize_location(
            self,
            location: Any,
            context: ProcessingContext
    ) -> Any:
        """Anonymize location data for privacy protection."""
        if not self.enable_location_anonymization:
            return location

        try:
            # Add random noise to coordinates within specified radius
            anonymized_lat, anonymized_lon = self._add_coordinate_noise(
                location.latitude,
                location.longitude,
                self.anonymization_radius_meters
            )

            # Create anonymized location object
            anonymized_location = type(location)(
                latitude=anonymized_lat,
                longitude=anonymized_lon,
                accuracy_meters=max(
                    getattr(location, 'accuracy_meters', self.anonymization_radius_meters),
                    self.anonymization_radius_meters
                ),
                address=None  # Remove precise address for privacy
            )

            self.logger.debug(
                "Location anonymized",
                original_lat=location.latitude,
                original_lon=location.longitude,
                anonymized_lat=anonymized_lat,
                anonymized_lon=anonymized_lon,
                radius=self.anonymization_radius_meters
            )

            return anonymized_location

        except Exception as e:
            self.logger.error("Location anonymization failed", error=str(e))
            return location

    def _add_coordinate_noise(
            self,
            latitude: float,
            longitude: float,
            radius_meters: float
    ) -> Tuple[float, float]:
        """Add random noise to coordinates within specified radius."""
        import random

        # Convert radius to degrees (rough approximation)
        # 1 degree ≈ 111,320 meters at equator
        radius_degrees = radius_meters / 111320.0

        # Generate random offset within circle
        angle = random.random() * 2 * math.pi
        distance = random.random() * radius_degrees

        # Apply offset
        lat_offset = distance * math.cos(angle)
        lon_offset = distance * math.sin(angle) / math.cos(math.radians(latitude))

        new_latitude = latitude + lat_offset
        new_longitude = longitude + lon_offset

        # Ensure coordinates remain within valid bounds
        new_latitude = max(-90, min(90, new_latitude))
        new_longitude = max(-180, min(180, new_longitude))

        return new_latitude, new_longitude

    async def find_nearby_places(
            self,
            latitude: float,
            longitude: float,
            context: ProcessingContext
    ) -> Dict[str, Any]:
        """Find nearby places of interest."""
        if not self.enable_nearby_search:
            return {}

        try:
            # In production, integrate with places API
            # For now, return placeholder structure

            nearby_result = {
                "search_performed": False,
                "places_found": 0,
                "places": [],
                "search_radius_meters": 1000,
                "reason": "Nearby search not implemented"
            }

            return nearby_result

        except Exception as e:
            self.logger.error("Nearby places search failed", error=str(e))
            return {"search_performed": False, "error": str(e)}

    async def assess_location_quality(
            self,
            location: Any,
            context: ProcessingContext
    ) -> Dict[str, Any]:
        """Assess the quality of location data."""
        try:
            assessment = {
                "overall_score": 0.5,
                "dimensions": {},
                "issues": [],
                "recommendations": []
            }

            # Coordinate validity
            coord_score = 1.0 if self._are_coordinates_reasonable(location.latitude, location.longitude) else 0.0
            assessment["dimensions"]["coordinate_validity"] = coord_score

            # Accuracy assessment
            accuracy_score = await self._assess_accuracy_quality(location)
            assessment["dimensions"]["accuracy"] = accuracy_score

            # Completeness assessment
            completeness_score = await self._assess_location_completeness(location)
            assessment["dimensions"]["completeness"] = completeness_score

            # Precision assessment
            precision_score = await self._assess_location_precision(location)
            assessment["dimensions"]["precision"] = precision_score

            # Calculate overall score
            scores = [coord_score, accuracy_score, completeness_score, precision_score]
            assessment["overall_score"] = sum(scores) / len(scores)

            # Generate recommendations
            if coord_score < 1.0:
                assessment["issues"].append("invalid_coordinates")
                assessment["recommendations"].append("Verify coordinate validity")

            if accuracy_score < 0.6:
                assessment["issues"].append("low_accuracy")
                assessment["recommendations"].append("Improve location accuracy")

            if completeness_score < 0.6:
                assessment["issues"].append("incomplete_data")
                assessment["recommendations"].append("Provide more location details")

            return assessment

        except Exception as e:
            self.logger.error("Location quality assessment failed", error=str(e))
            return {"overall_score": 0.5, "error": str(e)}

    def _are_coordinates_reasonable(self, latitude: float, longitude: float) -> bool:
        """Check if coordinates are reasonable (not obviously invalid)."""
        # Check basic bounds
        if not (-90 <= latitude <= 90 and -180 <= longitude <= 180):
            return False

        # Check for null island
        if abs(latitude) < 0.001 and abs(longitude) < 0.001:
            return False

        # Check for unrealistic precision
        if self._assess_coordinate_precision(latitude) > 8 or self._assess_coordinate_precision(longitude) > 8:
            return False

        return True

    async def _assess_accuracy_quality(self, location: Any) -> float:
        """Assess accuracy quality score."""
        if not hasattr(location, 'accuracy_meters') or location.accuracy_meters is None:
            return 0.5  # Neutral score for unknown accuracy

        accuracy = location.accuracy_meters

        if accuracy <= 5:
            return 1.0
        elif accuracy <= 10:
            return 0.9
        elif accuracy <= 50:
            return 0.7
        elif accuracy <= 100:
            return 0.6
        elif accuracy <= 500:
            return 0.4
        else:
            return 0.2

    async def _assess_location_completeness(self, location: Any) -> float:
        """Assess location data completeness."""
        score = 0.5  # Base score for coordinates

        # Add points for additional data
        if hasattr(location, 'accuracy_meters') and location.accuracy_meters is not None:
            score += 0.2

        if hasattr(location, 'address') and location.address:
            score += 0.2

        if hasattr(location, 'place_name') and hasattr(location, 'place_name'):
            score += 0.1

        return min(1.0, score)

    async def _assess_location_precision(self, location: Any) -> float:
        """Assess coordinate precision quality."""
        lat_precision = self._assess_coordinate_precision(location.latitude)
        lon_precision = self._assess_coordinate_precision(location.longitude)
        avg_precision = (lat_precision + lon_precision) / 2

        if avg_precision >= 6:
            return 1.0
        elif avg_precision >= 5:
            return 0.9
        elif avg_precision >= 4:
            return 0.7
        elif avg_precision >= 3:
            return 0.5
        else:
            return 0.3

    async def categorize_location_content(
            self,
            location: Any,
            classification: Dict[str, Any],
            context: ProcessingContext
    ) -> List[str]:
        """Categorize location content based on analysis."""
        categories = ["location"]

        try:
            # Add primary type
            if classification.get("primary_type"):
                categories.append(classification["primary_type"])

            # Add business context
            if classification.get("business_context") == "business_related":
                categories.append("business_location")
            else:
                categories.append("personal_location")

            # Add accuracy-based categories
            if hasattr(location, 'accuracy_meters') and location.accuracy_meters:
                if location.accuracy_meters <= 10:
                    categories.append("precise_location")
                elif location.accuracy_meters <= 100:
                    categories.append("accurate_location")
                else:
                    categories.append("approximate_location")

            # Add channel-based categories
            if context.channel:
                categories.append(f"{context.channel}_location")

            return list(set(categories))

        except Exception as e:
            self.logger.error("Location categorization failed", error=str(e))
            return ["location"]

    def _generate_location_tags(
            self,
            location: Any,
            geocoding_result: Dict[str, Any],
            classification: Dict[str, Any]
    ) -> List[str]:
        """Generate tags for location content."""
        tags = ["location", "coordinates"]

        try:
            # Add geographic tags
            tags.extend(classification.get("topics", []))

            # Add place type tags
            if classification.get("primary_type"):
                tags.append(classification["primary_type"])

            # Add country/region tags from geocoding
            if geocoding_result.get("address_components"):
                addr = geocoding_result["address_components"]
                if addr.get("country"):
                    tags.append(f"country_{addr['country'].lower().replace(' ', '_')}")
                if addr.get("administrative_area_level_1"):
                    tags.append("region_specific")

            # Add accuracy tags
            if hasattr(location, 'accuracy_meters') and location.accuracy_meters:
                if location.accuracy_meters <= 10:
                    tags.append("high_precision")
                elif location.accuracy_meters <= 100:
                    tags.append("medium_precision")
                else:
                    tags.append("low_precision")

            return tags[:10]  # Limit to 10 tags

        except Exception as e:
            self.logger.error("Tag generation failed", error=str(e))
            return ["location"]

    def _load_place_categories(self) -> Dict[str, List[str]]:
        """Load place categories for classification."""
        return {
            "business": ["restaurant", "store", "office", "bank", "hotel", "gas_station"],
            "transportation": ["airport", "train_station", "bus_stop", "parking", "subway_station"],
            "healthcare": ["hospital", "pharmacy", "doctor", "dentist", "veterinarian"],
            "education": ["school", "university", "library", "museum"],
            "entertainment": ["movie_theater", "park", "stadium", "gym", "spa"],
            "government": ["city_hall", "post_office", "courthouse", "embassy"],
            "religious": ["church", "mosque", "temple", "synagogue"],
            "residential": ["home", "apartment", "neighborhood"]
        }

    def _load_coordinate_patterns(self) -> Dict[str, str]:
        """Load coordinate parsing patterns."""
        return {
            "decimal_degrees": r"^[-+]?([1-8]?\d(\.\d+)?|90(\.0+)?),\s*[-+]?(180(\.0+)?|((1[0-7]\d)|([1-9]?\d))(\.\d+)?)$",
            "degrees_minutes_seconds": r"^\d+°\d+'[\d.]+\"[NS],\s*\d+°\d+'[\d.]+\"[EW]$",
            "degrees_decimal_minutes": r"^\d+°[\d.]+[NS],\s*\d+°[\d.]+[EW]$"
        }