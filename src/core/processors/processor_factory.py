"""
Processor factory for dynamic component selection and creation.

This module provides factory functions for creating processor instances
based on message type and configuration.
"""

from typing import Dict, Type, Optional, Any, List, Set
import structlog

from src.core.processors.base_processor import BaseProcessor
from src.core.processors.text_processor import TextProcessor
from src.core.processors.media_processor import MediaProcessor
from src.core.processors.location_processor import LocationProcessor
from src.models.types import MessageType, MessageContent
from src.core.exceptions import ProcessorNotFoundError, ProcessingError

logger = structlog.get_logger(__name__)


class ProcessorFactory:
    """Factory for creating processor instances."""

    def __init__(self):
        self._processor_registry: Dict[MessageType, Type[BaseProcessor]] = {}
        self._processor_configs: Dict[str, Dict[str, Any]] = {}
        self._processor_instances: Dict[str, BaseProcessor] = {}

        # Register built-in processors
        self._register_builtin_processors()

        logger.info(
            "Processor factory initialized",
            registered_processors=list(self._processor_registry.keys())
        )

    def _register_builtin_processors(self) -> None:
        """Register built-in processor implementations."""
        self.register_processor(MessageType.TEXT, TextProcessor)
        self.register_processor(MessageType.IMAGE, MediaProcessor)
        self.register_processor(MessageType.AUDIO, MediaProcessor)
        self.register_processor(MessageType.VIDEO, MediaProcessor)
        self.register_processor(MessageType.FILE, MediaProcessor)
        self.register_processor(MessageType.LOCATION, LocationProcessor)

        # Additional message types can be registered here
        # self.register_processor(MessageType.QUICK_REPLY, TextProcessor)
        # self.register_processor(MessageType.CAROUSEL, MediaProcessor)

    def register_processor(
            self,
            message_type: MessageType,
            processor_class: Type[BaseProcessor]
    ) -> None:
        """
        Register a processor implementation for a message type.

        Args:
            message_type: Type of message to handle
            processor_class: Processor implementation class
        """
        if message_type in self._processor_registry:
            logger.warning(
                "Message type already registered, overriding",
                message_type=message_type.value,
                existing_class=self._processor_registry[message_type].__name__,
                new_class=processor_class.__name__
            )

        self._processor_registry[message_type] = processor_class

        logger.info(
            "Processor registered",
            message_type=message_type.value,
            processor_class=processor_class.__name__
        )

    def unregister_processor(self, message_type: MessageType) -> None:
        """
        Unregister a processor implementation.

        Args:
            message_type: Type of message to unregister
        """
        if message_type in self._processor_registry:
            del self._processor_registry[message_type]
            logger.info(
                "Processor unregistered",
                message_type=message_type.value
            )

    def get_registered_message_types(self) -> List[MessageType]:
        """Get list of registered message types."""
        return list(self._processor_registry.keys())

    def is_message_type_supported(self, message_type: MessageType) -> bool:
        """Check if a message type is supported."""
        return message_type in self._processor_registry

    def get_processor_for_message_type(
            self,
            message_type: MessageType,
            config: Optional[Dict[str, Any]] = None
    ) -> BaseProcessor:
        """
        Get a processor instance for a specific message type.

        Args:
            message_type: Type of message to process
            config: Optional processor configuration

        Returns:
            Configured processor instance

        Raises:
            ProcessorNotFoundError: If no processor is registered for message type
        """
        if message_type not in self._processor_registry:
            available_types = [mt.value for mt in self._processor_registry.keys()]
            raise ProcessorNotFoundError(
                content_type=message_type.value,
                available_processors=available_types
            )

        processor_class = self._processor_registry[message_type]

        try:
            # Create processor instance with configuration
            processor = processor_class(config)

            logger.debug(
                "Processor created",
                message_type=message_type.value,
                processor_class=processor_class.__name__
            )

            return processor

        except Exception as e:
            logger.error(
                "Processor creation failed",
                message_type=message_type.value,
                processor_class=processor_class.__name__,
                error=str(e)
            )

            raise ProcessingError(
                message=f"Failed to create processor for {message_type.value}: {str(e)}",
                error_code="PROCESSOR_CREATION_FAILED",
                details={
                    "message_type": message_type.value,
                    "processor_class": processor_class.__name__,
                    "error": str(e)
                }
            )

    def get_processor_for_content(
            self,
            content: MessageContent,
            config: Optional[Dict[str, Any]] = None
    ) -> BaseProcessor:
        """
        Get appropriate processor for message content.

        Args:
            content: Message content to analyze
            config: Optional processor configuration

        Returns:
            Configured processor instance

        Raises:
            ProcessorNotFoundError: If no suitable processor found
        """
        return self.get_processor_for_message_type(content.type, config)

    def get_processors_for_multiple_types(
            self,
            message_types: List[MessageType],
            config: Optional[Dict[str, Any]] = None
    ) -> Dict[MessageType, BaseProcessor]:
        """
        Get processors for multiple message types.

        Args:
            message_types: List of message types to get processors for
            config: Optional processor configuration

        Returns:
            Dictionary mapping message types to processor instances
        """
        processors = {}

        for message_type in message_types:
            try:
                processor = self.get_processor_for_message_type(message_type, config)
                processors[message_type] = processor
            except ProcessorNotFoundError:
                logger.warning(
                    "No processor available for message type",
                    message_type=message_type.value
                )
                continue

        logger.info(
            "Multiple processors created",
            requested_types=[mt.value for mt in message_types],
            created_types=[mt.value for mt in processors.keys()],
            success_count=len(processors)
        )

        return processors

    def get_all_processors(
            self,
            config: Optional[Dict[str, Any]] = None
    ) -> Dict[MessageType, BaseProcessor]:
        """
        Get processors for all registered message types.

        Args:
            config: Optional processor configuration

        Returns:
            Dictionary mapping all message types to processor instances
        """
        return self.get_processors_for_multiple_types(
            list(self._processor_registry.keys()),
            config
        )

    def get_or_create_processor(
            self,
            message_type: MessageType,
            instance_key: str,
            config: Optional[Dict[str, Any]] = None,
            force_recreate: bool = False
    ) -> BaseProcessor:
        """
        Get existing processor instance or create new one.

        Args:
            message_type: Type of message to process
            instance_key: Unique key for caching instance
            config: Optional processor configuration
            force_recreate: Force recreation of processor

        Returns:
            Processor instance
        """
        full_key = f"{message_type.value}:{instance_key}"

        # Return existing instance if available and not forcing recreation
        if not force_recreate and full_key in self._processor_instances:
            return self._processor_instances[full_key]

        # Create new instance
        processor = self.get_processor_for_message_type(message_type, config)
        self._processor_instances[full_key] = processor

        return processor

    def remove_processor_instance(self, message_type: MessageType, instance_key: str) -> None:
        """Remove cached processor instance."""
        full_key = f"{message_type.value}:{instance_key}"
        if full_key in self._processor_instances:
            del self._processor_instances[full_key]
            logger.debug(
                "Processor instance removed",
                message_type=message_type.value,
                instance_key=instance_key
            )

    def store_config(self, config_name: str, config: Dict[str, Any]) -> None:
        """Store a named configuration for reuse."""
        self._processor_configs[config_name] = config
        logger.debug(
            "Processor config stored",
            config_name=config_name
        )

    def get_stored_config(self, config_name: str) -> Optional[Dict[str, Any]]:
        """Retrieve a stored configuration."""
        return self._processor_configs.get(config_name)

    def detect_content_types(self, content: MessageContent) -> List[MessageType]:
        """
        Detect applicable content types for mixed content.

        Args:
            content: Message content to analyze

        Returns:
            List of applicable message types
        """
        detected_types = [content.type]  # Primary type

        try:
            # Detect additional types based on content
            if content.text and content.type != MessageType.TEXT:
                detected_types.append(MessageType.TEXT)

            if content.media and content.type not in [MessageType.IMAGE, MessageType.AUDIO, MessageType.VIDEO,
                                                      MessageType.FILE]:
                # Determine media type from MIME type
                if content.media.type.startswith('image/'):
                    detected_types.append(MessageType.IMAGE)
                elif content.media.type.startswith('audio/'):
                    detected_types.append(MessageType.AUDIO)
                elif content.media.type.startswith('video/'):
                    detected_types.append(MessageType.VIDEO)
                else:
                    detected_types.append(MessageType.FILE)

            if content.location and content.type != MessageType.LOCATION:
                detected_types.append(MessageType.LOCATION)

            return list(set(detected_types))  # Remove duplicates

        except Exception as e:
            logger.error(
                "Content type detection failed",
                error=str(e),
                primary_type=content.type.value
            )
            return [content.type]

    def get_processing_strategy(
            self,
            content: MessageContent,
            config: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Get processing strategy for content.

        Args:
            content: Message content to analyze
            config: Optional configuration

        Returns:
            Processing strategy information
        """
        strategy = {
            "primary_type": content.type,
            "detected_types": [],
            "processors_needed": [],
            "processing_order": [],
            "parallel_processing": False,
            "estimated_complexity": "medium"
        }

        try:
            # Detect all applicable content types
            detected_types = self.detect_content_types(content)
            strategy["detected_types"] = detected_types

            # Determine processors needed
            processors_needed = []
            for msg_type in detected_types:
                if self.is_message_type_supported(msg_type):
                    processors_needed.append(msg_type)

            strategy["processors_needed"] = processors_needed

            # Determine processing order
            # Text processing usually comes first, location last
            processing_order = []

            if MessageType.TEXT in processors_needed:
                processing_order.append(MessageType.TEXT)

            # Media types can be processed in parallel
            media_types = [MessageType.IMAGE, MessageType.AUDIO, MessageType.VIDEO, MessageType.FILE]
            media_in_content = [mt for mt in processors_needed if mt in media_types]
            processing_order.extend(media_in_content)

            if MessageType.LOCATION in processors_needed:
                processing_order.append(MessageType.LOCATION)

            strategy["processing_order"] = processing_order

            # Determine if parallel processing is beneficial
            strategy["parallel_processing"] = len(media_in_content) > 1

            # Estimate complexity
            complexity_score = len(processors_needed)
            if content.text and len(content.text) > 1000:
                complexity_score += 1
            if content.media and hasattr(content.media, 'size_bytes') and content.media.size_bytes > 10 * 1024 * 1024:
                complexity_score += 1

            if complexity_score <= 1:
                strategy["estimated_complexity"] = "low"
            elif complexity_score <= 3:
                strategy["estimated_complexity"] = "medium"
            else:
                strategy["estimated_complexity"] = "high"

            return strategy

        except Exception as e:
            logger.error(
                "Processing strategy determination failed",
                error=str(e),
                content_type=content.type.value
            )
            return strategy

    async def health_check_all_processors(
            self,
            config: Optional[Dict[str, Any]] = None
    ) -> Dict[MessageType, bool]:
        """
        Perform health checks on all processor types.

        Args:
            config: Optional processor configuration

        Returns:
            Dictionary mapping message types to health status
        """
        health_results = {}

        for message_type in self._processor_registry.keys():
            try:
                processor = self.get_processor_for_message_type(message_type, config)
                is_healthy = await processor.health_check()
                health_results[message_type] = is_healthy

                logger.debug(
                    "Processor health check completed",
                    message_type=message_type.value,
                    healthy=is_healthy
                )

            except Exception as e:
                health_results[message_type] = False
                logger.error(
                    "Processor health check failed",
                    message_type=message_type.value,
                    error=str(e)
                )

        return health_results

    def get_processor_metrics(
            self,
            config: Optional[Dict[str, Any]] = None
    ) -> Dict[MessageType, Dict[str, Any]]:
        """
        Get metrics for all active processor instances.

        Args:
            config: Optional processor configuration

        Returns:
            Dictionary mapping message types to metrics
        """
        metrics_results = {}

        # Get metrics from cached instances
        for instance_key, processor in self._processor_instances.items():
            try:
                message_type_str = instance_key.split(':', 1)[0]
                message_type = MessageType(message_type_str)

                metrics = processor.get_metrics()

                if message_type not in metrics_results:
                    metrics_results[message_type] = []

                metrics_results[message_type].append({
                    "instance_key": instance_key,
                    "metrics": metrics.dict()
                })

            except Exception as e:
                logger.error(
                    "Failed to get processor metrics",
                    instance_key=instance_key,
                    error=str(e)
                )

        return metrics_results

    def get_processor_capabilities(self) -> Dict[MessageType, Dict[str, Any]]:
        """
        Get capabilities of all registered processors.

        Returns:
            Dictionary mapping message types to capabilities
        """
        capabilities = {}

        for message_type, processor_class in self._processor_registry.items():
            try:
                # Create temporary instance to get capabilities
                temp_processor = processor_class()

                capabilities[message_type] = {
                    "processor_name": temp_processor.processor_name,
                    "processor_version": temp_processor.processor_version,
                    "supported_message_types": [mt.value for mt in temp_processor.supported_message_types],
                    "class_name": processor_class.__name__,
                    "module": processor_class.__module__
                }

            except Exception as e:
                logger.error(
                    "Failed to get processor capabilities",
                    message_type=message_type.value,
                    processor_class=processor_class.__name__,
                    error=str(e)
                )

                capabilities[message_type] = {
                    "error": str(e),
                    "class_name": processor_class.__name__
                }

        return capabilities

    def clear_processor_cache(self) -> None:
        """Clear all cached processor instances."""
        instance_count = len(self._processor_instances)
        self._processor_instances.clear()

        logger.info(
            "Processor cache cleared",
            instances_removed=instance_count
        )

    def get_factory_stats(self) -> Dict[str, Any]:
        """Get factory statistics."""
        return {
            "registered_processors": len(self._processor_registry),
            "cached_instances": len(self._processor_instances),
            "stored_configs": len(self._processor_configs),
            "supported_message_types": [mt.value for mt in self._processor_registry.keys()],
            "active_instances": list(self._processor_instances.keys())
        }


# Global factory instance
processor_factory = ProcessorFactory()


# Convenience functions
def get_processor_for_message_type(
        message_type: MessageType,
        config: Optional[Dict[str, Any]] = None
) -> BaseProcessor:
    """Get a processor using the global factory."""
    return processor_factory.get_processor_for_message_type(message_type, config)


def get_processor_for_content(
        content: MessageContent,
        config: Optional[Dict[str, Any]] = None
) -> BaseProcessor:
    """Get appropriate processor for content using the global factory."""
    return processor_factory.get_processor_for_content(content, config)


def get_processing_strategy(
        content: MessageContent,
        config: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """Get processing strategy for content using the global factory."""
    return processor_factory.get_processing_strategy(content, config)


def is_message_type_supported(message_type: MessageType) -> bool:
    """Check if message type is supported."""
    return processor_factory.is_message_type_supported(message_type)


def get_supported_message_types() -> List[MessageType]:
    """Get list of supported message types."""
    return processor_factory.get_registered_message_types()


def register_custom_processor(
        message_type: MessageType,
        processor_class: Type[BaseProcessor]
) -> None:
    """Register a custom processor implementation."""
    processor_factory.register_processor(message_type, processor_class)