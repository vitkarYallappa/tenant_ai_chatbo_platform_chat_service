"""
Processors package for message content processing and analysis.

This package provides processors for analyzing and transforming different types
of message content including text, media, and location data.
"""

from src.core.processors.base_processor import (
    BaseProcessor,
    ProcessingContext,
    ProcessingResult,
    ProcessorMetrics
)
from src.core.processors.text_processor import TextProcessor
from src.core.processors.media_processor import MediaProcessor
from src.core.processors.location_processor import LocationProcessor
from src.core.processors.processor_factory import (
    ProcessorFactory,
    processor_factory,
    get_processor_for_message_type,
    get_processor_for_content,
    get_processing_strategy,
    is_message_type_supported,
    get_supported_message_types,
    register_custom_processor
)

__all__ = [
    # Base classes and data models
    "BaseProcessor",
    "ProcessingContext",
    "ProcessingResult",
    "ProcessorMetrics",

    # Processor implementations
    "TextProcessor",
    "MediaProcessor",
    "LocationProcessor",

    # Factory and convenience functions
    "ProcessorFactory",
    "processor_factory",
    "get_processor_for_message_type",
    "get_processor_for_content",
    "get_processing_strategy",
    "is_message_type_supported",
    "get_supported_message_types",
    "register_custom_processor",
    "get_processor_health_status"
]

# Version information
__version__ = "1.0.0"
__author__ = "Chatbot Platform Team"
__description__ = "Message content processing and analysis utilities"

# Module-level configuration
DEFAULT_PROCESSOR_CONFIG = {
    "text": {
        "max_text_length": 10000,
        "enable_language_detection": True,
        "enable_sentiment_analysis": True,
        "enable_keyword_extraction": True,
        "enable_text_normalization": True
    },
    "media": {
        "max_file_size_mb": 50,
        "enable_metadata_extraction": True,
        "enable_content_analysis": True,
        "enable_virus_scanning": False,
        "enable_ocr": False,
        "enable_transcription": False
    },
    "location": {
        "enable_geocoding": False,
        "enable_reverse_geocoding": True,
        "enable_nearby_search": False,
        "enable_location_validation": True,
        "enable_location_anonymization": False
    }
}


def get_default_processor_config(processor_type: str = None) -> dict:
    """
    Get default configuration for processors.

    Args:
        processor_type: Specific processor type ('text', 'media', 'location')
                       or None for all configs

    Returns:
        Configuration dictionary
    """
    if processor_type and processor_type in DEFAULT_PROCESSOR_CONFIG:
        return DEFAULT_PROCESSOR_CONFIG[processor_type].copy()
    return DEFAULT_PROCESSOR_CONFIG.copy()


def create_processor_suite(config: dict = None) -> dict:
    """
    Create a complete suite of processors with optional configuration.

    Args:
        config: Optional configuration overrides

    Returns:
        Dictionary mapping message types to processor instances
    """
    from src.models.types import MessageType

    # Merge with default config
    final_config = DEFAULT_PROCESSOR_CONFIG.copy()
    if config:
        for key, value in config.items():
            if key in final_config and isinstance(value, dict):
                final_config[key].update(value)
            else:
                final_config[key] = value

    processors = {}

    # Create text processor
    processors[MessageType.TEXT] = TextProcessor(final_config.get("text", {}))

    # Create media processors (shared instance for all media types)
    media_processor = MediaProcessor(final_config.get("media", {}))
    processors[MessageType.IMAGE] = media_processor
    processors[MessageType.AUDIO] = media_processor
    processors[MessageType.VIDEO] = media_processor
    processors[MessageType.FILE] = media_processor

    # Create location processor
    processors[MessageType.LOCATION] = LocationProcessor(final_config.get("location", {}))

    return processors


async def process_content_with_strategy(
        content,
        context,
        config: dict = None
):
    """
    Process content using the optimal strategy for the content type(s).

    Args:
        content: MessageContent to process
        context: ProcessingContext
        config: Optional processor configuration

    Returns:
        ProcessingResult or list of ProcessingResults for multi-type content
    """
    strategy = get_processing_strategy(content, config)

    if len(strategy["processors_needed"]) == 1:
        # Single processor needed
        processor = get_processor_for_content(content, config)
        return await processor.process(content, context)

    elif len(strategy["processors_needed"]) > 1:
        # Multiple processors needed
        results = []

        if strategy["parallel_processing"]:
            # Process in parallel for media types
            import asyncio
            tasks = []

            for msg_type in strategy["processors_needed"]:
                processor = get_processor_for_message_type(msg_type, config)
                tasks.append(processor.process(content, context))

            results = await asyncio.gather(*tasks, return_exceptions=True)
        else:
            # Process sequentially
            for msg_type in strategy["processing_order"]:
                processor = get_processor_for_message_type(msg_type, config)
                result = await processor.process(content, context)
                results.append(result)

        return results

    else:
        # No processors available
        raise ValueError(f"No processors available for content type: {content.type}")


def get_processor_health_status(config: dict = None) -> dict:
    """
    Get health status of all available processors.

    Args:
        config: Optional processor configuration

    Returns:
        Dictionary with health status for each processor type
    """
    import asyncio

    async def check_health():
        return await processor_factory.health_check_all_processors(config)

    return asyncio.run(check_health())


def get_processor_capabilities() -> dict:
    """
    Get capabilities of all registered processors.

    Returns:
        Dictionary mapping message types to processor capabilities
    """
    return processor_factory.get_processor_capabilities()


def reset_processor_factory():
    """Reset the global processor factory to initial state."""
    processor_factory.clear_processor_cache()
    processor_factory._processor_instances.clear()


# Initialize the factory with built-in processors on import
_factory_initialized = False


def _ensure_factory_initialized():
    """Ensure the factory is initialized with built-in processors."""
    global _factory_initialized
    if not _factory_initialized:
        # Factory auto-initializes with built-in processors
        _factory_initialized = True


# Initialize on module import
_ensure_factory_initialized()