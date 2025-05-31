"""
Normalizers package for message and content normalization.

This package provides normalizers for converting messages from different channels
into a consistent format for processing.
"""

from src.core.normalizers.message_normalizer import MessageNormalizer
from src.core.normalizers.content_normalizer import ContentNormalizer
from src.core.normalizers.metadata_normalizer import MetadataNormalizer

__all__ = [
    "MessageNormalizer",
    "ContentNormalizer",
    "MetadataNormalizer"
]

# Version information
__version__ = "1.0.0"
__author__ = "Chatbot Platform Team"
__description__ = "Message and content normalization utilities"