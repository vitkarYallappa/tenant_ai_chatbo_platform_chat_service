"""
Common Validation Models
Shared validation models and utilities used across different API endpoints.
"""

from datetime import datetime
from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field, validator
import re


class PhoneNumberValidator(BaseModel):
    """Validator for E.164 phone number format"""
    phone: str

    @validator('phone')
    def validate_e164(cls, v):
        pattern = r'^\+[1-9]\d{1,14}$'
        if not re.match(pattern, v):
            raise ValueError('Invalid E.164 phone number format')
        return v


class EmailValidator(BaseModel):
    """Validator for email addresses"""
    email: str

    @validator('email')
    def validate_email(cls, v):
        pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
        if not re.match(pattern, v.lower()):
            raise ValueError('Invalid email format')
        return v.lower()


class URLValidator(BaseModel):
    """Validator for URLs"""
    url: str

    @validator('url')
    def validate_url(cls, v):
        pattern = r'^https?://(?:[-\w.])+(?:\:[0-9]+)?(?:/(?:[\w/_.])*(?:\?(?:[\w&=%.])*)?(?:\#(?:\w*))?)?$'
        if not re.match(pattern, v):
            raise ValueError('Invalid URL format')
        return v


class PaginationParams(BaseModel):
    """Common pagination parameters"""
    page: int = Field(default=1, ge=1, description="Page number")
    page_size: int = Field(default=20, ge=1, le=100, description="Items per page")
    sort_by: Optional[str] = Field(None, description="Field to sort by")
    sort_order: str = Field(default="desc", regex=r'^(asc|desc)$', description="Sort order")


class DateRangeFilter(BaseModel):
    """Date range filter for queries"""
    start_date: Optional[datetime] = None
    end_date: Optional[datetime] = None

    @validator('end_date')
    def validate_date_range(cls, v, values):
        if v and values.get('start_date') and v < values['start_date']:
            raise ValueError('End date must be after start date')
        return v

    @validator('start_date', 'end_date')
    def validate_future_dates(cls, v):
        if v and v > datetime.utcnow():
            raise ValueError('Date cannot be in the future')
        return v


class SearchParams(BaseModel):
    """Common search parameters"""
    query: Optional[str] = Field(None, max_length=200, description="Search query")
    exact_match: bool = Field(default=False, description="Whether to use exact matching")
    case_sensitive: bool = Field(default=False, description="Whether search is case sensitive")
    fields: Optional[List[str]] = Field(None, max_items=10, description="Fields to search in")

    @validator('query')
    def validate_query(cls, v):
        if v and len(v.strip()) < 2:
            raise ValueError('Search query must be at least 2 characters long')
        return v.strip() if v else v


class MetadataValidator(BaseModel):
    """Validator for metadata objects"""
    metadata: Dict[str, Any] = Field(default_factory=dict)

    @validator('metadata')
    def validate_metadata_size(cls, v):
        # Convert to JSON string to check size
        import json
        metadata_str = json.dumps(v)
        if len(metadata_str) > 32768:  # 32KB limit
            raise ValueError('Metadata size cannot exceed 32KB')
        return v

    @validator('metadata')
    def validate_metadata_depth(cls, v):
        def check_depth(obj, current_depth=0, max_depth=5):
            if current_depth > max_depth:
                raise ValueError(f'Metadata nesting depth cannot exceed {max_depth} levels')
            if isinstance(obj, dict):
                for value in obj.values():
                    check_depth(value, current_depth + 1, max_depth)
            elif isinstance(obj, list):
                for item in obj:
                    check_depth(item, current_depth + 1, max_depth)

        check_depth(v)
        return v


class TagsValidator(BaseModel):
    """Validator for tags"""
    tags: List[str] = Field(default_factory=list, max_items=20)

    @validator('tags')
    def validate_tags(cls, v):
        if not v:
            return v

        # Validate each tag
        validated_tags = []
        for tag in v:
            tag = tag.strip().lower()
            if not tag:
                continue
            if len(tag) > 50:
                raise ValueError('Each tag must be 50 characters or less')
            if not re.match(r'^[a-z0-9_-]+$', tag):
                raise ValueError('Tags can only contain lowercase letters, numbers, hyphens, and underscores')
            validated_tags.append(tag)

        # Remove duplicates while preserving order
        seen = set()
        unique_tags = []
        for tag in validated_tags:
            if tag not in seen:
                seen.add(tag)
                unique_tags.append(tag)

        return unique_tags


class TimeRangeValidator(BaseModel):
    """Validator for time ranges with common patterns"""
    time_range: str = Field(..., regex=r'^(last_hour|last_24h|last_7d|last_30d|last_90d|custom)$')
    custom_start: Optional[datetime] = None
    custom_end: Optional[datetime] = None

    @validator('custom_start', 'custom_end')
    def validate_custom_dates(cls, v, values):
        if values.get('time_range') == 'custom':
            if not v:
                raise ValueError('Custom start and end dates are required when time_range is "custom"')
        elif v:
            raise ValueError('Custom dates should only be provided when time_range is "custom"')
        return v

    @validator('custom_end')
    def validate_custom_range(cls, v, values):
        if v and values.get('custom_start') and v <= values['custom_start']:
            raise ValueError('Custom end date must be after start date')
        return v


class LanguageValidator(BaseModel):
    """Validator for language codes"""
    language: str = Field(..., regex=r'^[a-z]{2}(-[A-Z]{2})?$')

    @validator('language')
    def validate_supported_language(cls, v):
        # Common supported languages - can be extended
        supported_languages = [
            'en', 'es', 'fr', 'de', 'it', 'pt', 'ru', 'ja', 'ko', 'zh',
            'ar', 'hi', 'bn', 'tr', 'pl', 'nl', 'sv', 'da', 'no', 'fi'
        ]

        base_lang = v.split('-')[0]
        if base_lang not in supported_languages:
            raise ValueError(f'Language "{base_lang}" is not supported')
        return v


class CurrencyValidator(BaseModel):
    """Validator for currency codes and amounts"""
    currency_code: str = Field(..., regex=r'^[A-Z]{3}$')
    amount: float = Field(..., ge=0)

    @validator('currency_code')
    def validate_currency(cls, v):
        # Common supported currencies
        supported_currencies = [
            'USD', 'EUR', 'GBP', 'JPY', 'CNY', 'INR', 'BRL', 'CAD',
            'AUD', 'CHF', 'SEK', 'NOK', 'DKK', 'PLN', 'CZK', 'HUF'
        ]

        if v not in supported_currencies:
            raise ValueError(f'Currency "{v}" is not supported')
        return v

    @validator('amount')
    def validate_amount_precision(cls, v):
        # Limit to 2 decimal places for most currencies
        if round(v, 2) != v:
            raise ValueError('Amount cannot have more than 2 decimal places')
        return v


class FileValidator(BaseModel):
    """Validator for file uploads"""
    filename: str
    content_type: str
    size_bytes: int = Field(..., ge=1)

    @validator('filename')
    def validate_filename(cls, v):
        if not v or len(v.strip()) == 0:
            raise ValueError('Filename cannot be empty')

        # Check for path traversal attempts
        if '..' in v or '/' in v or '\\' in v:
            raise ValueError('Invalid filename')

        # Check length
        if len(v) > 255:
            raise ValueError('Filename too long (max 255 characters)')

        return v.strip()

    @validator('content_type')
    def validate_content_type(cls, v):
        allowed_types = [
            'image/jpeg', 'image/png', 'image/gif', 'image/webp',
            'video/mp4', 'video/webm', 'video/quicktime',
            'audio/mpeg', 'audio/wav', 'audio/ogg',
            'application/pdf', 'text/plain', 'text/csv',
            'application/json', 'application/xml'
        ]

        if v not in allowed_types:
            raise ValueError(f'Content type "{v}" is not allowed')
        return v

    @validator('size_bytes')
    def validate_file_size(cls, v, values):
        # Size limits by content type
        size_limits = {
            'image/': 10 * 1024 * 1024,  # 10MB for images
            'video/': 100 * 1024 * 1024,  # 100MB for videos
            'audio/': 50 * 1024 * 1024,  # 50MB for audio
            'application/pdf': 25 * 1024 * 1024,  # 25MB for PDFs
            'text/': 1 * 1024 * 1024,  # 1MB for text files
        }

        content_type = values.get('content_type', '')
        for type_prefix, limit in size_limits.items():
            if content_type.startswith(type_prefix):
                if v > limit:
                    raise ValueError(f'File size exceeds limit of {limit // (1024 * 1024)}MB for {type_prefix} files')
                break
        else:
            # Default limit for other types
            if v > 5 * 1024 * 1024:  # 5MB
                raise ValueError('File size exceeds default limit of 5MB')

        return v


class ConfigurationValidator(BaseModel):
    """Validator for configuration objects"""
    config_type: str = Field(..., regex=r'^[a-z_]+$')
    config_data: Dict[str, Any]
    version: str = Field(default="1.0", regex=r'^\d+\.\d+$')

    @validator('config_type')
    def validate_config_type(cls, v):
        allowed_types = [
            'channel_config', 'flow_config', 'model_config',
            'integration_config', 'analytics_config', 'security_config'
        ]

        if v not in allowed_types:
            raise ValueError(f'Configuration type "{v}" is not supported')
        return v

    @validator('config_data')
    def validate_config_data(cls, v):
        if not v:
            raise ValueError('Configuration data cannot be empty')

        # Check for required fields based on config type
        # This would be expanded based on specific config requirements
        return v


class BulkOperationValidator(BaseModel):
    """Validator for bulk operations"""
    operation_type: str = Field(..., regex=r'^(create|update|delete)$')
    batch_size: int = Field(default=100, ge=1, le=1000)
    items: List[Dict[str, Any]] = Field(..., min_items=1)

    @validator('items')
    def validate_batch_size_consistency(cls, v, values):
        batch_size = values.get('batch_size', 100)
        if len(v) > batch_size:
            raise ValueError(f'Number of items ({len(v)}) exceeds batch size ({batch_size})')
        return v