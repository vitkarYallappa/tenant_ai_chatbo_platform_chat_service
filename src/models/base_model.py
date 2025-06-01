# src/models/base_model.py
"""
Base model classes providing common functionality for all models.
Includes audit trails, validation helpers, and serialization utilities.
"""

from datetime import datetime
from typing import Optional, Dict, Any, List
from pydantic import BaseModel, Field, validator
from bson import ObjectId
import json
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


class BaseMongoModel(BaseModel):
    """
    Base model for MongoDB documents with common fields and utilities.
    Provides audit trail fields and serialization helpers.
    """

    # MongoDB ObjectId field
    id: Optional[ObjectId] = Field(default=None, alias="_id")

    # Audit trail fields
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)

    # Version control for optimistic locking
    version: int = Field(default=1)

    class Config:
        allow_population_by_field_name = True
        arbitrary_types_allowed = True
        json_encoders = {
            ObjectId: str,
            datetime: lambda v: v.isoformat()
        }

        # Use enum values in serialization
        use_enum_values = True

    def to_dict(self, exclude_none: bool = True, by_alias: bool = True) -> Dict[str, Any]:
        """
        Convert model to dictionary format suitable for MongoDB storage.

        Args:
            exclude_none: Whether to exclude None values
            by_alias: Whether to use field aliases (like _id)

        Returns:
            Dictionary representation of the model
        """
        data = self.dict(exclude_none=exclude_none, by_alias=by_alias)

        # Handle ObjectId serialization
        if self.id and by_alias:
            data["_id"] = self.id

        return data

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "BaseMongoModel":
        """
        Create model instance from dictionary (typically from MongoDB).

        Args:
            data: Dictionary data from MongoDB

        Returns:
            Model instance
        """
        # Handle MongoDB ObjectId field
        if "_id" in data:
            data["id"] = data.pop("_id")

        return cls(**data)

    def update_timestamp(self) -> None:
        """Update the updated_at timestamp and increment version."""
        self.updated_at = datetime.utcnow()
        self.version += 1

    def to_json(self, **kwargs) -> str:
        """
        Convert model to JSON string.

        Returns:
            JSON string representation
        """
        return self.json(**kwargs)

    @classmethod
    def from_json(cls, json_str: str) -> "BaseMongoModel":
        """
        Create model instance from JSON string.

        Args:
            json_str: JSON string representation

        Returns:
            Model instance
        """
        return cls.parse_raw(json_str)


class BaseRedisModel(BaseModel):
    """
    Base model for Redis data structures with serialization utilities.
    Provides methods for converting to/from Redis hash format.
    """

    # Timestamp fields
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)

    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }
        use_enum_values = True

    def to_redis_hash(self) -> Dict[str, str]:
        """
        Convert model to Redis hash format (all string values).

        Returns:
            Dictionary with string keys and values for Redis HSET
        """
        data = self.dict()
        redis_data = {}

        for key, value in data.items():
            if value is None:
                continue
            elif isinstance(value, (dict, list)):
                redis_data[key] = json.dumps(value)
            elif isinstance(value, datetime):
                redis_data[key] = value.isoformat()
            elif isinstance(value, bool):
                redis_data[key] = "true" if value else "false"
            else:
                redis_data[key] = str(value)

        return redis_data

    @classmethod
    def from_redis_hash(cls, data: Dict[str, str]) -> "BaseRedisModel":
        """
        Create model instance from Redis hash data.

        Args:
            data: Dictionary from Redis HGETALL

        Returns:
            Model instance
        """
        processed_data = {}

        # Get model field info for proper type conversion
        field_info = cls.__fields__

        for key, value in data.items():
            if key not in field_info:
                continue

            field = field_info[key]
            field_type = field.type_

            if value == "":
                processed_data[key] = None
            elif field_type == datetime:
                processed_data[key] = datetime.fromisoformat(value)
            elif field_type == bool:
                processed_data[key] = value.lower() == "true"
            elif field_type == int:
                processed_data[key] = int(value)
            elif field_type == float:
                processed_data[key] = float(value)
            elif hasattr(field_type, '__origin__') and field_type.__origin__ in (dict, list):
                processed_data[key] = json.loads(value)
            else:
                processed_data[key] = value

        return cls(**processed_data)

    def update_timestamp(self) -> None:
        """Update the updated_at timestamp."""
        self.updated_at = datetime.utcnow()


class BasePostgresModel(DeclarativeBase):
    """Base class for all SQLAlchemy models"""

    def to_dict(self) -> Dict[str, Any]:
        """Convert model instance to dictionary"""
        result = {}
        for column in self.__table__.columns:
            value = getattr(self, column.name)
            if isinstance(value, datetime):
                result[column.name] = value.isoformat()
            elif hasattr(value, 'value'):  # Handle enums
                result[column.name] = value.value
            else:
                result[column.name] = value
        return result

    @classmethod
    def from_dict(cls, data: Dict[str, Any]):
        """Create model instance from dictionary"""
        # Filter out keys that don't exist in the model
        valid_keys = {c.name for c in cls.__table__.columns}
        filtered_data = {k: v for k, v in data.items() if k in valid_keys}
        return cls(**filtered_data)

    def update_from_dict(self, data: Dict[str, Any]) -> None:
        """Update model instance from dictionary"""
        valid_keys = {c.name for c in self.__table__.columns}
        for key, value in data.items():
            if key in valid_keys and hasattr(self, key):
                setattr(self, key, value)

class BaseRequestModel(BaseModel):
    """
    Base model for API request validation.
    Provides common request fields and validation.
    """

    class Config:
        # Validate on assignment
        validate_assignment = True
        # Use enum values
        use_enum_values = True
        # Allow extra fields (for extensibility)
        extra = "forbid"

    def validate_required_fields(self, fields: List[str]) -> None:
        """
        Validate that required fields are present and not None.

        Args:
            fields: List of field names to validate

        Raises:
            ValueError: If any required field is missing or None
        """
        missing_fields = []

        for field_name in fields:
            if not hasattr(self, field_name) or getattr(self, field_name) is None:
                missing_fields.append(field_name)

        if missing_fields:
            raise ValueError(f"Missing required fields: {', '.join(missing_fields)}")


class BaseResponseModel(BaseModel):
    """
    Base model for API responses.
    Provides consistent response structure and metadata.
    """

    # Response metadata
    meta: Dict[str, Any] = Field(default_factory=dict)

    class Config:
        use_enum_values = True
        json_encoders = {
            datetime: lambda v: v.isoformat(),
            ObjectId: str
        }

    def add_meta(self, key: str, value: Any) -> None:
        """
        Add metadata to the response.

        Args:
            key: Metadata key
            value: Metadata value
        """
        self.meta[key] = value

    def set_processing_time(self, start_time: datetime) -> None:
        """
        Calculate and set processing time in metadata.

        Args:
            start_time: Request start timestamp
        """
        processing_time = datetime.utcnow() - start_time
        self.add_meta("processing_time_ms", int(processing_time.total_seconds() * 1000))


class TimestampMixin(BaseModel):
    """
    Mixin for models that need timestamp tracking.
    Can be used with multiple inheritance.
    """

    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)

    def touch(self) -> None:
        """Update the updated_at timestamp."""
        self.updated_at = datetime.utcnow()


class AuditMixin(BaseModel):
    """
    Mixin for models that need audit trail tracking.
    Tracks who created/modified records.
    """

    created_by: Optional[str] = None
    updated_by: Optional[str] = None

    def set_creator(self, user_id: str) -> None:
        """Set the creator user ID."""
        self.created_by = user_id

    def set_updater(self, user_id: str) -> None:
        """Set the updater user ID."""
        self.updated_by = user_id


class SoftDeleteMixin(BaseModel):
    """
    Mixin for models that support soft deletion.
    Marks records as deleted instead of physically removing them.
    """

    deleted_at: Optional[datetime] = None
    deleted_by: Optional[str] = None
    is_deleted: bool = Field(default=False)

    def soft_delete(self, user_id: Optional[str] = None) -> None:
        """Mark record as deleted."""
        self.is_deleted = True
        self.deleted_at = datetime.utcnow()
        if user_id:
            self.deleted_by = user_id

    def restore(self) -> None:
        """Restore a soft-deleted record."""
        self.is_deleted = False
        self.deleted_at = None
        self.deleted_by = None


class ValidationHelpers:
    """
    Static helper methods for common validations.
    """

    @staticmethod
    def validate_non_empty_string(value: str, field_name: str) -> str:
        """
        Validate that a string is not empty or whitespace only.

        Args:
            value: String value to validate
            field_name: Name of the field for error messages

        Returns:
            Stripped string value

        Raises:
            ValueError: If string is empty or whitespace only
        """
        if not value or not value.strip():
            raise ValueError(f"{field_name} cannot be empty")
        return value.strip()

    @staticmethod
    def validate_positive_integer(value: int, field_name: str) -> int:
        """
        Validate that an integer is positive.

        Args:
            value: Integer value to validate
            field_name: Name of the field for error messages

        Returns:
            Integer value

        Raises:
            ValueError: If integer is not positive
        """
        if value <= 0:
            raise ValueError(f"{field_name} must be positive")
        return value

    @staticmethod
    def validate_uuid_format(value: str, field_name: str) -> str:
        """
        Validate that a string is a valid UUID format.

        Args:
            value: String value to validate
            field_name: Name of the field for error messages

        Returns:
            UUID string

        Raises:
            ValueError: If string is not a valid UUID
        """
        from uuid import UUID
        try:
            UUID(value)
            return value
        except ValueError:
            raise ValueError(f"{field_name} must be a valid UUID")

    @staticmethod
    def validate_url_format(value: str, field_name: str) -> str:
        """
        Validate that a string is a valid URL format.

        Args:
            value: String value to validate
            field_name: Name of the field for error messages

        Returns:
            URL string

        Raises:
            ValueError: If string is not a valid URL
        """
        import re
        url_pattern = re.compile(
            r'^https?://'  # http:// or https://
            r'(?:(?:[A-Z0-9](?:[A-Z0-9-]{0,61}[A-Z0-9])?\.)+[A-Z]{2,6}\.?|'  # domain...
            r'localhost|'  # localhost...
            r'\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3})'  # ...or ip
            r'(?::\d+)?'  # optional port
            r'(?:/?|[/?]\S+)$', re.IGNORECASE)

        if not url_pattern.match(value):
            raise ValueError(f"{field_name} must be a valid URL")
        return value


# Export utility functions for easy access
__all__ = [
    "BaseMongoModel",
    "BaseRedisModel",
    "BasePostgresModel",
    "BaseRequestModel",
    "BaseResponseModel",
    "TimestampMixin",
    "AuditMixin",
    "SoftDeleteMixin",
    "ValidationHelpers"
]