"""
Validation-specific exceptions for Chat Service.

This module provides custom exception classes for validation errors
including field validation, business rule validation, and data
format validation issues.
"""

from typing import Any, Dict, List, Optional, Union

from src.exceptions.base_exceptions import ChatServiceException
from src.config.constants import ErrorCategory


class ValidationException(ChatServiceException):
    """Base exception for all validation-related errors."""

    def __init__(
            self,
            message: str,
            field: Optional[str] = None,
            value: Optional[Any] = None,
            **kwargs
    ):
        details = kwargs.pop("details", {})
        if field:
            details["field"] = field
        if value is not None:
            details["invalid_value"] = str(value)

        super().__init__(
            message=message,
            error_code="VALIDATION_ERROR",
            status_code=400,
            category=ErrorCategory.VALIDATION,
            user_message="Invalid input data. Please check your request and try again.",
            details=details,
            **kwargs
        )


class FieldValidationError(ValidationException):
    """Exception for single field validation errors."""

    def __init__(
            self,
            field: str,
            value: Any,
            reason: str,
            expected_type: Optional[str] = None,
            **kwargs
    ):
        details = kwargs.pop("details", {})
        details.update({
            "field": field,
            "invalid_value": str(value),
            "reason": reason
        })

        if expected_type:
            details["expected_type"] = expected_type

        message = f"Field validation failed for '{field}': {reason}"
        user_message = f"Invalid value for {field}: {reason}"

        super().__init__(
            message=message,
            field=field,
            value=value,
            error_code="FIELD_VALIDATION_ERROR",
            user_message=user_message,
            details=details,
            **kwargs
        )


class MultipleFieldValidationError(ValidationException):
    """Exception for multiple field validation errors."""

    def __init__(
            self,
            field_errors: Dict[str, str],
            **kwargs
    ):
        details = kwargs.pop("details", {})
        details["field_errors"] = field_errors

        error_count = len(field_errors)
        fields = ", ".join(field_errors.keys())

        message = f"Validation failed for {error_count} field(s): {fields}"
        user_message = f"Validation errors found in {error_count} field(s). Please check your input."

        super().__init__(
            message=message,
            error_code="MULTIPLE_FIELD_VALIDATION_ERROR",
            user_message=user_message,
            details=details,
            **kwargs
        )


class RequiredFieldError(ValidationException):
    """Exception for missing required fields."""

    def __init__(
            self,
            field: str,
            **kwargs
    ):
        details = kwargs.pop("details", {})
        details["required_field"] = field

        message = f"Required field '{field}' is missing"
        user_message = f"The field '{field}' is required"

        super().__init__(
            message=message,
            field=field,
            error_code="REQUIRED_FIELD_ERROR",
            user_message=user_message,
            details=details,
            **kwargs
        )


class InvalidFormatError(ValidationException):
    """Exception for invalid data format errors."""

    def __init__(
            self,
            field: str,
            value: Any,
            expected_format: str,
            example: Optional[str] = None,
            **kwargs
    ):
        details = kwargs.pop("details", {})
        details.update({
            "field": field,
            "invalid_value": str(value),
            "expected_format": expected_format
        })

        if example:
            details["example"] = example

        message = f"Invalid format for '{field}': expected {expected_format}"
        user_message = f"Invalid format for {field}. Expected format: {expected_format}"

        if example:
            user_message += f" (example: {example})"

        super().__init__(
            message=message,
            field=field,
            value=value,
            error_code="INVALID_FORMAT_ERROR",
            user_message=user_message,
            details=details,
            **kwargs
        )


class ValueOutOfRangeError(ValidationException):
    """Exception for values outside acceptable range."""

    def __init__(
            self,
            field: str,
            value: Union[int, float],
            min_value: Optional[Union[int, float]] = None,
            max_value: Optional[Union[int, float]] = None,
            **kwargs
    ):
        details = kwargs.pop("details", {})
        details.update({
            "field": field,
            "value": value
        })

        if min_value is not None:
            details["min_value"] = min_value
        if max_value is not None:
            details["max_value"] = max_value

        # Build range description
        if min_value is not None and max_value is not None:
            range_desc = f"between {min_value} and {max_value}"
        elif min_value is not None:
            range_desc = f"at least {min_value}"
        elif max_value is not None:
            range_desc = f"at most {max_value}"
        else:
            range_desc = "within acceptable range"

        message = f"Value {value} for '{field}' is out of range: must be {range_desc}"
        user_message = f"Value for {field} must be {range_desc}"

        super().__init__(
            message=message,
            field=field,
            value=value,
            error_code="VALUE_OUT_OF_RANGE_ERROR",
            user_message=user_message,
            details=details,
            **kwargs
        )


class InvalidChoiceError(ValidationException):
    """Exception for invalid choice from predefined options."""

    def __init__(
            self,
            field: str,
            value: Any,
            valid_choices: List[Any],
            **kwargs
    ):
        details = kwargs.pop("details", {})
        details.update({
            "field": field,
            "invalid_value": str(value),
            "valid_choices": [str(choice) for choice in valid_choices]
        })

        choices_str = ", ".join(str(choice) for choice in valid_choices)
        message = f"Invalid choice '{value}' for '{field}'. Valid choices: {choices_str}"
        user_message = f"Invalid value for {field}. Please choose from: {choices_str}"

        super().__init__(
            message=message,
            field=field,
            value=value,
            error_code="INVALID_CHOICE_ERROR",
            user_message=user_message,
            details=details,
            **kwargs
        )


class StringLengthError(ValidationException):
    """Exception for string length validation errors."""

    def __init__(
            self,
            field: str,
            value: str,
            min_length: Optional[int] = None,
            max_length: Optional[int] = None,
            **kwargs
    ):
        details = kwargs.pop("details", {})
        details.update({
            "field": field,
            "actual_length": len(value),
            "value_preview": value[:50] + "..." if len(value) > 50 else value
        })

        if min_length is not None:
            details["min_length"] = min_length
        if max_length is not None:
            details["max_length"] = max_length

        # Build length description
        actual_length = len(value)
        if min_length is not None and max_length is not None:
            length_desc = f"between {min_length} and {max_length} characters"
        elif min_length is not None:
            length_desc = f"at least {min_length} characters"
        elif max_length is not None:
            length_desc = f"at most {max_length} characters"
        else:
            length_desc = "appropriate length"

        message = f"String length {actual_length} for '{field}' is invalid: must be {length_desc}"
        user_message = f"Text for {field} must be {length_desc}"

        super().__init__(
            message=message,
            field=field,
            value=value,
            error_code="STRING_LENGTH_ERROR",
            user_message=user_message,
            details=details,
            **kwargs
        )


class EmailValidationError(ValidationException):
    """Exception for email format validation errors."""

    def __init__(
            self,
            email: str,
            reason: str = "Invalid email format",
            **kwargs
    ):
        details = kwargs.pop("details", {})
        details.update({
            "email": email,
            "reason": reason
        })

        message = f"Email validation failed: {reason}"
        user_message = "Please enter a valid email address"

        super().__init__(
            message=message,
            field="email",
            value=email,
            error_code="EMAIL_VALIDATION_ERROR",
            user_message=user_message,
            details=details,
            **kwargs
        )


class PhoneNumberValidationError(ValidationException):
    """Exception for phone number validation errors."""

    def __init__(
            self,
            phone: str,
            reason: str = "Invalid phone number format",
            country_code: Optional[str] = None,
            **kwargs
    ):
        details = kwargs.pop("details", {})
        details.update({
            "phone_number": phone,
            "reason": reason
        })

        if country_code:
            details["country_code"] = country_code

        message = f"Phone number validation failed: {reason}"
        user_message = "Please enter a valid phone number in international format (e.g., +1234567890)"

        super().__init__(
            message=message,
            field="phone",
            value=phone,
            error_code="PHONE_VALIDATION_ERROR",
            user_message=user_message,
            details=details,
            **kwargs
        )


class URLValidationError(ValidationException):
    """Exception for URL validation errors."""

    def __init__(
            self,
            url: str,
            reason: str = "Invalid URL format",
            **kwargs
    ):
        details = kwargs.pop("details", {})
        details.update({
            "url": url,
            "reason": reason
        })

        message = f"URL validation failed: {reason}"
        user_message = "Please enter a valid URL (e.g., https://example.com)"

        super().__init__(
            message=message,
            field="url",
            value=url,
            error_code="URL_VALIDATION_ERROR",
            user_message=user_message,
            details=details,
            **kwargs
        )


class JSONValidationError(ValidationException):
    """Exception for JSON format validation errors."""

    def __init__(
            self,
            field: str,
            json_string: str,
            parse_error: str,
            **kwargs
    ):
        details = kwargs.pop("details", {})
        details.update({
            "field": field,
            "json_preview": json_string[:100] + "..." if len(json_string) > 100 else json_string,
            "parse_error": parse_error
        })

        message = f"JSON validation failed for '{field}': {parse_error}"
        user_message = f"Invalid JSON format for {field}"

        super().__init__(
            message=message,
            field=field,
            value=json_string,
            error_code="JSON_VALIDATION_ERROR",
            user_message=user_message,
            details=details,
            **kwargs
        )


class DateTimeValidationError(ValidationException):
    """Exception for datetime validation errors."""

    def __init__(
            self,
            field: str,
            datetime_string: str,
            reason: str = "Invalid datetime format",
            expected_format: str = "ISO 8601",
            **kwargs
    ):
        details = kwargs.pop("details", {})
        details.update({
            "field": field,
            "datetime_string": datetime_string,
            "reason": reason,
            "expected_format": expected_format
        })

        message = f"DateTime validation failed for '{field}': {reason}"
        user_message = f"Invalid datetime format for {field}. Expected format: {expected_format}"

        super().__init__(
            message=message,
            field=field,
            value=datetime_string,
            error_code="DATETIME_VALIDATION_ERROR",
            user_message=user_message,
            details=details,
            **kwargs
        )


class UUIDValidationError(ValidationException):
    """Exception for UUID validation errors."""

    def __init__(
            self,
            field: str,
            uuid_string: str,
            **kwargs
    ):
        details = kwargs.pop("details", {})
        details.update({
            "field": field,
            "uuid_string": uuid_string
        })

        message = f"UUID validation failed for '{field}': invalid UUID format"
        user_message = f"Invalid UUID format for {field}"

        super().__init__(
            message=message,
            field=field,
            value=uuid_string,
            error_code="UUID_VALIDATION_ERROR",
            user_message=user_message,
            details=details,
            **kwargs
        )


class BusinessRuleValidationError(ValidationException):
    """Exception for business rule validation errors."""

    def __init__(
            self,
            rule: str,
            message: str,
            affected_fields: Optional[List[str]] = None,
            **kwargs
    ):
        details = kwargs.pop("details", {})
        details.update({
            "business_rule": rule,
            "rule_violation": message
        })

        if affected_fields:
            details["affected_fields"] = affected_fields

        user_message = f"Business rule violation: {message}"

        super().__init__(
            message=f"Business rule validation failed: {rule} - {message}",
            error_code="BUSINESS_RULE_VALIDATION_ERROR",
            user_message=user_message,
            details=details,
            **kwargs
        )


class FileValidationError(ValidationException):
    """Exception for file validation errors."""

    def __init__(
            self,
            filename: str,
            reason: str,
            file_size: Optional[int] = None,
            file_type: Optional[str] = None,
            max_size: Optional[int] = None,
            allowed_types: Optional[List[str]] = None,
            **kwargs
    ):
        details = kwargs.pop("details", {})
        details.update({
            "filename": filename,
            "reason": reason
        })

        if file_size is not None:
            details["file_size"] = file_size
        if file_type:
            details["file_type"] = file_type
        if max_size is not None:
            details["max_size"] = max_size
        if allowed_types:
            details["allowed_types"] = allowed_types

        message = f"File validation failed for '{filename}': {reason}"
        user_message = f"File validation error: {reason}"

        super().__init__(
            message=message,
            field="file",
            value=filename,
            error_code="FILE_VALIDATION_ERROR",
            user_message=user_message,
            details=details,
            **kwargs
        )


class ContentValidationError(ValidationException):
    """Exception for content validation errors (profanity, spam, etc.)."""

    def __init__(
            self,
            content: str,
            violation_type: str,
            confidence_score: Optional[float] = None,
            **kwargs
    ):
        details = kwargs.pop("details", {})
        details.update({
            "content_preview": content[:100] + "..." if len(content) > 100 else content,
            "violation_type": violation_type
        })

        if confidence_score is not None:
            details["confidence_score"] = confidence_score

        message = f"Content validation failed: {violation_type}"
        user_message = "Content contains inappropriate material and cannot be processed"

        super().__init__(
            message=message,
            field="content",
            value=content,
            error_code="CONTENT_VALIDATION_ERROR",
            user_message=user_message,
            details=details,
            **kwargs
        )


# Convenience functions for raising validation errors
def raise_required_field_error(field: str):
    """Raise RequiredFieldError for missing field."""
    raise RequiredFieldError(field=field)


def raise_invalid_format_error(
        field: str,
        value: Any,
        expected_format: str,
        example: str = None
):
    """Raise InvalidFormatError for format validation."""
    raise InvalidFormatError(
        field=field,
        value=value,
        expected_format=expected_format,
        example=example
    )


def raise_value_out_of_range_error(
        field: str,
        value: Union[int, float],
        min_value: Union[int, float] = None,
        max_value: Union[int, float] = None
):
    """Raise ValueOutOfRangeError for range validation."""
    raise ValueOutOfRangeError(
        field=field,
        value=value,
        min_value=min_value,
        max_value=max_value
    )


def raise_invalid_choice_error(field: str, value: Any, valid_choices: List[Any]):
    """Raise InvalidChoiceError for choice validation."""
    raise InvalidChoiceError(
        field=field,
        value=value,
        valid_choices=valid_choices
    )


def raise_string_length_error(
        field: str,
        value: str,
        min_length: int = None,
        max_length: int = None
):
    """Raise StringLengthError for length validation."""
    raise StringLengthError(
        field=field,
        value=value,
        min_length=min_length,
        max_length=max_length
    )


def raise_email_validation_error(email: str, reason: str = "Invalid email format"):
    """Raise EmailValidationError for email validation."""
    raise EmailValidationError(email=email, reason=reason)


def raise_phone_validation_error(
        phone: str,
        reason: str = "Invalid phone number format",
        country_code: str = None
):
    """Raise PhoneNumberValidationError for phone validation."""
    raise PhoneNumberValidationError(
        phone=phone,
        reason=reason,
        country_code=country_code
    )


def raise_business_rule_error(rule: str, message: str, affected_fields: List[str] = None):
    """Raise BusinessRuleValidationError for business rule violations."""
    raise BusinessRuleValidationError(
        rule=rule,
        message=message,
        affected_fields=affected_fields
    )


# Export all validation exception classes and utilities
__all__ = [
    # Base validation exceptions
    'ValidationException',
    'FieldValidationError',
    'MultipleFieldValidationError',
    'RequiredFieldError',

    # Format validation exceptions
    'InvalidFormatError',
    'ValueOutOfRangeError',
    'InvalidChoiceError',
    'StringLengthError',

    # Specific format validations
    'EmailValidationError',
    'PhoneNumberValidationError',
    'URLValidationError',
    'JSONValidationError',
    'DateTimeValidationError',
    'UUIDValidationError',

    # Business and content validation
    'BusinessRuleValidationError',
    'FileValidationError',
    'ContentValidationError',

    # Convenience functions
    'raise_required_field_error',
    'raise_invalid_format_error',
    'raise_value_out_of_range_error',
    'raise_invalid_choice_error',
    'raise_string_length_error',
    'raise_email_validation_error',
    'raise_phone_validation_error',
    'raise_business_rule_error',
]