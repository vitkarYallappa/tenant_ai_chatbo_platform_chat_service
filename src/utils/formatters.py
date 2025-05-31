"""
Data formatting and transformation utilities.

This module provides various formatting functions for different data types,
message content, and response structures used throughout the Chat Service.
"""

import json
import html
import re
from typing import Any, Dict, List, Optional, Union
from datetime import datetime, timezone
from decimal import Decimal, ROUND_HALF_UP
from urllib.parse import quote, unquote

import structlog
from src.config.constants import SUPPORTED_LANGUAGES, CHANNEL_CONFIG
from src.utils.logger import get_logger

# Initialize logger
logger = get_logger(__name__)


class MessageFormatter:
    """
    Message content formatting for different channels.

    Handles channel-specific formatting requirements and
    content adaptation for various communication platforms.
    """

    @staticmethod
    def format_text_message(
            text: str,
            channel: str,
            max_length: Optional[int] = None,
            escape_html: bool = True
    ) -> str:
        """
        Format text message for specific channel.

        Args:
            text: Raw text content
            channel: Target channel
            max_length: Optional maximum length override
            escape_html: Whether to escape HTML characters

        Returns:
            Formatted text suitable for the channel
        """
        if not text:
            return ""

        # Get channel configuration
        config = CHANNEL_CONFIG.get(channel, {})
        max_len = max_length or config.get("max_message_length", 4096)

        # Escape HTML if needed
        if escape_html:
            text = html.escape(text)

        # Truncate if necessary
        if len(text) > max_len:
            text = text[:max_len - 3] + "..."

        # Channel-specific formatting
        if channel == "slack":
            # Convert markdown-style formatting to Slack format
            text = MessageFormatter._format_slack_text(text)
        elif channel == "teams":
            # Convert to Teams-compatible format
            text = MessageFormatter._format_teams_text(text)
        elif channel == "whatsapp":
            # WhatsApp formatting
            text = MessageFormatter._format_whatsapp_text(text)

        return text

    @staticmethod
    def _format_slack_text(text: str) -> str:
        """Format text for Slack channel."""
        # Convert **bold** to *bold*
        text = re.sub(r'\*\*(.*?)\*\*', r'*\1*', text)
        # Convert __italic__ to _italic_
        text = re.sub(r'__(.*?)__', r'_\1_', text)
        # Convert `code` to `code`
        text = re.sub(r'`([^`]+)`', r'`\1`', text)
        return text

    @staticmethod
    def _format_teams_text(text: str) -> str:
        """Format text for Microsoft Teams channel."""
        # Convert **bold** to **bold**
        text = re.sub(r'\*\*(.*?)\*\*', r'**\1**', text)
        # Convert *italic* to *italic*
        text = re.sub(r'\*(.*?)\*', r'*\1*', text)
        return text

    @staticmethod
    def _format_whatsapp_text(text: str) -> str:
        """Format text for WhatsApp channel."""
        # WhatsApp supports basic markdown
        # Convert **bold** to *bold*
        text = re.sub(r'\*\*(.*?)\*\*', r'*\1*', text)
        # Convert __italic__ to _italic_
        text = re.sub(r'__(.*?)__', r'_\1_', text)
        # Convert ~~strikethrough~~ to ~strikethrough~
        text = re.sub(r'~~(.*?)~~', r'~\1~', text)
        return text

    @staticmethod
    def format_quick_replies(
            quick_replies: List[Dict[str, Any]],
            channel: str
    ) -> List[Dict[str, Any]]:
        """
        Format quick replies for specific channel.

        Args:
            quick_replies: List of quick reply objects
            channel: Target channel

        Returns:
            Formatted quick replies suitable for the channel
        """
        if not quick_replies:
            return []

        config = CHANNEL_CONFIG.get(channel, {})
        max_replies = config.get("max_quick_replies", len(quick_replies))

        # Truncate if necessary
        formatted_replies = quick_replies[:max_replies]

        # Channel-specific formatting
        for reply in formatted_replies:
            if "title" in reply:
                # Truncate title if needed
                max_title_length = 20  # Most channels have this limit
                if len(reply["title"]) > max_title_length:
                    reply["title"] = reply["title"][:max_title_length - 3] + "..."

        return formatted_replies

    @staticmethod
    def format_carousel(
            carousel_items: List[Dict[str, Any]],
            channel: str
    ) -> List[Dict[str, Any]]:
        """
        Format carousel items for specific channel.

        Args:
            carousel_items: List of carousel item objects
            channel: Target channel

        Returns:
            Formatted carousel items suitable for the channel
        """
        if not carousel_items:
            return []

        config = CHANNEL_CONFIG.get(channel, {})

        # Check if channel supports carousels
        if not config.get("supports_carousels", False):
            logger.warning(f"Channel {channel} does not support carousels")
            return []

        # Format each carousel item
        formatted_items = []
        for item in carousel_items:
            formatted_item = item.copy()

            # Format title and subtitle
            if "title" in formatted_item:
                formatted_item["title"] = MessageFormatter.format_text_message(
                    formatted_item["title"],
                    channel,
                    max_length=80
                )

            if "subtitle" in formatted_item:
                formatted_item["subtitle"] = MessageFormatter.format_text_message(
                    formatted_item["subtitle"],
                    channel,
                    max_length=80
                )

            # Format buttons
            if "buttons" in formatted_item:
                max_buttons = config.get("max_buttons", 3)
                formatted_item["buttons"] = formatted_item["buttons"][:max_buttons]

            formatted_items.append(formatted_item)

        return formatted_items


class ResponseFormatter:
    """
    Response formatting utilities for API responses.

    Provides consistent formatting for various response types
    and data structures returned by the Chat Service.
    """

    @staticmethod
    def format_success_response(
            data: Any,
            message: Optional[str] = None,
            meta: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Format successful API response.

        Args:
            data: Response data
            message: Optional success message
            meta: Optional metadata

        Returns:
            Formatted success response
        """
        response = {
            "status": "success",
            "data": data
        }

        if message:
            response["message"] = message

        if meta:
            response["meta"] = meta
        else:
            response["meta"] = {
                "timestamp": datetime.now(timezone.utc).isoformat()
            }

        return response

    @staticmethod
    def format_error_response(
            error_code: str,
            error_message: str,
            details: Optional[Dict[str, Any]] = None,
            status_code: int = 400
    ) -> Dict[str, Any]:
        """
        Format error API response.

        Args:
            error_code: Machine-readable error code
            error_message: Human-readable error message
            details: Optional error details
            status_code: HTTP status code

        Returns:
            Formatted error response
        """
        response = {
            "status": "error",
            "error": {
                "code": error_code,
                "message": error_message,
                "timestamp": datetime.now(timezone.utc).isoformat()
            }
        }

        if details:
            response["error"]["details"] = details

        return response

    @staticmethod
    def format_paginated_response(
            items: List[Any],
            page: int,
            page_size: int,
            total_count: int,
            meta: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Format paginated API response.

        Args:
            items: List of items for current page
            page: Current page number (1-based)
            page_size: Number of items per page
            total_count: Total number of items
            meta: Optional additional metadata

        Returns:
            Formatted paginated response
        """
        total_pages = (total_count + page_size - 1) // page_size

        pagination_meta = {
            "pagination": {
                "page": page,
                "page_size": page_size,
                "total_count": total_count,
                "total_pages": total_pages,
                "has_next": page < total_pages,
                "has_previous": page > 1
            },
            "timestamp": datetime.now(timezone.utc).isoformat()
        }

        if meta:
            pagination_meta.update(meta)

        return ResponseFormatter.format_success_response(
            data=items,
            meta=pagination_meta
        )


class DataFormatter:
    """
    General data formatting utilities.

    Provides formatting functions for various data types
    including numbers, dates, strings, and complex objects.
    """

    @staticmethod
    def format_currency(
            amount: Union[int, float, Decimal],
            currency: str = "USD",
            decimal_places: int = 2
    ) -> str:
        """
        Format currency amount.

        Args:
            amount: Amount to format
            currency: Currency code
            decimal_places: Number of decimal places

        Returns:
            Formatted currency string
        """
        if isinstance(amount, (int, float)):
            amount = Decimal(str(amount))

        # Round to specified decimal places
        quantizer = Decimal('0.1') ** decimal_places
        rounded_amount = amount.quantize(quantizer, rounding=ROUND_HALF_UP)

        # Format with currency symbol
        currency_symbols = {
            "USD": "$",
            "EUR": "€",
            "GBP": "£",
            "JPY": "¥",
            "CNY": "¥",
            "INR": "₹"
        }

        symbol = currency_symbols.get(currency, currency)

        if currency == "JPY":
            # Japanese Yen typically has no decimal places
            return f"{symbol}{int(rounded_amount):,}"
        else:
            return f"{symbol}{rounded_amount:,.{decimal_places}f}"

    @staticmethod
    def format_file_size(size_bytes: int) -> str:
        """
        Format file size in human-readable format.

        Args:
            size_bytes: Size in bytes

        Returns:
            Formatted file size string
        """
        if size_bytes == 0:
            return "0 B"

        units = ["B", "KB", "MB", "GB", "TB"]
        unit_index = 0
        size = float(size_bytes)

        while size >= 1024 and unit_index < len(units) - 1:
            size /= 1024
            unit_index += 1

        if unit_index == 0:
            return f"{int(size)} {units[unit_index]}"
        else:
            return f"{size:.1f} {units[unit_index]}"

    @staticmethod
    def format_duration(seconds: Union[int, float]) -> str:
        """
        Format duration in human-readable format.

        Args:
            seconds: Duration in seconds

        Returns:
            Formatted duration string
        """
        if seconds < 60:
            return f"{int(seconds)}s"
        elif seconds < 3600:
            minutes = int(seconds // 60)
            remaining_seconds = int(seconds % 60)
            if remaining_seconds > 0:
                return f"{minutes}m {remaining_seconds}s"
            else:
                return f"{minutes}m"
        else:
            hours = int(seconds // 3600)
            remaining_minutes = int((seconds % 3600) // 60)
            if remaining_minutes > 0:
                return f"{hours}h {remaining_minutes}m"
            else:
                return f"{hours}h"

    @staticmethod
    def format_phone_number(
            phone: str,
            format_type: str = "international"
    ) -> str:
        """
        Format phone number for display.

        Args:
            phone: Phone number to format
            format_type: Format type ("international", "national", "e164")

        Returns:
            Formatted phone number
        """
        try:
            import phonenumbers

            # Parse the phone number
            parsed = phonenumbers.parse(phone, None)

            # Format based on type
            if format_type == "international":
                return phonenumbers.format_number(
                    parsed,
                    phonenumbers.PhoneNumberFormat.INTERNATIONAL
                )
            elif format_type == "national":
                return phonenumbers.format_number(
                    parsed,
                    phonenumbers.PhoneNumberFormat.NATIONAL
                )
            elif format_type == "e164":
                return phonenumbers.format_number(
                    parsed,
                    phonenumbers.PhoneNumberFormat.E164
                )
            else:
                return phone

        except Exception:
            # Return original if formatting fails
            return phone

    @staticmethod
    def format_datetime(
            dt: datetime,
            format_type: str = "iso",
            timezone_name: Optional[str] = None
    ) -> str:
        """
        Format datetime for display.

        Args:
            dt: Datetime to format
            format_type: Format type ("iso", "human", "short", "date_only")
            timezone_name: Optional timezone for conversion

        Returns:
            Formatted datetime string
        """
        # Convert timezone if specified
        if timezone_name:
            try:
                import pytz
                tz = pytz.timezone(timezone_name)
                dt = dt.astimezone(tz)
            except Exception:
                pass  # Use original timezone

        if format_type == "iso":
            return dt.isoformat()
        elif format_type == "human":
            return dt.strftime("%B %d, %Y at %I:%M %p")
        elif format_type == "short":
            return dt.strftime("%m/%d/%Y %H:%M")
        elif format_type == "date_only":
            return dt.strftime("%Y-%m-%d")
        else:
            return dt.isoformat()

    @staticmethod
    def sanitize_filename(filename: str) -> str:
        """
        Sanitize filename for safe storage.

        Args:
            filename: Original filename

        Returns:
            Sanitized filename
        """
        if not filename:
            return "untitled"

        # Remove path separators and other unsafe characters
        unsafe_chars = r'[<>:"/\\|?*\x00-\x1f]'
        sanitized = re.sub(unsafe_chars, '_', filename)

        # Remove leading/trailing dots and spaces
        sanitized = sanitized.strip('. ')

        # Ensure it's not empty
        if not sanitized:
            sanitized = "untitled"

        # Limit length
        if len(sanitized) > 255:
            name, ext = sanitized.rsplit('.', 1) if '.' in sanitized else (sanitized, '')
            max_name_length = 255 - len(ext) - 1 if ext else 255
            sanitized = name[:max_name_length] + (f'.{ext}' if ext else '')

        return sanitized


class TemplateFormatter:
    """
    Template formatting utilities for dynamic content.

    Provides template rendering and variable substitution
    for messages, emails, and other dynamic content.
    """

    @staticmethod
    def format_template(
            template: str,
            variables: Dict[str, Any],
            escape_html: bool = False
    ) -> str:
        """
        Format template with variable substitution.

        Args:
            template: Template string with {variable} placeholders
            variables: Dictionary of variables to substitute
            escape_html: Whether to escape HTML in variables

        Returns:
            Formatted template string
        """
        if not template:
            return ""

        # Prepare variables for substitution
        formatted_vars = {}
        for key, value in variables.items():
            if value is None:
                formatted_value = ""
            elif isinstance(value, (int, float)):
                formatted_value = str(value)
            elif isinstance(value, datetime):
                formatted_value = DataFormatter.format_datetime(value, "human")
            elif isinstance(value, bool):
                formatted_value = "yes" if value else "no"
            else:
                formatted_value = str(value)

            # Escape HTML if needed
            if escape_html:
                formatted_value = html.escape(formatted_value)

            formatted_vars[key] = formatted_value

        try:
            # Use safe string formatting
            return template.format(**formatted_vars)
        except KeyError as e:
            logger.warning(f"Missing template variable: {e}")
            return template
        except Exception as e:
            logger.error(f"Template formatting error: {e}")
            return template

    @staticmethod
    def extract_template_variables(template: str) -> List[str]:
        """
        Extract variable names from template.

        Args:
            template: Template string with {variable} placeholders

        Returns:
            List of variable names found in template
        """
        import re

        # Find all {variable} patterns
        pattern = r'\{([^}]+)\}'
        matches = re.findall(pattern, template)

        # Remove duplicates and return
        return list(set(matches))


# Convenience functions for common formatting operations
def format_message_for_channel(
        content: Dict[str, Any],
        channel: str
) -> Dict[str, Any]:
    """
    Format complete message content for specific channel.

    Args:
        content: Message content dictionary
        channel: Target channel

    Returns:
        Formatted message content
    """
    formatted_content = content.copy()

    # Format text if present
    if "text" in formatted_content:
        formatted_content["text"] = MessageFormatter.format_text_message(
            formatted_content["text"],
            channel
        )

    # Format quick replies if present
    if "quick_replies" in formatted_content:
        formatted_content["quick_replies"] = MessageFormatter.format_quick_replies(
            formatted_content["quick_replies"],
            channel
        )

    # Format carousel if present
    if "carousel" in formatted_content:
        formatted_content["carousel"] = MessageFormatter.format_carousel(
            formatted_content["carousel"],
            channel
        )

    return formatted_content


def format_api_response(
        data: Any = None,
        error: Optional[str] = None,
        meta: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Format standard API response.

    Args:
        data: Response data (for success)
        error: Error message (for error)
        meta: Optional metadata

    Returns:
        Formatted API response
    """
    if error:
        return ResponseFormatter.format_error_response(
            error_code="GENERIC_ERROR",
            error_message=error
        )
    else:
        return ResponseFormatter.format_success_response(
            data=data,
            meta=meta
        )


# Export commonly used classes and functions
__all__ = [
    'MessageFormatter',
    'ResponseFormatter',
    'DataFormatter',
    'TemplateFormatter',
    'format_message_for_channel',
    'format_api_response',
]