"""
Date and time utilities for Chat Service.

This module provides comprehensive date/time handling utilities including
timezone management, formatting, parsing, and business date calculations.
"""

import calendar
from datetime import datetime, timezone, timedelta, date, time
from typing import Optional, Union, List, Tuple
from zoneinfo import ZoneInfo

import structlog
from src.utils.logger import get_logger

# Initialize logger
logger = get_logger(__name__)

# Common timezone objects
UTC = timezone.utc
EASTERN = ZoneInfo("America/New_York")
PACIFIC = ZoneInfo("America/Los_Angeles")
CENTRAL = ZoneInfo("America/Chicago")
MOUNTAIN = ZoneInfo("America/Denver")

# Business hours and holidays configuration
DEFAULT_BUSINESS_HOURS = {
    "start": time(9, 0),  # 9:00 AM
    "end": time(17, 0),  # 5:00 PM
    "timezone": "UTC"
}

DEFAULT_BUSINESS_DAYS = [0, 1, 2, 3, 4]  # Monday to Friday (0=Monday)


class DateTimeUtils:
    """
    Comprehensive date and time utilities.

    Provides various date/time operations including parsing,
    formatting, timezone conversions, and business logic.
    """

    @staticmethod
    def now_utc() -> datetime:
        """
        Get current UTC datetime.

        Returns:
            Current UTC datetime
        """
        return datetime.now(UTC)

    @staticmethod
    def now_in_timezone(tz: Union[str, ZoneInfo]) -> datetime:
        """
        Get current datetime in specified timezone.

        Args:
            tz: Timezone name or ZoneInfo object

        Returns:
            Current datetime in specified timezone
        """
        if isinstance(tz, str):
            tz = ZoneInfo(tz)

        return datetime.now(tz)

    @staticmethod
    def parse_datetime(
            dt_string: str,
            default_timezone: Optional[Union[str, ZoneInfo]] = None
    ) -> datetime:
        """
        Parse datetime string with timezone handling.

        Args:
            dt_string: DateTime string to parse
            default_timezone: Default timezone if none specified

        Returns:
            Parsed datetime object

        Raises:
            ValueError: If datetime string is invalid
        """
        if not dt_string:
            raise ValueError("DateTime string is required")

        # Common datetime formats to try
        formats = [
            "%Y-%m-%dT%H:%M:%S.%fZ",  # ISO format with microseconds UTC
            "%Y-%m-%dT%H:%M:%SZ",  # ISO format UTC
            "%Y-%m-%dT%H:%M:%S.%f%z",  # ISO format with microseconds and timezone
            "%Y-%m-%dT%H:%M:%S%z",  # ISO format with timezone
            "%Y-%m-%dT%H:%M:%S.%f",  # ISO format with microseconds
            "%Y-%m-%dT%H:%M:%S",  # ISO format
            "%Y-%m-%d %H:%M:%S.%f",  # Space separated with microseconds
            "%Y-%m-%d %H:%M:%S",  # Space separated
            "%Y-%m-%d",  # Date only
            "%m/%d/%Y %H:%M:%S",  # US format with time
            "%m/%d/%Y",  # US format date only
            "%d/%m/%Y %H:%M:%S",  # European format with time
            "%d/%m/%Y",  # European format date only
        ]

        # Try parsing with different formats
        parsed_dt = None
        for fmt in formats:
            try:
                parsed_dt = datetime.strptime(dt_string, fmt)
                break
            except ValueError:
                continue

        if parsed_dt is None:
            # Try fromisoformat as a last resort
            try:
                # Handle 'Z' suffix for UTC
                if dt_string.endswith('Z'):
                    dt_string = dt_string[:-1] + '+00:00'
                parsed_dt = datetime.fromisoformat(dt_string)
            except ValueError:
                raise ValueError(f"Unable to parse datetime string: {dt_string}")

        # Add timezone if missing
        if parsed_dt.tzinfo is None:
            if default_timezone:
                if isinstance(default_timezone, str):
                    default_timezone = ZoneInfo(default_timezone)
                parsed_dt = parsed_dt.replace(tzinfo=default_timezone)
            else:
                parsed_dt = parsed_dt.replace(tzinfo=UTC)

        return parsed_dt

    @staticmethod
    def convert_timezone(
            dt: datetime,
            target_tz: Union[str, ZoneInfo]
    ) -> datetime:
        """
        Convert datetime to target timezone.

        Args:
            dt: Datetime to convert
            target_tz: Target timezone

        Returns:
            Datetime in target timezone
        """
        if isinstance(target_tz, str):
            target_tz = ZoneInfo(target_tz)

        # Ensure datetime has timezone info
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=UTC)

        return dt.astimezone(target_tz)

    @staticmethod
    def format_datetime(
            dt: datetime,
            format_type: str = "iso",
            timezone_name: Optional[str] = None
    ) -> str:
        """
        Format datetime for display or storage.

        Args:
            dt: Datetime to format
            format_type: Format type (iso, human, short, timestamp)
            timezone_name: Optional timezone for conversion

        Returns:
            Formatted datetime string
        """
        # Convert timezone if specified
        if timezone_name:
            dt = DateTimeUtils.convert_timezone(dt, timezone_name)

        format_map = {
            "iso": "%Y-%m-%dT%H:%M:%S.%f%z",
            "iso_simple": "%Y-%m-%dT%H:%M:%S%z",
            "human": "%B %d, %Y at %I:%M %p %Z",
            "short": "%m/%d/%Y %H:%M",
            "date_only": "%Y-%m-%d",
            "time_only": "%H:%M:%S",
            "timestamp": "%Y%m%d_%H%M%S",
            "log": "%Y-%m-%d %H:%M:%S %Z",
            "api": "%Y-%m-%dT%H:%M:%S.%fZ" if dt.tzinfo == UTC else "%Y-%m-%dT%H:%M:%S.%f%z"
        }

        format_string = format_map.get(format_type, format_map["iso"])

        if format_type == "api" and dt.tzinfo == UTC:
            # Special handling for API format with UTC
            return dt.strftime("%Y-%m-%dT%H:%M:%S.%fZ")

        return dt.strftime(format_string)

    @staticmethod
    def get_age_from_date(birth_date: date) -> int:
        """
        Calculate age from birth date.

        Args:
            birth_date: Date of birth

        Returns:
            Age in years
        """
        today = date.today()
        age = today.year - birth_date.year

        # Adjust if birthday hasn't occurred this year
        if today.month < birth_date.month or \
                (today.month == birth_date.month and today.day < birth_date.day):
            age -= 1

        return age

    @staticmethod
    def get_time_until(target_dt: datetime) -> timedelta:
        """
        Get time remaining until target datetime.

        Args:
            target_dt: Target datetime

        Returns:
            Time difference as timedelta
        """
        now = DateTimeUtils.now_utc()

        # Ensure target is timezone-aware
        if target_dt.tzinfo is None:
            target_dt = target_dt.replace(tzinfo=UTC)

        # Convert to UTC for comparison
        target_utc = target_dt.astimezone(UTC)

        return target_utc - now

    @staticmethod
    def get_time_since(past_dt: datetime) -> timedelta:
        """
        Get time elapsed since past datetime.

        Args:
            past_dt: Past datetime

        Returns:
            Time difference as timedelta
        """
        now = DateTimeUtils.now_utc()

        # Ensure past datetime is timezone-aware
        if past_dt.tzinfo is None:
            past_dt = past_dt.replace(tzinfo=UTC)

        # Convert to UTC for comparison
        past_utc = past_dt.astimezone(UTC)

        return now - past_utc


class BusinessDateUtils:
    """
    Business-specific date and time utilities.

    Handles business hours, working days, holidays,
    and other business-related date calculations.
    """

    @staticmethod
    def is_business_day(
            dt: datetime,
            business_days: Optional[List[int]] = None
    ) -> bool:
        """
        Check if datetime falls on a business day.

        Args:
            dt: Datetime to check
            business_days: List of business weekdays (0=Monday)

        Returns:
            True if it's a business day
        """
        if business_days is None:
            business_days = DEFAULT_BUSINESS_DAYS

        return dt.weekday() in business_days

    @staticmethod
    def is_business_hours(
            dt: datetime,
            business_hours: Optional[dict] = None,
            timezone_name: Optional[str] = None
    ) -> bool:
        """
        Check if datetime falls within business hours.

        Args:
            dt: Datetime to check
            business_hours: Business hours configuration
            timezone_name: Timezone for business hours

        Returns:
            True if within business hours
        """
        if business_hours is None:
            business_hours = DEFAULT_BUSINESS_HOURS

        # Convert to business timezone
        if timezone_name:
            dt = DateTimeUtils.convert_timezone(dt, timezone_name)
        elif business_hours.get("timezone"):
            dt = DateTimeUtils.convert_timezone(dt, business_hours["timezone"])

        # Check if it's a business day
        if not BusinessDateUtils.is_business_day(dt):
            return False

        # Check time range
        current_time = dt.time()
        start_time = business_hours.get("start", DEFAULT_BUSINESS_HOURS["start"])
        end_time = business_hours.get("end", DEFAULT_BUSINESS_HOURS["end"])

        return start_time <= current_time <= end_time

    @staticmethod
    def get_next_business_day(
            from_date: datetime,
            business_days: Optional[List[int]] = None
    ) -> datetime:
        """
        Get the next business day from given date.

        Args:
            from_date: Starting date
            business_days: List of business weekdays

        Returns:
            Next business day datetime
        """
        if business_days is None:
            business_days = DEFAULT_BUSINESS_DAYS

        current_date = from_date.replace(hour=0, minute=0, second=0, microsecond=0)
        current_date += timedelta(days=1)  # Start from next day

        while current_date.weekday() not in business_days:
            current_date += timedelta(days=1)

        return current_date

    @staticmethod
    def get_business_days_between(
            start_date: datetime,
            end_date: datetime,
            business_days: Optional[List[int]] = None
    ) -> int:
        """
        Count business days between two dates.

        Args:
            start_date: Start date
            end_date: End date
            business_days: List of business weekdays

        Returns:
            Number of business days
        """
        if business_days is None:
            business_days = DEFAULT_BUSINESS_DAYS

        if start_date > end_date:
            start_date, end_date = end_date, start_date

        count = 0
        current_date = start_date.date()
        end_date = end_date.date()

        while current_date <= end_date:
            if current_date.weekday() in business_days:
                count += 1
            current_date += timedelta(days=1)

        return count

    @staticmethod
    def add_business_days(
            start_date: datetime,
            business_days_to_add: int,
            business_days: Optional[List[int]] = None
    ) -> datetime:
        """
        Add business days to a date.

        Args:
            start_date: Starting date
            business_days_to_add: Number of business days to add
            business_days: List of business weekdays

        Returns:
            Resulting datetime
        """
        if business_days is None:
            business_days = DEFAULT_BUSINESS_DAYS

        current_date = start_date
        days_added = 0

        while days_added < business_days_to_add:
            current_date += timedelta(days=1)
            if current_date.weekday() in business_days:
                days_added += 1

        return current_date


class TimeRangeUtils:
    """
    Time range and interval utilities.

    Provides utilities for working with time ranges,
    intervals, and duration calculations.
    """

    @staticmethod
    def create_time_range(
            start: datetime,
            end: datetime
    ) -> Tuple[datetime, datetime]:
        """
        Create a validated time range.

        Args:
            start: Start datetime
            end: End datetime

        Returns:
            Tuple of (start, end) datetimes

        Raises:
            ValueError: If end is before start
        """
        if end < start:
            raise ValueError("End time must be after start time")

        return (start, end)

    @staticmethod
    def is_datetime_in_range(
            dt: datetime,
            start: datetime,
            end: datetime
    ) -> bool:
        """
        Check if datetime falls within time range.

        Args:
            dt: Datetime to check
            start: Range start
            end: Range end

        Returns:
            True if datetime is in range
        """
        return start <= dt <= end

    @staticmethod
    def get_overlapping_duration(
            range1_start: datetime,
            range1_end: datetime,
            range2_start: datetime,
            range2_end: datetime
    ) -> timedelta:
        """
        Get overlapping duration between two time ranges.

        Args:
            range1_start: First range start
            range1_end: First range end
            range2_start: Second range start
            range2_end: Second range end

        Returns:
            Overlapping duration
        """
        # Find the overlap
        overlap_start = max(range1_start, range2_start)
        overlap_end = min(range1_end, range2_end)

        # If there's no overlap, return zero duration
        if overlap_start >= overlap_end:
            return timedelta(0)

        return overlap_end - overlap_start

    @staticmethod
    def split_time_range_by_days(
            start: datetime,
            end: datetime
    ) -> List[Tuple[datetime, datetime]]:
        """
        Split time range into daily chunks.

        Args:
            start: Range start
            end: Range end

        Returns:
            List of daily time range tuples
        """
        ranges = []
        current_start = start

        while current_start < end:
            # Get end of current day
            day_end = current_start.replace(
                hour=23, minute=59, second=59, microsecond=999999
            )
            current_end = min(day_end, end)

            ranges.append((current_start, current_end))

            # Move to start of next day
            current_start = (current_start + timedelta(days=1)).replace(
                hour=0, minute=0, second=0, microsecond=0
            )

        return ranges


class RelativeTimeUtils:
    """
    Relative time formatting utilities.

    Provides human-readable relative time descriptions
    like "2 minutes ago" or "in 3 hours".
    """

    @staticmethod
    def humanize_timedelta(delta: timedelta) -> str:
        """
        Convert timedelta to human-readable format.

        Args:
            delta: Time difference

        Returns:
            Human-readable time string
        """
        total_seconds = int(delta.total_seconds())

        if total_seconds == 0:
            return "just now"

        future = total_seconds > 0
        total_seconds = abs(total_seconds)

        # Define time units
        units = [
            (31536000, "year"),  # 365 days
            (2592000, "month"),  # 30 days
            (604800, "week"),  # 7 days
            (86400, "day"),  # 24 hours
            (3600, "hour"),  # 60 minutes
            (60, "minute"),  # 60 seconds
            (1, "second")
        ]

        for seconds_in_unit, unit_name in units:
            if total_seconds >= seconds_in_unit:
                unit_count = total_seconds // seconds_in_unit
                unit_text = unit_name + ("s" if unit_count != 1 else "")

                if future:
                    return f"in {unit_count} {unit_text}"
                else:
                    return f"{unit_count} {unit_text} ago"

        return "just now"

    @staticmethod
    def time_ago(dt: datetime) -> str:
        """
        Get "time ago" string for datetime.

        Args:
            dt: Past datetime

        Returns:
            Human-readable "time ago" string
        """
        delta = DateTimeUtils.get_time_since(dt)
        return RelativeTimeUtils.humanize_timedelta(-delta)

    @staticmethod
    def time_until(dt: datetime) -> str:
        """
        Get "time until" string for datetime.

        Args:
            dt: Future datetime

        Returns:
            Human-readable "time until" string
        """
        delta = DateTimeUtils.get_time_until(dt)
        return RelativeTimeUtils.humanize_timedelta(delta)


# Convenience functions for common operations
def utc_now() -> datetime:
    """Get current UTC datetime."""
    return DateTimeUtils.now_utc()


def parse_iso_datetime(iso_string: str) -> datetime:
    """Parse ISO datetime string."""
    return DateTimeUtils.parse_datetime(iso_string)


def format_for_api(dt: datetime) -> str:
    """Format datetime for API response."""
    return DateTimeUtils.format_datetime(dt, "api")


def is_recent(dt: datetime, max_age_minutes: int = 60) -> bool:
    """Check if datetime is recent (within max_age_minutes)."""
    delta = DateTimeUtils.get_time_since(dt)
    return delta.total_seconds() <= (max_age_minutes * 60)


def get_start_of_day(dt: datetime) -> datetime:
    """Get start of day (00:00:00) for given datetime."""
    return dt.replace(hour=0, minute=0, second=0, microsecond=0)


def get_end_of_day(dt: datetime) -> datetime:
    """Get end of day (23:59:59.999999) for given datetime."""
    return dt.replace(hour=23, minute=59, second=59, microsecond=999999)


def get_date_range_for_month(year: int, month: int) -> Tuple[datetime, datetime]:
    """
    Get date range for entire month.

    Args:
        year: Year
        month: Month (1-12)

    Returns:
        Tuple of (start_of_month, end_of_month)
    """
    start_of_month = datetime(year, month, 1, tzinfo=UTC)

    # Get last day of month
    last_day = calendar.monthrange(year, month)[1]
    end_of_month = datetime(year, month, last_day, 23, 59, 59, 999999, tzinfo=UTC)

    return (start_of_month, end_of_month)


# Export commonly used classes and functions
__all__ = [
    'DateTimeUtils',
    'BusinessDateUtils',
    'TimeRangeUtils',
    'RelativeTimeUtils',
    'UTC',
    'EASTERN',
    'PACIFIC',
    'CENTRAL',
    'MOUNTAIN',
    'utc_now',
    'parse_iso_datetime',
    'format_for_api',
    'is_recent',
    'get_start_of_day',
    'get_end_of_day',
    'get_date_range_for_month',
]