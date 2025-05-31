"""
Logging configuration and utilities using structlog.

This module provides centralized logging setup with structured logging,
context management, and performance monitoring capabilities.
"""

import logging
import logging.config
import sys
import time
from functools import wraps
from typing import Any, Dict, Optional, Union, Callable
from datetime import datetime, timezone

import structlog
from structlog.types import Processor


def timestamper(logger: Any, name: str, event_dict: Dict[str, Any]) -> Dict[str, Any]:
    """Add ISO timestamp to log events."""
    event_dict["timestamp"] = datetime.now(timezone.utc).isoformat()
    return event_dict


def add_severity(logger: Any, name: str, event_dict: Dict[str, Any]) -> Dict[str, Any]:
    """Add severity level for structured logging."""
    event_dict["severity"] = event_dict.get("level", "").upper()
    return event_dict


def add_service_context(logger: Any, name: str, event_dict: Dict[str, Any]) -> Dict[str, Any]:
    """Add service context information."""
    from src.config.constants import SERVICE_NAME, SERVICE_VERSION

    event_dict.update({
        "service": SERVICE_NAME,
        "version": SERVICE_VERSION,
    })
    return event_dict


def filter_sensitive_data(logger: Any, name: str, event_dict: Dict[str, Any]) -> Dict[str, Any]:
    """Filter sensitive information from logs."""
    sensitive_keys = {
        "password", "secret", "token", "key", "authorization",
        "auth", "credential", "api_key", "jwt", "session_id"
    }

    def _filter_dict(data: Dict[str, Any]) -> Dict[str, Any]:
        """Recursively filter sensitive data from nested dictionaries."""
        filtered = {}
        for key, value in data.items():
            key_lower = key.lower()

            # Check if key contains sensitive information
            if any(sensitive in key_lower for sensitive in sensitive_keys):
                filtered[key] = "[REDACTED]"
            elif isinstance(value, dict):
                filtered[key] = _filter_dict(value)
            elif isinstance(value, list):
                filtered[key] = [
                    _filter_dict(item) if isinstance(item, dict) else item
                    for item in value
                ]
            else:
                filtered[key] = value

        return filtered

    # Filter the event dictionary
    for key, value in list(event_dict.items()):
        if isinstance(value, dict):
            event_dict[key] = _filter_dict(value)

    return event_dict


def performance_logger(threshold_ms: float = 1000.0) -> Callable:
    """
    Decorator to log function performance.

    Args:
        threshold_ms: Log warning if execution time exceeds this threshold

    Returns:
        Decorated function with performance logging
    """

    def decorator(func: Callable) -> Callable:
        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            logger = get_logger(func.__module__)
            start_time = time.time()

            try:
                result = await func(*args, **kwargs)
                execution_time = (time.time() - start_time) * 1000

                log_data = {
                    "function": func.__name__,
                    "execution_time_ms": round(execution_time, 2),
                    "args_count": len(args),
                    "kwargs_count": len(kwargs),
                }

                if execution_time > threshold_ms:
                    logger.warning("Slow function execution", **log_data)
                else:
                    logger.debug("Function execution completed", **log_data)

                return result

            except Exception as e:
                execution_time = (time.time() - start_time) * 1000
                logger.error(
                    "Function execution failed",
                    function=func.__name__,
                    execution_time_ms=round(execution_time, 2),
                    error=str(e),
                    error_type=type(e).__name__,
                    exc_info=True
                )
                raise

        @wraps(func)
        def sync_wrapper(*args, **kwargs):
            logger = get_logger(func.__module__)
            start_time = time.time()

            try:
                result = func(*args, **kwargs)
                execution_time = (time.time() - start_time) * 1000

                log_data = {
                    "function": func.__name__,
                    "execution_time_ms": round(execution_time, 2),
                    "args_count": len(args),
                    "kwargs_count": len(kwargs),
                }

                if execution_time > threshold_ms:
                    logger.warning("Slow function execution", **log_data)
                else:
                    logger.debug("Function execution completed", **log_data)

                return result

            except Exception as e:
                execution_time = (time.time() - start_time) * 1000
                logger.error(
                    "Function execution failed",
                    function=func.__name__,
                    execution_time_ms=round(execution_time, 2),
                    error=str(e),
                    error_type=type(e).__name__,
                    exc_info=True
                )
                raise

        # Return appropriate wrapper based on function type
        import asyncio
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        else:
            return sync_wrapper

    return decorator


def setup_logging(
        log_level: str = "INFO",
        log_format: str = "json",
        enable_correlation: bool = True
) -> None:
    """
    Setup structured logging configuration.

    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_format: Output format ("json" or "text")
        enable_correlation: Enable correlation ID tracking
    """
    # Configure standard library logging
    logging.basicConfig(
        format="%(message)s",
        stream=sys.stdout,
        level=getattr(logging, log_level.upper(), logging.INFO)
    )

    # Disable noisy loggers
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    logging.getLogger("uvicorn.access").setLevel(logging.WARNING)

    # Build processor chain
    processors: list[Processor] = [
        structlog.stdlib.filter_by_level,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.stdlib.PositionalArgumentsFormatter(),
        timestamper,
        add_severity,
        add_service_context,
        filter_sensitive_data,
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
        structlog.processors.UnicodeDecoder(),
    ]

    # Add correlation context if enabled
    if enable_correlation:
        processors.insert(-3, structlog.contextvars.merge_contextvars)

    # Add appropriate formatter
    if log_format.lower() == "json":
        processors.append(structlog.processors.JSONRenderer())
    else:
        processors.append(
            structlog.dev.ConsoleRenderer(
                colors=sys.stdout.isatty(),
                exception_formatter=structlog.dev.rich_traceback
            )
        )

    # Configure structlog
    structlog.configure(
        processors=processors,
        context_class=dict,
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=True,
    )


def get_logger(name: Optional[str] = None) -> structlog.BoundLogger:
    """
    Get a configured logger instance.

    Args:
        name: Logger name, defaults to caller module if None

    Returns:
        Configured structlog BoundLogger instance
    """
    if name is None:
        # Get the caller's module name
        import inspect
        frame = inspect.currentframe()
        if frame and frame.f_back:
            name = frame.f_back.f_globals.get('__name__', 'unknown')
        else:
            name = 'unknown'

    return structlog.get_logger(name)


def bind_context(**kwargs: Any) -> None:
    """
    Bind context variables to the current request/task.

    Args:
        **kwargs: Context variables to bind
    """
    structlog.contextvars.bind_contextvars(**kwargs)


def clear_context() -> None:
    """Clear all context variables."""
    structlog.contextvars.clear_contextvars()


def log_function_call(
        logger: Optional[structlog.BoundLogger] = None,
        include_args: bool = False,
        include_result: bool = False
) -> Callable:
    """
    Decorator to log function calls.

    Args:
        logger: Logger instance to use, creates new if None
        include_args: Whether to log function arguments
        include_result: Whether to log function result

    Returns:
        Decorated function with call logging
    """

    def decorator(func: Callable) -> Callable:
        nonlocal logger
        if logger is None:
            logger = get_logger(func.__module__)

        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            log_data = {"function": func.__name__}

            if include_args:
                log_data.update({
                    "args": args,
                    "kwargs": kwargs
                })

            logger.debug("Function called", **log_data)

            try:
                result = await func(*args, **kwargs)

                if include_result:
                    logger.debug(
                        "Function completed",
                        function=func.__name__,
                        result=result
                    )
                else:
                    logger.debug("Function completed", function=func.__name__)

                return result

            except Exception as e:
                logger.error(
                    "Function failed",
                    function=func.__name__,
                    error=str(e),
                    error_type=type(e).__name__,
                    exc_info=True
                )
                raise

        @wraps(func)
        def sync_wrapper(*args, **kwargs):
            log_data = {"function": func.__name__}

            if include_args:
                log_data.update({
                    "args": args,
                    "kwargs": kwargs
                })

            logger.debug("Function called", **log_data)

            try:
                result = func(*args, **kwargs)

                if include_result:
                    logger.debug(
                        "Function completed",
                        function=func.__name__,
                        result=result
                    )
                else:
                    logger.debug("Function completed", function=func.__name__)

                return result

            except Exception as e:
                logger.error(
                    "Function failed",
                    function=func.__name__,
                    error=str(e),
                    error_type=type(e).__name__,
                    exc_info=True
                )
                raise

        # Return appropriate wrapper
        import asyncio
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        else:
            return sync_wrapper

    return decorator


class LoggingContext:
    """Context manager for temporary logging context."""

    def __init__(self, **context: Any):
        """
        Initialize logging context.

        Args:
            **context: Context variables to bind
        """
        self.context = context
        self.previous_context: Dict[str, Any] = {}

    def __enter__(self):
        """Enter the context and bind variables."""
        # Store current context
        try:
            self.previous_context = structlog.contextvars.get_contextvars()
        except LookupError:
            self.previous_context = {}

        # Bind new context
        bind_context(**self.context)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Exit the context and restore previous variables."""
        # Clear current context
        clear_context()

        # Restore previous context
        if self.previous_context:
            bind_context(**self.previous_context)


def create_correlation_id() -> str:
    """
    Create a new correlation ID for request tracking.

    Returns:
        UUID string for correlation tracking
    """
    import uuid
    return str(uuid.uuid4())


def log_request_start(
        method: str,
        path: str,
        query_params: Optional[Dict[str, Any]] = None,
        headers: Optional[Dict[str, str]] = None,
        correlation_id: Optional[str] = None
) -> None:
    """
    Log the start of an HTTP request.

    Args:
        method: HTTP method
        path: Request path
        query_params: Query parameters
        headers: Request headers (sensitive headers will be filtered)
        correlation_id: Correlation ID for tracking
    """
    logger = get_logger("request")

    log_data = {
        "event": "request_start",
        "method": method,
        "path": path,
    }

    if correlation_id:
        log_data["correlation_id"] = correlation_id
        bind_context(correlation_id=correlation_id)

    if query_params:
        log_data["query_params"] = query_params

    if headers:
        # Filter sensitive headers
        filtered_headers = {
            k: v if k.lower() not in {"authorization", "cookie", "x-api-key"}
            else "[REDACTED]"
            for k, v in headers.items()
        }
        log_data["headers"] = filtered_headers

    logger.info("HTTP request started", **log_data)


def log_request_end(
        method: str,
        path: str,
        status_code: int,
        response_time_ms: float,
        response_size: Optional[int] = None,
        error: Optional[str] = None
) -> None:
    """
    Log the end of an HTTP request.

    Args:
        method: HTTP method
        path: Request path
        status_code: HTTP status code
        response_time_ms: Response time in milliseconds
        response_size: Response size in bytes
        error: Error message if request failed
    """
    logger = get_logger("request")

    log_data = {
        "event": "request_end",
        "method": method,
        "path": path,
        "status_code": status_code,
        "response_time_ms": round(response_time_ms, 2),
    }

    if response_size is not None:
        log_data["response_size"] = response_size

    if error:
        log_data["error"] = error
        logger.error("HTTP request failed", **log_data)
    elif status_code >= 400:
        logger.warning("HTTP request completed with error", **log_data)
    else:
        logger.info("HTTP request completed", **log_data)


# Export commonly used functions
__all__ = [
    "setup_logging",
    "get_logger",
    "bind_context",
    "clear_context",
    "performance_logger",
    "log_function_call",
    "LoggingContext",
    "create_correlation_id",
    "log_request_start",
    "log_request_end",
]
