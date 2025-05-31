"""
Metrics collection and monitoring utilities.

This module provides comprehensive metrics collection, monitoring,
and performance tracking capabilities for the Chat Service.
"""

import time
import functools
from typing import Dict, Any, Optional, Callable, Union
from contextlib import contextmanager
from datetime import datetime, timezone
from collections import defaultdict, deque
from dataclasses import dataclass, field

try:
    from prometheus_client import (
        Counter, Histogram, Gauge, Summary, Info,
        CollectorRegistry, generate_latest, CONTENT_TYPE_LATEST
    )

    PROMETHEUS_AVAILABLE = True
except ImportError:
    PROMETHEUS_AVAILABLE = False

import structlog
from src.utils.logger import get_logger
from src.config.constants import SERVICE_NAME, SERVICE_VERSION

# Initialize logger
logger = get_logger(__name__)


@dataclass
class MetricData:
    """Data structure for storing metric information."""
    name: str
    value: Union[int, float]
    labels: Dict[str, str] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    metric_type: str = "counter"


class MetricsCollector:
    """
    Centralized metrics collection and management.

    This class provides a unified interface for collecting various
    types of metrics including counters, histograms, and gauges.
    """

    def __init__(self, enable_prometheus: bool = True):
        """
        Initialize metrics collector.

        Args:
            enable_prometheus: Whether to enable Prometheus metrics
        """
        self.enable_prometheus = enable_prometheus and PROMETHEUS_AVAILABLE
        self.metrics_store: Dict[str, MetricData] = {}
        self.time_series: Dict[str, deque] = defaultdict(lambda: deque(maxlen=1000))

        # Prometheus metrics registry
        if self.enable_prometheus:
            self.registry = CollectorRegistry()
            self._setup_prometheus_metrics()

        # Internal counters for basic metrics
        self.counters: Dict[str, int] = defaultdict(int)
        self.gauges: Dict[str, float] = defaultdict(float)
        self.histograms: Dict[str, list] = defaultdict(list)

        logger.info(
            "Metrics collector initialized",
            prometheus_enabled=self.enable_prometheus,
            service=SERVICE_NAME
        )

    def _setup_prometheus_metrics(self) -> None:
        """Setup Prometheus metrics collectors."""
        if not self.enable_prometheus:
            return

        # Request metrics
        self.request_counter = Counter(
            'chat_service_requests_total',
            'Total number of HTTP requests',
            ['method', 'endpoint', 'status_code', 'tenant_id'],
            registry=self.registry
        )

        self.request_duration = Histogram(
            'chat_service_request_duration_seconds',
            'HTTP request duration in seconds',
            ['method', 'endpoint', 'tenant_id'],
            registry=self.registry
        )

        # Message processing metrics
        self.messages_processed = Counter(
            'chat_service_messages_processed_total',
            'Total number of messages processed',
            ['channel', 'message_type', 'tenant_id'],
            registry=self.registry
        )

        self.message_processing_duration = Histogram(
            'chat_service_message_processing_duration_seconds',
            'Message processing duration in seconds',
            ['channel', 'message_type', 'tenant_id'],
            registry=self.registry
        )

        # Conversation metrics
        self.conversations_active = Gauge(
            'chat_service_conversations_active',
            'Number of active conversations',
            ['tenant_id'],
            registry=self.registry
        )

        self.conversations_total = Counter(
            'chat_service_conversations_total',
            'Total number of conversations',
            ['channel', 'tenant_id'],
            registry=self.registry
        )

        # Error metrics
        self.errors_total = Counter(
            'chat_service_errors_total',
            'Total number of errors',
            ['error_type', 'error_code', 'tenant_id'],
            registry=self.registry
        )

        # External service metrics
        self.external_service_requests = Counter(
            'chat_service_external_requests_total',
            'Total external service requests',
            ['service_name', 'operation', 'status'],
            registry=self.registry
        )

        self.external_service_duration = Histogram(
            'chat_service_external_request_duration_seconds',
            'External service request duration',
            ['service_name', 'operation'],
            registry=self.registry
        )

        # System metrics
        self.active_connections = Gauge(
            'chat_service_active_connections',
            'Number of active connections',
            ['connection_type'],
            registry=self.registry
        )

        # Service info
        self.service_info = Info(
            'chat_service_info',
            'Service information',
            registry=self.registry
        )
        self.service_info.info({
            'version': SERVICE_VERSION,
            'service': SERVICE_NAME
        })

    def increment_counter(
            self,
            name: str,
            value: int = 1,
            labels: Optional[Dict[str, str]] = None
    ) -> None:
        """
        Increment a counter metric.

        Args:
            name: Metric name
            value: Increment value
            labels: Metric labels
        """
        labels = labels or {}

        # Internal counter
        key = f"{name}:{':'.join(f'{k}={v}' for k, v in sorted(labels.items()))}"
        self.counters[key] += value

        # Store metric data
        self.metrics_store[key] = MetricData(
            name=name,
            value=self.counters[key],
            labels=labels,
            metric_type="counter"
        )

        # Add to time series
        self.time_series[key].append({
            'timestamp': datetime.now(timezone.utc),
            'value': self.counters[key]
        })

    def set_gauge(
            self,
            name: str,
            value: float,
            labels: Optional[Dict[str, str]] = None
    ) -> None:
        """
        Set a gauge metric value.

        Args:
            name: Metric name
            value: Gauge value
            labels: Metric labels
        """
        labels = labels or {}

        # Internal gauge
        key = f"{name}:{':'.join(f'{k}={v}' for k, v in sorted(labels.items()))}"
        self.gauges[key] = value

        # Store metric data
        self.metrics_store[key] = MetricData(
            name=name,
            value=value,
            labels=labels,
            metric_type="gauge"
        )

        # Add to time series
        self.time_series[key].append({
            'timestamp': datetime.now(timezone.utc),
            'value': value
        })

    def observe_histogram(
            self,
            name: str,
            value: float,
            labels: Optional[Dict[str, str]] = None
    ) -> None:
        """
        Observe a value in a histogram metric.

        Args:
            name: Metric name
            value: Observed value
            labels: Metric labels
        """
        labels = labels or {}

        # Internal histogram
        key = f"{name}:{':'.join(f'{k}={v}' for k, v in sorted(labels.items()))}"
        self.histograms[key].append(value)

        # Keep only last 1000 observations
        if len(self.histograms[key]) > 1000:
            self.histograms[key] = self.histograms[key][-1000:]

        # Store metric data
        self.metrics_store[key] = MetricData(
            name=name,
            value=value,
            labels=labels,
            metric_type="histogram"
        )

        # Add to time series
        self.time_series[key].append({
            'timestamp': datetime.now(timezone.utc),
            'value': value
        })

    def record_request(
            self,
            method: str,
            endpoint: str,
            status_code: int,
            duration: float,
            tenant_id: Optional[str] = None
    ) -> None:
        """
        Record HTTP request metrics.

        Args:
            method: HTTP method
            endpoint: Request endpoint
            status_code: HTTP status code
            duration: Request duration in seconds
            tenant_id: Optional tenant ID
        """
        labels = {
            'method': method,
            'endpoint': endpoint,
            'status_code': str(status_code)
        }

        if tenant_id:
            labels['tenant_id'] = tenant_id

        # Prometheus metrics
        if self.enable_prometheus:
            self.request_counter.labels(**labels).inc()
            self.request_duration.labels(
                method=method,
                endpoint=endpoint,
                tenant_id=tenant_id or 'unknown'
            ).observe(duration)

        # Internal metrics
        self.increment_counter('http_requests_total', 1, labels)
        self.observe_histogram('http_request_duration_seconds', duration, labels)

    def record_message_processing(
            self,
            channel: str,
            message_type: str,
            duration: float,
            tenant_id: Optional[str] = None,
            success: bool = True
    ) -> None:
        """
        Record message processing metrics.

        Args:
            channel: Message channel
            message_type: Type of message
            duration: Processing duration in seconds
            tenant_id: Optional tenant ID
            success: Whether processing was successful
        """
        labels = {
            'channel': channel,
            'message_type': message_type,
            'status': 'success' if success else 'error'
        }

        if tenant_id:
            labels['tenant_id'] = tenant_id

        # Prometheus metrics
        if self.enable_prometheus:
            self.messages_processed.labels(
                channel=channel,
                message_type=message_type,
                tenant_id=tenant_id or 'unknown'
            ).inc()

            self.message_processing_duration.labels(
                channel=channel,
                message_type=message_type,
                tenant_id=tenant_id or 'unknown'
            ).observe(duration)

        # Internal metrics
        self.increment_counter('messages_processed_total', 1, labels)
        self.observe_histogram('message_processing_duration_seconds', duration, labels)

    def record_error(
            self,
            error_type: str,
            error_code: str,
            tenant_id: Optional[str] = None
    ) -> None:
        """
        Record error metrics.

        Args:
            error_type: Type of error
            error_code: Error code
            tenant_id: Optional tenant ID
        """
        labels = {
            'error_type': error_type,
            'error_code': error_code
        }

        if tenant_id:
            labels['tenant_id'] = tenant_id

        # Prometheus metrics
        if self.enable_prometheus:
            self.errors_total.labels(**labels).inc()

        # Internal metrics
        self.increment_counter('errors_total', 1, labels)

    def record_external_service_call(
            self,
            service_name: str,
            operation: str,
            duration: float,
            success: bool = True
    ) -> None:
        """
        Record external service call metrics.

        Args:
            service_name: Name of external service
            operation: Operation performed
            duration: Call duration in seconds
            success: Whether call was successful
        """
        status = 'success' if success else 'error'

        # Prometheus metrics
        if self.enable_prometheus:
            self.external_service_requests.labels(
                service_name=service_name,
                operation=operation,
                status=status
            ).inc()

            self.external_service_duration.labels(
                service_name=service_name,
                operation=operation
            ).observe(duration)

        # Internal metrics
        labels = {
            'service_name': service_name,
            'operation': operation,
            'status': status
        }
        self.increment_counter('external_service_requests_total', 1, labels)
        self.observe_histogram('external_service_duration_seconds', duration, labels)

    def set_active_conversations(self, count: int, tenant_id: Optional[str] = None) -> None:
        """
        Set the number of active conversations.

        Args:
            count: Number of active conversations
            tenant_id: Optional tenant ID
        """
        labels = {'tenant_id': tenant_id or 'unknown'}

        # Prometheus metrics
        if self.enable_prometheus:
            self.conversations_active.labels(tenant_id=tenant_id or 'unknown').set(count)

        # Internal metrics
        self.set_gauge('conversations_active', count, labels)

    def set_active_connections(self, count: int, connection_type: str) -> None:
        """
        Set the number of active connections.

        Args:
            count: Number of active connections
            connection_type: Type of connection (database, redis, etc.)
        """
        # Prometheus metrics
        if self.enable_prometheus:
            self.active_connections.labels(connection_type=connection_type).set(count)

        # Internal metrics
        labels = {'connection_type': connection_type}
        self.set_gauge('active_connections', count, labels)

    def get_metrics_summary(self) -> Dict[str, Any]:
        """
        Get a summary of all collected metrics.

        Returns:
            Dictionary containing metrics summary
        """
        summary = {
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'service': SERVICE_NAME,
            'version': SERVICE_VERSION,
            'counters': dict(self.counters),
            'gauges': dict(self.gauges),
            'histogram_counts': {k: len(v) for k, v in self.histograms.items()}
        }

        # Add histogram statistics
        histogram_stats = {}
        for name, values in self.histograms.items():
            if values:
                import statistics
                histogram_stats[name] = {
                    'count': len(values),
                    'min': min(values),
                    'max': max(values),
                    'mean': statistics.mean(values),
                    'median': statistics.median(values)
                }

        summary['histogram_stats'] = histogram_stats

        return summary

    def export_prometheus_metrics(self) -> str:
        """
        Export metrics in Prometheus format.

        Returns:
            Prometheus metrics string
        """
        if not self.enable_prometheus:
            return "# Prometheus metrics not enabled\n"

        return generate_latest(self.registry).decode('utf-8')

    def reset_metrics(self) -> None:
        """Reset all metrics (useful for testing)."""
        self.counters.clear()
        self.gauges.clear()
        self.histograms.clear()
        self.metrics_store.clear()
        self.time_series.clear()

        logger.info("Metrics reset completed")


# Global metrics collector instance
metrics_collector = MetricsCollector()


def get_metrics_collector() -> MetricsCollector:
    """
    Get the global metrics collector instance.

    Returns:
        MetricsCollector instance
    """
    return metrics_collector


@contextmanager
def measure_time(
        metric_name: str,
        labels: Optional[Dict[str, str]] = None
):
    """
    Context manager to measure execution time.

    Args:
        metric_name: Name of the metric to record
        labels: Optional metric labels

    Usage:
        with measure_time('operation_duration', {'operation': 'process_message'}):
            # Your code here
            pass
    """
    start_time = time.time()
    try:
        yield
    finally:
        duration = time.time() - start_time
        metrics_collector.observe_histogram(metric_name, duration, labels)


def measure_execution_time(
        metric_name: Optional[str] = None,
        labels: Optional[Dict[str, str]] = None
):
    """
    Decorator to measure function execution time.

    Args:
        metric_name: Custom metric name (defaults to function name)
        labels: Optional metric labels

    Returns:
        Decorated function
    """

    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        async def async_wrapper(*args, **kwargs):
            name = metric_name or f"{func.__module__}.{func.__name__}_duration_seconds"
            start_time = time.time()

            try:
                result = await func(*args, **kwargs)
                success = True
                return result
            except Exception as e:
                success = False
                raise
            finally:
                duration = time.time() - start_time
                final_labels = labels or {}
                final_labels['function'] = func.__name__
                final_labels['status'] = 'success' if success else 'error'
                metrics_collector.observe_histogram(name, duration, final_labels)

        @functools.wraps(func)
        def sync_wrapper(*args, **kwargs):
            name = metric_name or f"{func.__module__}.{func.__name__}_duration_seconds"
            start_time = time.time()

            try:
                result = func(*args, **kwargs)
                success = True
                return result
            except Exception as e:
                success = False
                raise
            finally:
                duration = time.time() - start_time
                final_labels = labels or {}
                final_labels['function'] = func.__name__
                final_labels['status'] = 'success' if success else 'error'
                metrics_collector.observe_histogram(name, duration, final_labels)

        # Return appropriate wrapper
        import asyncio
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        else:
            return sync_wrapper

    return decorator


def count_calls(
        metric_name: Optional[str] = None,
        labels: Optional[Dict[str, str]] = None
):
    """
    Decorator to count function calls.

    Args:
        metric_name: Custom metric name (defaults to function name)
        labels: Optional metric labels

    Returns:
        Decorated function
    """

    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        async def async_wrapper(*args, **kwargs):
            name = metric_name or f"{func.__module__}.{func.__name__}_calls_total"

            try:
                result = await func(*args, **kwargs)
                success = True
                return result
            except Exception as e:
                success = False
                raise
            finally:
                final_labels = labels or {}
                final_labels['function'] = func.__name__
                final_labels['status'] = 'success' if success else 'error'
                metrics_collector.increment_counter(name, 1, final_labels)

        @functools.wraps(func)
        def sync_wrapper(*args, **kwargs):
            name = metric_name or f"{func.__module__}.{func.__name__}_calls_total"

            try:
                result = func(*args, **kwargs)
                success = True
                return result
            except Exception as e:
                success = False
                raise
            finally:
                final_labels = labels or {}
                final_labels['function'] = func.__name__
                final_labels['status'] = 'success' if success else 'error'
                metrics_collector.increment_counter(name, 1, final_labels)

        # Return appropriate wrapper
        import asyncio
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        else:
            return sync_wrapper

    return decorator


# Business metrics helpers
def track_message_sent(
        channel: str,
        message_type: str,
        tenant_id: Optional[str] = None
) -> None:
    """Track a sent message."""
    labels = {
        'channel': channel,
        'message_type': message_type,
        'direction': 'outbound'
    }
    if tenant_id:
        labels['tenant_id'] = tenant_id

    metrics_collector.increment_counter('messages_total', 1, labels)


def track_message_received(
        channel: str,
        message_type: str,
        tenant_id: Optional[str] = None
) -> None:
    """Track a received message."""
    labels = {
        'channel': channel,
        'message_type': message_type,
        'direction': 'inbound'
    }
    if tenant_id:
        labels['tenant_id'] = tenant_id

    metrics_collector.increment_counter('messages_total', 1, labels)


def track_conversation_started(
        channel: str,
        tenant_id: Optional[str] = None
) -> None:
    """Track a new conversation."""
    labels = {
        'channel': channel,
        'event': 'started'
    }
    if tenant_id:
        labels['tenant_id'] = tenant_id

    metrics_collector.increment_counter('conversations_events_total', 1, labels)


def track_conversation_ended(
        channel: str,
        duration_seconds: float,
        tenant_id: Optional[str] = None
) -> None:
    """Track a completed conversation."""
    labels = {
        'channel': channel,
        'event': 'ended'
    }
    if tenant_id:
        labels['tenant_id'] = tenant_id

    metrics_collector.increment_counter('conversations_events_total', 1, labels)
    metrics_collector.observe_histogram('conversation_duration_seconds', duration_seconds, labels)


# Export commonly used functions
__all__ = [
    'MetricsCollector',
    'MetricData',
    'get_metrics_collector',
    'metrics_collector',
    'measure_time',
    'measure_execution_time',
    'count_calls',
    'track_message_sent',
    'track_message_received',
    'track_conversation_started',
    'track_conversation_ended',
]