"""
Webhook PostgreSQL Model
========================

SQLAlchemy model for webhook management and delivery tracking.
"""

from datetime import datetime, timedelta
from typing import Optional, List, Dict, Any
from sqlalchemy.orm import Mapped, mapped_column, relationship
from sqlalchemy import String, Integer, DateTime, Boolean, Text, JSON, ForeignKey, Index
from sqlalchemy.dialects.postgresql import ENUM

from ..base_model import BasePostgresModel, TimestampMixin
from ..types import TenantId, UserId, WebhookId, DeliveryId, WebhookStatus, WebhookEventType, DeliveryStatus


class Webhook(BasePostgresModel, TimestampMixin):
    """Webhook configuration model"""
    __tablename__ = "webhooks"

    # Primary fields
    webhook_id: Mapped[str] = mapped_column(String(50), primary_key=True)
    tenant_id: Mapped[str] = mapped_column(
        String(50),
        ForeignKey("tenants.tenant_id", ondelete="CASCADE"),
        nullable=False
    )
    name: Mapped[str] = mapped_column(String(255), nullable=False)
    description: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    url: Mapped[str] = mapped_column(String(1000), nullable=False)
    secret: Mapped[str] = mapped_column(String(255), nullable=False)
    status: Mapped[WebhookStatus] = mapped_column(
        ENUM(WebhookStatus, name="webhook_status"),
        default=WebhookStatus.ACTIVE,
        nullable=False
    )

    # Event configuration
    events: Mapped[List[str]] = mapped_column(JSON, nullable=False, default=list)
    event_filters: Mapped[Optional[Dict[str, Any]]] = mapped_column(JSON, nullable=True)

    # Security settings
    verify_ssl: Mapped[bool] = mapped_column(Boolean, default=True)
    timeout_seconds: Mapped[int] = mapped_column(Integer, default=30)
    authentication: Mapped[Optional[Dict[str, Any]]] = mapped_column(JSON, nullable=True)

    # Request configuration
    http_method: Mapped[str] = mapped_column(String(10), default="POST")
    content_type: Mapped[str] = mapped_column(String(100), default="application/json")
    custom_headers: Mapped[Optional[Dict[str, str]]] = mapped_column(JSON, nullable=True)
    payload_template: Mapped[Optional[str]] = mapped_column(Text, nullable=True)

    # Retry configuration
    max_retries: Mapped[int] = mapped_column(Integer, default=3)
    retry_delay_seconds: Mapped[int] = mapped_column(Integer, default=60)
    exponential_backoff: Mapped[bool] = mapped_column(Boolean, default=True)
    backoff_multiplier: Mapped[float] = mapped_column(default=2.0)
    max_retry_delay: Mapped[int] = mapped_column(Integer, default=3600)  # 1 hour max

    # Rate limiting
    rate_limit_enabled: Mapped[bool] = mapped_column(Boolean, default=False)
    requests_per_minute: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    requests_per_hour: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)

    # Metadata
    created_by: Mapped[Optional[str]] = mapped_column(String(50), nullable=True)
    tags: Mapped[Optional[List[str]]] = mapped_column(JSON, nullable=True, default=list)

    # Activity tracking
    last_triggered_at: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)
    last_success_at: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)
    last_failure_at: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)

    # Statistics
    total_deliveries: Mapped[int] = mapped_column(Integer, default=0)
    successful_deliveries: Mapped[int] = mapped_column(Integer, default=0)
    failed_deliveries: Mapped[int] = mapped_column(Integer, default=0)
    average_response_time_ms: Mapped[Optional[float]] = mapped_column(nullable=True)

    # Health monitoring
    consecutive_failures: Mapped[int] = mapped_column(Integer, default=0)
    health_check_enabled: Mapped[bool] = mapped_column(Boolean, default=False)
    health_check_interval: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)  # minutes
    last_health_check: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)

    # Relationships
    tenant: Mapped["Tenant"] = relationship("Tenant", back_populates="webhooks")
    deliveries: Mapped[List["WebhookDelivery"]] = relationship(
        "WebhookDelivery",
        back_populates="webhook",
        cascade="all, delete-orphan"
    )

    # Indexes
    __table_args__ = (
        Index('idx_webhook_tenant', 'tenant_id'),
        Index('idx_webhook_status', 'status'),
        Index('idx_webhook_created_by', 'created_by'),
        Index('idx_webhook_last_triggered', 'last_triggered_at'),
        Index('idx_webhook_tenant_status', 'tenant_id', 'status'),
        Index('idx_webhook_events', 'events'),  # GIN index for JSON array
    )

    def __repr__(self) -> str:
        return f"<Webhook(webhook_id='{self.webhook_id}', name='{self.name}', status='{self.status}')>"

    def is_active(self) -> bool:
        """Check if webhook is active"""
        return self.status == WebhookStatus.ACTIVE

    def should_trigger_for_event(self, event_type: WebhookEventType) -> bool:
        """Check if webhook should trigger for given event type"""
        if not self.is_active():
            return False
        return event_type.value in self.events

    def get_success_rate(self) -> float:
        """Calculate success rate percentage"""
        if self.total_deliveries == 0:
            return 0.0
        return (self.successful_deliveries / self.total_deliveries) * 100

    def is_healthy(self) -> bool:
        """Check if webhook is considered healthy"""
        if self.consecutive_failures >= 5:
            return False
        if self.total_deliveries > 10 and self.get_success_rate() < 50:
            return False
        return True

    def should_disable_automatically(self) -> bool:
        """Check if webhook should be automatically disabled"""
        # Disable after 10 consecutive failures
        if self.consecutive_failures >= 10:
            return True

        # Disable if success rate is very low with enough samples
        if self.total_deliveries >= 50 and self.get_success_rate() < 10:
            return True

        return False

    def update_statistics(self, success: bool, response_time_ms: Optional[float] = None) -> None:
        """Update delivery statistics"""
        self.total_deliveries += 1
        self.last_triggered_at = datetime.utcnow()

        if success:
            self.successful_deliveries += 1
            self.last_success_at = datetime.utcnow()
            self.consecutive_failures = 0
        else:
            self.failed_deliveries += 1
            self.last_failure_at = datetime.utcnow()
            self.consecutive_failures += 1

        # Update average response time
        if response_time_ms is not None:
            if self.average_response_time_ms is None:
                self.average_response_time_ms = response_time_ms
            else:
                # Exponential moving average
                self.average_response_time_ms = (
                        0.9 * self.average_response_time_ms + 0.1 * response_time_ms
                )

    def calculate_next_retry_delay(self, attempt_number: int) -> int:
        """Calculate delay for next retry attempt"""
        if not self.exponential_backoff:
            return self.retry_delay_seconds

        delay = self.retry_delay_seconds * (self.backoff_multiplier ** (attempt_number - 1))
        return min(int(delay), self.max_retry_delay)

    def get_event_types(self) -> List[WebhookEventType]:
        """Get configured event types as enum objects"""
        return [
            WebhookEventType(event)
            for event in self.events
            if event in [e.value for e in WebhookEventType]
        ]


class WebhookDelivery(BasePostgresModel, TimestampMixin):
    """Webhook delivery attempt record"""
    __tablename__ = "webhook_deliveries"

    # Primary fields
    delivery_id: Mapped[str] = mapped_column(String(50), primary_key=True)
    webhook_id: Mapped[str] = mapped_column(
        String(50),
        ForeignKey("webhooks.webhook_id", ondelete="CASCADE"),
        nullable=False
    )
    tenant_id: Mapped[str] = mapped_column(
        String(50),
        ForeignKey("tenants.tenant_id", ondelete="CASCADE"),
        nullable=False
    )
    event_type: Mapped[WebhookEventType] = mapped_column(
        ENUM(WebhookEventType, name="webhook_event_type"),
        nullable=False
    )
    event_data: Mapped[Dict[str, Any]] = mapped_column(JSON, nullable=False)

    # Delivery details
    status: Mapped[DeliveryStatus] = mapped_column(
        ENUM(DeliveryStatus, name="delivery_status"),
        default=DeliveryStatus.PENDING,
        nullable=False
    )
    attempt_number: Mapped[int] = mapped_column(Integer, default=1)
    max_attempts: Mapped[int] = mapped_column(Integer, default=3)

    # Request details
    request_url: Mapped[str] = mapped_column(String(1000), nullable=False)
    request_method: Mapped[str] = mapped_column(String(10), default="POST")
    request_headers: Mapped[Optional[Dict[str, str]]] = mapped_column(JSON, nullable=True)
    request_body: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    request_size_bytes: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)

    # Response details
    response_status: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    response_headers: Mapped[Optional[Dict[str, str]]] = mapped_column(JSON, nullable=True)
    response_body: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    response_size_bytes: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    response_time_ms: Mapped[Optional[float]] = mapped_column(nullable=True)

    # Timing
    scheduled_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    started_at: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)
    delivered_at: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)
    next_retry_at: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)
    expires_at: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)

    # Error handling
    error_message: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    error_code: Mapped[Optional[str]] = mapped_column(String(100), nullable=True)
    error_type: Mapped[Optional[str]] = mapped_column(String(50),
                                                      nullable=True)  # timeout, connection, http_error, etc.

    # Tracking
    correlation_id: Mapped[Optional[str]] = mapped_column(String(100), nullable=True)
    user_agent: Mapped[str] = mapped_column(String(255), default="chatbot-platform-webhook/1.0")

    # Relationships
    webhook: Mapped["Webhook"] = relationship("Webhook", back_populates="deliveries")
    tenant: Mapped["Tenant"] = relationship("Tenant")

    # Indexes
    __table_args__ = (
        Index('idx_delivery_webhook', 'webhook_id'),
        Index('idx_delivery_tenant', 'tenant_id'),
        Index('idx_delivery_status', 'status'),
        Index('idx_delivery_event_type', 'event_type'),
        Index('idx_delivery_scheduled', 'scheduled_at'),
        Index('idx_delivery_next_retry', 'next_retry_at'),
        Index('idx_delivery_correlation', 'correlation_id'),
        Index('idx_delivery_webhook_status', 'webhook_id', 'status'),
        Index('idx_delivery_tenant_event', 'tenant_id', 'event_type'),
        Index('idx_delivery_pending_retry', 'status', 'next_retry_at'),
    )

    def __repr__(self) -> str:
        return (
            f"<WebhookDelivery(delivery_id='{self.delivery_id}', "
            f"webhook_id='{self.webhook_id}', status='{self.status}')>"
        )

    def is_pending(self) -> bool:
        """Check if delivery is pending"""
        return self.status in [DeliveryStatus.PENDING, DeliveryStatus.RETRYING]

    def is_successful(self) -> bool:
        """Check if delivery was successful"""
        return self.status == DeliveryStatus.DELIVERED

    def is_failed(self) -> bool:
        """Check if delivery failed permanently"""
        return self.status in [DeliveryStatus.FAILED, DeliveryStatus.EXPIRED, DeliveryStatus.CANCELLED]

    def can_retry(self) -> bool:
        """Check if delivery can be retried"""
        return (
                self.status in [DeliveryStatus.PENDING, DeliveryStatus.RETRYING] and
                self.attempt_number < self.max_attempts and
                (self.expires_at is None or self.expires_at > datetime.utcnow())
        )

    def should_retry_now(self) -> bool:
        """Check if delivery should be retried now"""
        return (
                self.can_retry() and
                (self.next_retry_at is None or self.next_retry_at <= datetime.utcnow())
        )

    def mark_started(self) -> None:
        """Mark delivery as started"""
        self.started_at = datetime.utcnow()

    def mark_delivered(self, response_status: int, response_headers: Optional[Dict[str, str]] = None,
                       response_body: Optional[str] = None, response_time_ms: Optional[float] = None) -> None:
        """Mark delivery as successfully delivered"""
        self.status = DeliveryStatus.DELIVERED
        self.delivered_at = datetime.utcnow()
        self.response_status = response_status
        self.response_headers = response_headers or {}
        self.response_body = response_body
        self.response_time_ms = response_time_ms

        if response_body:
            self.response_size_bytes = len(response_body.encode('utf-8'))

    def mark_failed(self, error_message: str, error_code: Optional[str] = None,
                    error_type: Optional[str] = None, response_status: Optional[int] = None,
                    response_headers: Optional[Dict[str, str]] = None,
                    response_body: Optional[str] = None) -> None:
        """Mark delivery as failed"""
        if self.attempt_number >= self.max_attempts:
            self.status = DeliveryStatus.FAILED
        else:
            self.status = DeliveryStatus.RETRYING

        self.error_message = error_message
        self.error_code = error_code
        self.error_type = error_type
        self.response_status = response_status
        self.response_headers = response_headers or {}
        self.response_body = response_body

    def schedule_retry(self, delay_seconds: int) -> None:
        """Schedule next retry attempt"""
        self.attempt_number += 1
        self.next_retry_at = datetime.utcnow() + timedelta(seconds=delay_seconds)
        self.status = DeliveryStatus.RETRYING

    def mark_expired(self) -> None:
        """Mark delivery as expired"""
        self.status = DeliveryStatus.EXPIRED

    def mark_cancelled(self, reason: Optional[str] = None) -> None:
        """Mark delivery as cancelled"""
        self.status = DeliveryStatus.CANCELLED
        if reason:
            self.error_message = f"Cancelled: {reason}"

    def get_duration_ms(self) -> Optional[float]:
        """Get total duration from start to completion in milliseconds"""
        if self.started_at and self.delivered_at:
            delta = self.delivered_at - self.started_at
            return delta.total_seconds() * 1000
        return None

    def is_http_success(self) -> bool:
        """Check if HTTP response indicates success"""
        return self.response_status is not None and 200 <= self.response_status < 300