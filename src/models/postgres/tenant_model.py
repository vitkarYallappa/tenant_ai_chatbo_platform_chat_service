"""
Tenant PostgreSQL Model
=======================

SQLAlchemy model for tenant management with full feature support.
"""

from datetime import datetime, timedelta
from typing import Optional, List, Dict, Any
from sqlalchemy.orm import Mapped, mapped_column, relationship
from sqlalchemy import String, Integer, DateTime, Boolean, Text, JSON, Index
from sqlalchemy.dialects.postgresql import ENUM, UUID

from ..base_model import BasePostgresModel, TimestampMixin
from ..types import TenantId, UserId, TenantPlan, TenantStatus, ComplianceLevel, DataResidency


class TenantFeatures(BasePostgresModel):
    """Tenant feature configuration"""
    __tablename__ = "tenant_features"

    tenant_id: Mapped[str] = mapped_column(String(50), primary_key=True)

    # AI Features
    ai_enabled: Mapped[bool] = mapped_column(Boolean, default=True)
    ai_model: Mapped[str] = mapped_column(String(50), default="gpt-3.5-turbo")
    ai_max_tokens: Mapped[int] = mapped_column(Integer, default=1000)
    ai_temperature: Mapped[float] = mapped_column(default=0.7)

    # Channel Features
    channels_enabled: Mapped[List[str]] = mapped_column(JSON, default=lambda: ["web"])
    whatsapp_enabled: Mapped[bool] = mapped_column(Boolean, default=False)
    telegram_enabled: Mapped[bool] = mapped_column(Boolean, default=False)
    slack_enabled: Mapped[bool] = mapped_column(Boolean, default=False)
    api_enabled: Mapped[bool] = mapped_column(Boolean, default=True)

    # Analytics Features
    analytics_enabled: Mapped[bool] = mapped_column(Boolean, default=True)
    advanced_analytics: Mapped[bool] = mapped_column(Boolean, default=False)
    custom_dashboards: Mapped[bool] = mapped_column(Boolean, default=False)

    # Integration Features
    webhook_enabled: Mapped[bool] = mapped_column(Boolean, default=True)
    api_access: Mapped[bool] = mapped_column(Boolean, default=True)
    sso_enabled: Mapped[bool] = mapped_column(Boolean, default=False)

    # Customization Features
    custom_branding: Mapped[bool] = mapped_column(Boolean, default=False)
    white_label: Mapped[bool] = mapped_column(Boolean, default=False)
    custom_domain: Mapped[bool] = mapped_column(Boolean, default=False)

    # Advanced Features
    multi_language: Mapped[bool] = mapped_column(Boolean, default=False)
    sentiment_analysis: Mapped[bool] = mapped_column(Boolean, default=False)
    conversation_flows: Mapped[bool] = mapped_column(Boolean, default=True)

    # Support Features
    priority_support: Mapped[bool] = mapped_column(Boolean, default=False)
    dedicated_manager: Mapped[bool] = mapped_column(Boolean, default=False)

    # Timestamps
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime, default=datetime.utcnow, onupdate=datetime.utcnow
    )


class TenantQuotas(BasePostgresModel):
    """Tenant usage quotas and limits"""
    __tablename__ = "tenant_quotas"

    tenant_id: Mapped[str] = mapped_column(String(50), primary_key=True)

    # Message Quotas
    messages_per_month: Mapped[int] = mapped_column(Integer, default=10000)
    messages_used_this_month: Mapped[int] = mapped_column(Integer, default=0)

    # API Quotas
    api_calls_per_minute: Mapped[int] = mapped_column(Integer, default=100)
    api_calls_per_day: Mapped[int] = mapped_column(Integer, default=10000)
    api_calls_used_today: Mapped[int] = mapped_column(Integer, default=0)

    # Storage Quotas
    storage_gb: Mapped[int] = mapped_column(Integer, default=10)
    storage_used_gb: Mapped[float] = mapped_column(default=0.0)

    # Team Quotas
    team_members: Mapped[int] = mapped_column(Integer, default=5)
    team_members_count: Mapped[int] = mapped_column(Integer, default=0)

    # Integration Quotas
    integrations_limit: Mapped[int] = mapped_column(Integer, default=5)
    integrations_count: Mapped[int] = mapped_column(Integer, default=0)

    # Webhook Quotas
    webhooks_limit: Mapped[int] = mapped_column(Integer, default=10)
    webhooks_count: Mapped[int] = mapped_column(Integer, default=0)
    webhook_calls_per_hour: Mapped[int] = mapped_column(Integer, default=1000)

    # Flow Quotas
    conversation_flows_limit: Mapped[int] = mapped_column(Integer, default=10)
    conversation_flows_count: Mapped[int] = mapped_column(Integer, default=0)

    # AI Quotas
    ai_tokens_per_month: Mapped[int] = mapped_column(Integer, default=100000)
    ai_tokens_used_this_month: Mapped[int] = mapped_column(Integer, default=0)

    # Reset tracking
    monthly_reset_date: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    daily_reset_date: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)

    # Timestamps
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime, default=datetime.utcnow, onupdate=datetime.utcnow
    )


class Tenant(BasePostgresModel, TimestampMixin):
    """Tenant model with comprehensive configuration"""
    __tablename__ = "tenants"

    # Primary fields
    tenant_id: Mapped[str] = mapped_column(String(50), primary_key=True)
    name: Mapped[str] = mapped_column(String(255), nullable=False)
    subdomain: Mapped[Optional[str]] = mapped_column(String(100), unique=True, nullable=True)
    status: Mapped[TenantStatus] = mapped_column(
        ENUM(TenantStatus, name="tenant_status"),
        default=TenantStatus.ACTIVE,
        nullable=False
    )
    plan_type: Mapped[TenantPlan] = mapped_column(
        ENUM(TenantPlan, name="tenant_plan"),
        default=TenantPlan.STARTER,
        nullable=False
    )

    # Subscription and billing
    stripe_customer_id: Mapped[Optional[str]] = mapped_column(String(255), nullable=True)
    subscription_id: Mapped[Optional[str]] = mapped_column(String(255), nullable=True)
    billing_email: Mapped[Optional[str]] = mapped_column(String(255), nullable=True)
    billing_cycle: Mapped[str] = mapped_column(String(20), default="monthly", nullable=False)
    trial_ends_at: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)
    subscription_active: Mapped[bool] = mapped_column(Boolean, default=True)

    # Compliance and security
    compliance_level: Mapped[ComplianceLevel] = mapped_column(
        ENUM(ComplianceLevel, name="compliance_level"),
        default=ComplianceLevel.STANDARD,
        nullable=False
    )
    data_residency: Mapped[DataResidency] = mapped_column(
        ENUM(DataResidency, name="data_residency"),
        default=DataResidency.US,
        nullable=False
    )
    encryption_key_id: Mapped[Optional[str]] = mapped_column(String(255), nullable=True)
    data_retention_days: Mapped[int] = mapped_column(Integer, default=365)

    # Customization
    custom_domain: Mapped[Optional[str]] = mapped_column(String(255), nullable=True)
    custom_domain_verified: Mapped[bool] = mapped_column(Boolean, default=False)

    # Branding configuration (JSON)
    branding_config: Mapped[Optional[Dict[str, Any]]] = mapped_column(JSON, nullable=True)

    # Organization info
    contact_info: Mapped[Optional[Dict[str, Any]]] = mapped_column(JSON, nullable=True)
    organization_size: Mapped[Optional[str]] = mapped_column(String(50), nullable=True)
    industry: Mapped[Optional[str]] = mapped_column(String(100), nullable=True)
    country: Mapped[Optional[str]] = mapped_column(String(100), nullable=True)
    timezone: Mapped[str] = mapped_column(String(50), default="UTC")

    # Activity tracking
    last_activity_at: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)
    last_login_at: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)
    login_count: Mapped[int] = mapped_column(Integer, default=0)

    # Feature flags (JSON for flexible configuration)
    feature_flags: Mapped[Optional[Dict[str, Any]]] = mapped_column(JSON, nullable=True)

    # Onboarding
    onboarding_completed: Mapped[bool] = mapped_column(Boolean, default=False)
    onboarding_step: Mapped[Optional[str]] = mapped_column(String(50), nullable=True)
    setup_wizard_completed: Mapped[bool] = mapped_column(Boolean, default=False)

    # Relationships
    features: Mapped["TenantFeatures"] = relationship(
        "TenantFeatures",
        uselist=False,
        cascade="all, delete-orphan",
        foreign_keys="TenantFeatures.tenant_id",
        primaryjoin="Tenant.tenant_id == TenantFeatures.tenant_id"
    )

    quotas: Mapped["TenantQuotas"] = relationship(
        "TenantQuotas",
        uselist=False,
        cascade="all, delete-orphan",
        foreign_keys="TenantQuotas.tenant_id",
        primaryjoin="Tenant.tenant_id == TenantQuotas.tenant_id"
    )

    webhooks: Mapped[List["Webhook"]] = relationship(
        "Webhook",
        back_populates="tenant",
        cascade="all, delete-orphan"
    )

    # Indexes
    __table_args__ = (
        Index('idx_tenant_subdomain', 'subdomain'),
        Index('idx_tenant_status', 'status'),
        Index('idx_tenant_plan', 'plan_type'),
        Index('idx_tenant_created', 'created_at'),
        Index('idx_tenant_last_activity', 'last_activity_at'),
        Index('idx_tenant_stripe_customer', 'stripe_customer_id'),
        Index('idx_tenant_billing_email', 'billing_email'),
    )

    def __repr__(self) -> str:
        return f"<Tenant(tenant_id='{self.tenant_id}', name='{self.name}', status='{self.status}')>"

    def is_trial(self) -> bool:
        """Check if tenant is in trial period"""
        return (
                self.status == TenantStatus.TRIAL and
                self.trial_ends_at and
                self.trial_ends_at > datetime.utcnow()
        )

    def trial_days_remaining(self) -> Optional[int]:
        """Get remaining trial days"""
        if not self.is_trial():
            return None

        delta = self.trial_ends_at - datetime.utcnow()
        return max(0, delta.days)

    def is_subscription_active(self) -> bool:
        """Check if subscription is active"""
        return (
                self.status == TenantStatus.ACTIVE and
                self.subscription_active and
                self.subscription_id is not None
        )

    def can_use_feature(self, feature_name: str) -> bool:
        """Check if tenant can use a specific feature"""
        if not self.features:
            return False
        return getattr(self.features, feature_name, False)

    def get_quota_usage(self, quota_name: str) -> tuple[int, int]:
        """Get quota usage (used, limit)"""
        if not self.quotas:
            return 0, 0

        used = getattr(self.quotas, f"{quota_name}_used", 0)
        limit = getattr(self.quotas, quota_name, 0)
        return used, limit

    def is_quota_exceeded(self, quota_name: str) -> bool:
        """Check if quota is exceeded"""
        used, limit = self.get_quota_usage(quota_name)
        return used >= limit

    def update_activity(self) -> None:
        """Update last activity timestamp"""
        self.last_activity_at = datetime.utcnow()

    def increment_login(self) -> None:
        """Increment login count and update timestamp"""
        self.login_count += 1
        self.last_login_at = datetime.utcnow()
        self.update_activity()

    def to_dict_with_relationships(self) -> Dict[str, Any]:
        """Convert to dictionary including relationships"""
        data = self.to_dict()

        if self.features:
            data['features'] = self.features.to_dict()

        if self.quotas:
            data['quotas'] = self.quotas.to_dict()

        return data


