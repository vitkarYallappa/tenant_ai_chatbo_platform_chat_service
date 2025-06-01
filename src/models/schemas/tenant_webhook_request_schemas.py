# src/models/schemas/request_schemas.py
"""
Request Schemas
==============

Pydantic schemas for API request validation.
"""

from datetime import datetime
from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field, validator, root_validator

from ..types import (
    TenantId, UserId, WebhookId, TenantPlan, TenantStatus,
    WebhookStatus, WebhookEventType, ComplianceLevel, DataResidency,
    BaseSchema, PaginationParams, FilterParams
)


# Tenant Schemas
class TenantCreateRequest(BaseSchema):
    """Schema for creating a tenant"""
    tenant_id: str = Field(..., min_length=1, max_length=50, description="Unique tenant identifier")
    name: str = Field(..., min_length=1, max_length=255, description="Tenant name")
    subdomain: Optional[str] = Field(None, min_length=1, max_length=100, description="Subdomain")
    plan_type: TenantPlan = Field(default=TenantPlan.STARTER, description="Subscription plan")
    billing_email: Optional[str] = Field(None, description="Billing email address")
    organization_size: Optional[str] = Field(None, description="Organization size")
    industry: Optional[str] = Field(None, description="Industry")
    country: Optional[str] = Field(None, description="Country")
    timezone: str = Field(default="UTC", description="Timezone")
    compliance_level: ComplianceLevel = Field(default=ComplianceLevel.STANDARD)
    data_residency: DataResidency = Field(default=DataResidency.US)

    @validator('subdomain')
    def validate_subdomain(cls, v):
        if v is not None:
            # Basic subdomain validation
            if not v.replace('-', '').replace('_', '').isalnum():
                raise ValueError('Subdomain must contain only alphanumeric characters, hyphens, and underscores')
            if v.startswith('-') or v.endswith('-'):
                raise ValueError('Subdomain cannot start or end with a hyphen')
        return v

    @validator('billing_email')
    def validate_email(cls, v):
        if v is not None:
            # Basic email validation
            if '@' not in v or '.' not in v.split('@')[-1]:
                raise ValueError('Invalid email format')
        return v


class TenantUpdateRequest(BaseSchema):
    """Schema for updating a tenant"""
    name: Optional[str] = Field(None, min_length=1, max_length=255)
    subdomain: Optional[str] = Field(None, min_length=1, max_length=100)
    status: Optional[TenantStatus] = None
    plan_type: Optional[TenantPlan] = None
    billing_email: Optional[str] = None
    organization_size: Optional[str] = None
    industry: Optional[str] = None
    country: Optional[str] = None
    timezone: Optional[str] = None
    compliance_level: Optional[ComplianceLevel] = None
    data_residency: Optional[DataResidency] = None
    custom_domain: Optional[str] = None
    branding_config: Optional[Dict[str, Any]] = None
    contact_info: Optional[Dict[str, Any]] = None
    feature_flags: Optional[Dict[str, Any]] = None


class TenantFilterParams(FilterParams):
    """Tenant-specific filter parameters"""
    status: Optional[TenantStatus] = None
    plan_type: Optional[TenantPlan] = None
    industry: Optional[str] = None
    compliance_level: Optional[ComplianceLevel] = None
    data_residency: Optional[DataResidency] = None
    has_custom_domain: Optional[bool] = None
    trial_expired: Optional[bool] = None


class TenantFeaturesUpdateRequest(BaseSchema):
    """Schema for updating tenant features"""
    ai_enabled: Optional[bool] = None
    ai_model: Optional[str] = None
    ai_max_tokens: Optional[int] = Field(None, ge=1, le=10000)
    channels_enabled: Optional[List[str]] = None
    whatsapp_enabled: Optional[bool] = None
    telegram_enabled: Optional[bool] = None
    slack_enabled: Optional[bool] = None
    analytics_enabled: Optional[bool] = None
    webhook_enabled: Optional[bool] = None
    custom_branding: Optional[bool] = None
    white_label: Optional[bool] = None
    priority_support: Optional[bool] = None


class TenantQuotasUpdateRequest(BaseSchema):
    """Schema for updating tenant quotas"""
    messages_per_month: Optional[int] = Field(None, ge=0)
    api_calls_per_minute: Optional[int] = Field(None, ge=1)
    api_calls_per_day: Optional[int] = Field(None, ge=1)
    storage_gb: Optional[int] = Field(None, ge=1)
    team_members: Optional[int] = Field(None, ge=1)
    integrations_limit: Optional[int] = Field(None, ge=1)
    webhooks_limit: Optional[int] = Field(None, ge=1)
    ai_tokens_per_month: Optional[int] = Field(None, ge=0)


# Webhook Schemas
class WebhookCreateRequest(BaseSchema):
    """Schema for creating a webhook"""
    webhook_id: str = Field(..., min_length=1, max_length=50, description="Unique webhook identifier")
    tenant_id: str = Field(..., min_length=1, max_length=50, description="Tenant identifier")
    name: str = Field(..., min_length=1, max_length=255, description="Webhook name")
    description: Optional[str] = Field(None, description="Webhook description")
    url: str = Field(..., min_length=1, max_length=1000, description="Webhook URL")
    secret: str = Field(..., min_length=8, max_length=255, description="Webhook secret")
    events: List[WebhookEventType] = Field(..., min_items=1, description="Event types to subscribe to")

    # Optional configuration
    event_filters: Optional[Dict[str, Any]] = Field(None, description="Event filters")
    verify_ssl: bool = Field(default=True, description="Verify SSL certificates")
    timeout_seconds: int = Field(default=30, ge=1, le=300, description="Request timeout")
    http_method: str = Field(default="POST", regex="^(GET|POST|PUT|PATCH)$")
    content_type: str = Field(default="application/json")
    custom_headers: Optional[Dict[str, str]] = Field(None, description="Custom headers")
    payload_template: Optional[str] = Field(None, description="Custom payload template")

    # Retry configuration
    max_retries: int = Field(default=3, ge=0, le=10, description="Maximum retry attempts")
    retry_delay_seconds: int = Field(default=60, ge=1, le=3600, description="Retry delay")
    exponential_backoff: bool = Field(default=True, description="Use exponential backoff")

    # Rate limiting
    rate_limit_enabled: bool = Field(default=False)
    requests_per_minute: Optional[int] = Field(None, ge=1, le=1000)
    requests_per_hour: Optional[int] = Field(None, ge=1, le=10000)

    # Metadata
    tags: Optional[List[str]] = Field(None, description="Tags for organization")

    @validator('url')
    def validate_url(cls, v):
        if not v.startswith(('http://', 'https://')):
            raise ValueError('URL must start with http:// or https://')
        return v

    @validator('events')
    def validate_events(cls, v):
        if not v:
            raise ValueError('At least one event type must be specified')
        return v

    @root_validator
    def validate_rate_limiting(cls, values):
        if values.get('rate_limit_enabled'):
            if not values.get('requests_per_minute') and not values.get('requests_per_hour'):
                raise ValueError('Rate limit values required when rate limiting is enabled')
        return values


class WebhookUpdateRequest(BaseSchema):
    """Schema for updating a webhook"""
    name: Optional[str] = Field(None, min_length=1, max_length=255)
    description: Optional[str] = None
    url: Optional[str] = Field(None, min_length=1, max_length=1000)
    secret: Optional[str] = Field(None, min_length=8, max_length=255)
    status: Optional[WebhookStatus] = None
    events: Optional[List[WebhookEventType]] = Field(None, min_items=1)
    event_filters: Optional[Dict[str, Any]] = None
    verify_ssl: Optional[bool] = None
    timeout_seconds: Optional[int] = Field(None, ge=1, le=300)
    http_method: Optional[str] = Field(None, regex="^(GET|POST|PUT|PATCH)$")
    content_type: Optional[str] = None
    custom_headers: Optional[Dict[str, str]] = None
    payload_template: Optional[str] = None
    max_retries: Optional[int] = Field(None, ge=0, le=10)
    retry_delay_seconds: Optional[int] = Field(None, ge=1, le=3600)
    exponential_backoff: Optional[bool] = None
    rate_limit_enabled: Optional[bool] = None
    requests_per_minute: Optional[int] = Field(None, ge=1, le=1000)
    requests_per_hour: Optional[int] = Field(None, ge=1, le=10000)
    tags: Optional[List[str]] = None


class WebhookFilterParams(FilterParams):
    """Webhook-specific filter parameters"""
    tenant_id: Optional[str] = None
    status: Optional[WebhookStatus] = None
    event_type: Optional[WebhookEventType] = None
    created_by: Optional[str] = None
    has_failures: Optional[bool] = None
    health_status: Optional[str] = Field(None, regex="^(healthy|unhealthy)$")


class WebhookTestRequest(BaseSchema):
    """Schema for testing a webhook"""
    event_type: WebhookEventType = Field(..., description="Event type to test")
    test_data: Optional[Dict[str, Any]] = Field(None, description="Custom test data")


# Webhook Delivery Schemas
class WebhookDeliveryCreateRequest(BaseSchema):
    """Schema for creating a webhook delivery"""
    delivery_id: str = Field(..., min_length=1, max_length=50)
    webhook_id: str = Field(..., min_length=1, max_length=50)
    tenant_id: str = Field(..., min_length=1, max_length=50)
    event_type: WebhookEventType = Field(...)
    event_data: Dict[str, Any] = Field(...)
    scheduled_at: Optional[datetime] = Field(default_factory=datetime.utcnow)
    expires_at: Optional[datetime] = None
    correlation_id: Optional[str] = None


class WebhookDeliveryFilterParams(FilterParams):
    """Webhook delivery filter parameters"""
    webhook_id: Optional[str] = None
    tenant_id: Optional[str] = None
    event_type: Optional[WebhookEventType] = None
    status: Optional[str] = None
    correlation_id: Optional[str] = None
    scheduled_after: Optional[datetime] = None
    scheduled_before: Optional[datetime] = None


# src/models/schemas/response_schemas.py
"""
Response Schemas
===============

Pydantic schemas for API response serialization.
"""

from datetime import datetime
from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field

from ..types import (
    TenantPlan, TenantStatus, WebhookStatus, WebhookEventType,
    DeliveryStatus, ComplianceLevel, DataResidency, BaseSchema
)


class PaginatedResponse(BaseSchema):
    """Generic paginated response"""
    items: List[Any] = Field(...)
    total: int = Field(..., ge=0)
    page: int = Field(..., ge=1)
    page_size: int = Field(..., ge=1)
    total_pages: int = Field(..., ge=0)
    has_next: bool = Field(...)
    has_previous: bool = Field(...)


# Tenant Response Schemas
class TenantFeaturesResponse(BaseSchema):
    """Tenant features response"""
    ai_enabled: bool
    ai_model: str
    ai_max_tokens: int
    channels_enabled: List[str]
    whatsapp_enabled: bool
    telegram_enabled: bool
    slack_enabled: bool
    analytics_enabled: bool
    webhook_enabled: bool
    custom_branding: bool
    white_label: bool
    priority_support: bool
    created_at: datetime
    updated_at: datetime


class TenantQuotasResponse(BaseSchema):
    """Tenant quotas response"""
    messages_per_month: int
    messages_used_this_month: int
    api_calls_per_minute: int
    api_calls_per_day: int
    api_calls_used_today: int
    storage_gb: int
    storage_used_gb: float
    team_members: int
    team_members_count: int
    integrations_limit: int
    integrations_count: int
    webhooks_limit: int
    webhooks_count: int
    ai_tokens_per_month: int
    ai_tokens_used_this_month: int
    monthly_reset_date: datetime
    daily_reset_date: datetime


class TenantResponse(BaseSchema):
    """Tenant response schema"""
    tenant_id: str
    name: str
    subdomain: Optional[str]
    status: TenantStatus
    plan_type: TenantPlan
    billing_email: Optional[str]
    billing_cycle: str
    trial_ends_at: Optional[datetime]
    subscription_active: bool
    compliance_level: ComplianceLevel
    data_residency: DataResidency
    custom_domain: Optional[str]
    custom_domain_verified: bool
    organization_size: Optional[str]
    industry: Optional[str]
    country: Optional[str]
    timezone: str
    onboarding_completed: bool
    created_at: datetime
    updated_at: datetime
    last_activity_at: Optional[datetime]

    # Optional relationships
    features: Optional[TenantFeaturesResponse] = None
    quotas: Optional[TenantQuotasResponse] = None


class TenantDetailResponse(TenantResponse):
    """Detailed tenant response with relationships"""
    features: TenantFeaturesResponse
    quotas: TenantQuotasResponse
    branding_config: Optional[Dict[str, Any]]
    contact_info: Optional[Dict[str, Any]]
    feature_flags: Optional[Dict[str, Any]]
    login_count: int
    last_login_at: Optional[datetime]


class TenantStatsResponse(BaseSchema):
    """Tenant statistics response"""
    total_tenants: int
    active_tenants: int
    trial_tenants: int
    suspended_tenants: int
    by_plan: Dict[str, int]
    by_industry: Dict[str, int]
    recent_signups: int
    churn_rate: float


# Webhook Response Schemas
class WebhookResponse(BaseSchema):
    """Webhook response schema"""
    webhook_id: str
    tenant_id: str
    name: str
    description: Optional[str]
    url: str
    status: WebhookStatus
    events: List[str]
    verify_ssl: bool
    timeout_seconds: int
    http_method: str
    content_type: str
    max_retries: int
    retry_delay_seconds: int
    exponential_backoff: bool
    rate_limit_enabled: bool
    requests_per_minute: Optional[int]
    requests_per_hour: Optional[int]
    total_deliveries: int
    successful_deliveries: int
    failed_deliveries: int
    consecutive_failures: int
    average_response_time_ms: Optional[float]
    last_triggered_at: Optional[datetime]
    last_success_at: Optional[datetime]
    last_failure_at: Optional[datetime]
    created_at: datetime
    updated_at: datetime
    created_by: Optional[str]
    tags: Optional[List[str]]


class WebhookDetailResponse(WebhookResponse):
    """Detailed webhook response"""
    event_filters: Optional[Dict[str, Any]]
    authentication: Optional[Dict[str, Any]]
    custom_headers: Optional[Dict[str, str]]
    payload_template: Optional[str]
    health_check_enabled: bool
    health_check_interval: Optional[int]
    last_health_check: Optional[datetime]


class WebhookStatsResponse(BaseSchema):
    """Webhook statistics response"""
    webhook_id: str
    total_deliveries: int
    successful_deliveries: int
    failed_deliveries: int
    success_rate: float
    average_response_time: Optional[float]
    last_24h_deliveries: int
    last_24h_failures: int
    health_status: str
    recent_errors: List[str]


class WebhookDeliveryResponse(BaseSchema):
    """Webhook delivery response"""
    delivery_id: str
    webhook_id: str
    tenant_id: str
    event_type: WebhookEventType
    status: DeliveryStatus
    attempt_number: int
    max_attempts: int
    request_url: str
    request_method: str
    response_status: Optional[int]
    response_time_ms: Optional[float]
    scheduled_at: datetime
    started_at: Optional[datetime]
    delivered_at: Optional[datetime]
    next_retry_at: Optional[datetime]
    error_message: Optional[str]
    error_code: Optional[str]
    error_type: Optional[str]
    correlation_id: Optional[str]
    created_at: datetime
    updated_at: datetime


class WebhookDeliveryDetailResponse(WebhookDeliveryResponse):
    """Detailed webhook delivery response"""
    event_data: Dict[str, Any]
    request_headers: Optional[Dict[str, str]]
    request_body: Optional[str]
    response_headers: Optional[Dict[str, str]]
    response_body: Optional[str]
    request_size_bytes: Optional[int]
    response_size_bytes: Optional[int]


