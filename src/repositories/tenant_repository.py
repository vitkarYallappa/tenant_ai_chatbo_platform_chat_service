# src/repositories/tenant_repository.py
"""
Tenant Repository with SQLAlchemy
=================================

Repository for tenant management using SQLAlchemy ORM with proper dependency injection.
"""

from datetime import datetime, timedelta
from typing import Optional, List, Dict, Any, Tuple, AsyncGenerator
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker
from sqlalchemy import select, update, delete, func, and_, or_, desc, asc, text
from sqlalchemy.orm import selectinload, joinedload
import structlog
from src.models.types import (
    TenantId, TenantPlan, TenantStatus, ComplianceLevel, DataResidency
)
from src.models.postgres.tenant_model import (
    Tenant, TenantFeatures, TenantQuotas,
)
from src.models.schemas.tenant_webhook_request_schemas import (
    TenantCreateRequest, TenantUpdateRequest, TenantFilterParams,
    TenantResponse, TenantDetailResponse, PaginatedResponse
)

from .base_repository import BaseRepository
from .exceptions import (
    RepositoryError, EntityNotFoundError, DuplicateEntityError
)

logger = structlog.get_logger(__name__)


class TenantRepository(BaseRepository):
    """Tenant repository using SQLAlchemy ORM with dependency injection"""

    def __init__(self, session_factory: async_sessionmaker[AsyncSession]):
        """
        Initialize tenant repository

        Args:
            session_factory: SQLAlchemy async session factory
        """
        super().__init__()
        self.session_factory = session_factory
        self.logger = structlog.get_logger("TenantRepository")

    async def _get_session(self) -> AsyncGenerator[AsyncSession, None]:
        """Get database session"""
        async with self.session_factory() as session:
            try:
                yield session
            except Exception:
                await session.rollback()
                raise
            finally:
                await session.close()

    async def create(self, tenant_request: TenantCreateRequest) -> Tenant | None:
        """
        Create a new tenant with features and quotas

        Args:
            tenant_request: Tenant creation request

        Returns:
            Created tenant with relationships

        Raises:
            DuplicateEntityError: If tenant already exists
            RepositoryError: If creation fails
        """
        try:
            async with self._timed_operation("create_tenant"):
                async for session in self._get_session():
                    # Check for existing tenant
                    existing = await session.get(Tenant, tenant_request.tenant_id)
                    if existing:
                        raise DuplicateEntityError("Tenant", tenant_request.tenant_id)

                    # Create tenant
                    tenant = Tenant(**tenant_request.model_dump())
                    session.add(tenant)

                    # Create default features
                    features = TenantFeatures(
                        tenant_id=tenant.tenant_id,
                        ai_enabled=True,
                        channels_enabled=["web"],
                        analytics_enabled=True,
                        webhook_enabled=True
                    )
                    session.add(features)

                    # Create default quotas based on plan
                    quota_config = self._get_plan_quotas(tenant.plan_type)
                    quotas = TenantQuotas(
                        tenant_id=tenant.tenant_id,
                        **quota_config
                    )
                    session.add(quotas)

                    await session.commit()
                    await session.refresh(tenant)

                    # Load relationships
                    await session.refresh(tenant, ['features', 'quotas'])

                    self._log_operation(
                        "create_tenant",
                        tenant_id=tenant.tenant_id,
                        name=tenant.name,
                        plan=tenant.plan_type.value
                    )

                    return tenant

            return None

        except DuplicateEntityError:
            raise
        except Exception as e:
            self._log_error("create_tenant", e, tenant_id=tenant_request.tenant_id)
            raise RepositoryError(f"Failed to create tenant: {e}", original_error=e)

    async def get_by_id(self, tenant_id: TenantId, include_relationships: bool = False) -> Optional[Tenant] | None:
        """
        Get tenant by ID

        Args:
            tenant_id: Tenant identifier
            include_relationships: Whether to include features and quotas

        Returns:
            Tenant if found, None otherwise
        """
        try:
            async with self._timed_operation("get_tenant"):
                async for session in self._get_session():
                    query = select(Tenant).where(
                        and_(
                            Tenant.tenant_id == tenant_id,
                            Tenant.status != TenantStatus.DELETED
                        )
                    )

                    if include_relationships:
                        query = query.options(
                            selectinload(Tenant.features),
                            selectinload(Tenant.quotas),
                            selectinload(Tenant.webhooks)
                        )

                    result = await session.execute(query)
                    tenant = result.scalar_one_or_none()

                    if tenant:
                        self._log_operation("get_tenant", tenant_id=tenant_id, found=True)
                    else:
                        self._log_operation("get_tenant", tenant_id=tenant_id, found=False)

                    return tenant

            return None

        except Exception as e:
            self._log_error("get_tenant", e, tenant_id=tenant_id)
            raise RepositoryError(f"Failed to get tenant: {e}", original_error=e)

    async def get_by_subdomain(self, subdomain: str) -> Optional[Tenant] | None:
        """Get tenant by subdomain"""
        try:
            async for session in self._get_session():
                query = select(Tenant).where(
                    and_(
                        Tenant.subdomain == subdomain,
                        Tenant.status == TenantStatus.ACTIVE
                    )
                ).options(
                    selectinload(Tenant.features),
                    selectinload(Tenant.quotas)
                )

                result = await session.execute(query)
                return result.scalar_one_or_none()

            return None

        except Exception as e:
            self._log_error("get_tenant_by_subdomain", e, subdomain=subdomain)
            raise RepositoryError(f"Failed to get tenant by subdomain: {e}", original_error=e)

    async def update(self, tenant_id: TenantId, updates: TenantUpdateRequest) -> Optional[Tenant] | None:
        """
        Update existing tenant

        Args:
            tenant_id: Tenant ID to update
            updates: Update data

        Returns:
            Updated tenant

        Raises:
            EntityNotFoundError: If tenant doesn't exist
        """
        try:
            async with self._timed_operation("update_tenant"):
                async for session in self._get_session():
                    # Get existing tenant
                    tenant = await session.get(Tenant, tenant_id)
                    if not tenant or tenant.status == TenantStatus.DELETED:
                        raise EntityNotFoundError("Tenant", tenant_id)

                    # Update fields
                    update_data = updates.model_dump(exclude_unset=True)
                    for field, value in update_data.items():
                        if hasattr(tenant, field):
                            setattr(tenant, field, value)

                    tenant.updated_at = datetime.utcnow()
                    await session.commit()
                    await session.refresh(tenant)

                    self._log_operation("update_tenant", tenant_id=tenant_id, status=tenant.status)
                    return tenant

            return None

        except EntityNotFoundError:
            raise
        except Exception as e:
            self._log_error("update_tenant", e, tenant_id=tenant_id)
            raise RepositoryError(f"Failed to update tenant: {e}", original_error=e)

    async def delete(self, tenant_id: TenantId) -> bool | None:
        """
        Soft delete tenant

        Args:
            tenant_id: Tenant ID to delete

        Returns:
            True if deleted, False if not found
        """
        try:
            async with self._timed_operation("delete_tenant"):
                async for session in self._get_session():
                    query = (
                        update(Tenant)
                        .where(
                            and_(
                                Tenant.tenant_id == tenant_id,
                                Tenant.status != TenantStatus.DELETED
                            )
                        )
                        .values(
                            status=TenantStatus.DELETED,
                            updated_at=datetime.utcnow()
                        )
                    )

                    result = await session.execute(query)
                    await session.commit()

                    deleted = result.rowcount > 0
                    self._log_operation("delete_tenant", tenant_id=tenant_id, deleted=deleted)
                    return deleted

            return None

        except Exception as e:
            self._log_error("delete_tenant", e, tenant_id=tenant_id)
            raise RepositoryError(f"Failed to delete tenant: {e}", original_error=e)

    async def list_tenants(
            self,
            filters: Optional[TenantFilterParams] = None,
            page: int = 1,
            page_size: int = 20,
            sort_by: str = "created_at",
            sort_order: str = "desc"
    ) -> Tuple[List[Tenant], int] | None:
        """
        List tenants with filters and pagination

        Args:
            filters: Filter parameters
            page: Page number (1-based)
            page_size: Items per page
            sort_by: Field to sort by
            sort_order: Sort order (asc/desc)

        Returns:
            Tuple of (tenants, total_count)
        """
        try:
            async with self._timed_operation("list_tenants"):
                async for session in self._get_session():
                    # Base query
                    query = select(Tenant).where(Tenant.status != TenantStatus.DELETED)

                    # Apply filters
                    if filters:
                        query = self._apply_tenant_filters(query, filters)

                    # Get total count
                    count_query = select(func.count()).select_from(query.subquery())
                    count_result = await session.execute(count_query)
                    total = count_result.scalar()

                    # Apply sorting
                    sort_column = getattr(Tenant, sort_by, Tenant.created_at)
                    if sort_order.lower() == "asc":
                        query = query.order_by(asc(sort_column))
                    else:
                        query = query.order_by(desc(sort_column))

                    # Apply pagination
                    offset = (page - 1) * page_size
                    query = query.offset(offset).limit(page_size)

                    # Execute query
                    result = await session.execute(query)
                    tenants = result.scalars().all()

                    self._log_operation(
                        "list_tenants",
                        total=total,
                        returned=len(tenants),
                        page=page,
                        page_size=page_size
                    )

                    return list(tenants), total

            return None

        except Exception as e:
            self._log_error("list_tenants", e, filters=filters)
            raise RepositoryError(f"Failed to list tenants: {e}", original_error=e)

    async def update_activity(self, tenant_id: TenantId) -> bool | None:
        """Update tenant last activity timestamp"""
        try:
            async for session in self._get_session():
                query = (
                    update(Tenant)
                    .where(
                        and_(
                            Tenant.tenant_id == tenant_id,
                            Tenant.status != TenantStatus.DELETED
                        )
                    )
                    .values(last_activity_at=datetime.utcnow())
                )

                result = await session.execute(query)
                await session.commit()
                return result.rowcount > 0

            return None

        except Exception as e:
            self._log_error("update_activity", e, tenant_id=tenant_id)
            return False

    async def update_subscription(
            self,
            tenant_id: TenantId,
            subscription_id: str,
            plan_type: TenantPlan,
            stripe_customer_id: Optional[str] = None
    ) -> bool | None:
        """Update tenant subscription information"""
        try:
            async for session in self._get_session():
                updates = {
                    'subscription_id': subscription_id,
                    'plan_type': plan_type,
                    'subscription_active': True,
                    'updated_at': datetime.utcnow()
                }
                
                if stripe_customer_id:
                    updates['stripe_customer_id'] = stripe_customer_id

                query = (
                    update(Tenant)
                    .where(
                        and_(
                            Tenant.tenant_id == tenant_id,
                            Tenant.status != TenantStatus.DELETED
                        )
                    )
                    .values(**updates)
                )

                result = await session.execute(query)
                await session.commit()
                return result.rowcount > 0

            return None

        except Exception as e:
            self._log_error("update_subscription", e, tenant_id=tenant_id)
            raise RepositoryError(f"Failed to update subscription: {e}", original_error=e)

    async def get_tenant_stats(self) -> Dict[str, Any] | None:
        """Get tenant statistics"""
        try:
            async for session in self._get_session():
                # Total counts by status
                status_query = (
                    select(Tenant.status, func.count())
                    .where(Tenant.status != TenantStatus.DELETED)
                    .group_by(Tenant.status)
                )
                status_result = await session.execute(status_query)
                status_counts = dict(status_result.fetchall())

                # Counts by plan
                plan_query = (
                    select(Tenant.plan_type, func.count())
                    .where(Tenant.status != TenantStatus.DELETED)
                    .group_by(Tenant.plan_type)
                )
                plan_result = await session.execute(plan_query)
                plan_counts = dict(plan_result.fetchall())

                # Recent signups (last 30 days)
                thirty_days_ago = datetime.utcnow() - timedelta(days=30)
                recent_query = (
                    select(func.count())
                    .where(
                        and_(
                            Tenant.created_at >= thirty_days_ago,
                            Tenant.status != TenantStatus.DELETED
                        )
                    )
                )
                recent_result = await session.execute(recent_query)
                recent_signups = recent_result.scalar()

                return {
                    "total_tenants": sum(status_counts.values()),
                    "active_tenants": status_counts.get(TenantStatus.ACTIVE, 0),
                    "trial_tenants": status_counts.get(TenantStatus.TRIAL, 0),
                    "suspended_tenants": status_counts.get(TenantStatus.SUSPENDED, 0),
                    "by_plan": {plan.value: count for plan, count in plan_counts.items()},
                    "recent_signups": recent_signups
                }

            return None

        except Exception as e:
            self._log_error("get_tenant_stats", e)
            raise RepositoryError(f"Failed to get tenant stats: {e}", original_error=e)

    # Helper methods
    def _get_plan_quotas(self, plan_type: TenantPlan) -> Dict[str, Any]:
        """Get default quotas for a plan type"""
        plan_configs = {
            TenantPlan.STARTER: {
                "messages_per_month": 10000,
                "api_calls_per_minute": 100,
                "api_calls_per_day": 10000,
                "storage_gb": 10,
                "team_members": 5,
                "integrations_limit": 5,
                "webhooks_limit": 10,
                "ai_tokens_per_month": 100000
            },
            TenantPlan.PROFESSIONAL: {
                "messages_per_month": 50000,
                "api_calls_per_minute": 1000,
                "api_calls_per_day": 100000,
                "storage_gb": 100,
                "team_members": 25,
                "integrations_limit": 25,
                "webhooks_limit": 50,
                "ai_tokens_per_month": 1000000
            },
            TenantPlan.ENTERPRISE: {
                "messages_per_month": 1000000,
                "api_calls_per_minute": 10000,
                "api_calls_per_day": 1000000,
                "storage_gb": 1000,
                "team_members": 100,
                "integrations_limit": 100,
                "webhooks_limit": 200,
                "ai_tokens_per_month": 10000000
            }
        }
        return plan_configs.get(plan_type, plan_configs[TenantPlan.STARTER])

    def _apply_tenant_filters(self, query, filters: TenantFilterParams):
        """Apply filters to tenant query"""
        if filters.status:
            query = query.where(Tenant.status == filters.status)
        if filters.plan_type:
            query = query.where(Tenant.plan_type == filters.plan_type)
        if filters.industry:
            query = query.where(Tenant.industry == filters.industry)
        if filters.compliance_level:
            query = query.where(Tenant.compliance_level == filters.compliance_level)
        if filters.data_residency:
            query = query.where(Tenant.data_residency == filters.data_residency)
        if filters.search:
            search_term = f"%{filters.search}%"
            query = query.where(
                or_(
                    Tenant.name.ilike(search_term),
                    Tenant.subdomain.ilike(search_term),
                    Tenant.billing_email.ilike(search_term)
                )
            )
        if filters.created_after:
            query = query.where(Tenant.created_at >= filters.created_after)
        if filters.created_before:
            query = query.where(Tenant.created_at <= filters.created_before)
        if filters.has_custom_domain is not None:
            if filters.has_custom_domain:
                query = query.where(Tenant.custom_domain.isnot(None))
            else:
                query = query.where(Tenant.custom_domain.is_(None))
        if filters.trial_expired is not None:
            now = datetime.utcnow()
            if filters.trial_expired:
                query = query.where(
                    and_(
                        Tenant.status == TenantStatus.TRIAL,
                        Tenant.trial_ends_at < now
                    )
                )
            else:
                query = query.where(
                    or_(
                        Tenant.status != TenantStatus.TRIAL,
                        Tenant.trial_ends_at >= now
                    )
                )
        return query
