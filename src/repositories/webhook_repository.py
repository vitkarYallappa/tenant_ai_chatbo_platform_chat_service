# src/repositories/webhook_repository.py
"""
Webhook Repository with SQLAlchemy
==================================

Repository for webhook management using SQLAlchemy ORM with proper dependency injection.
"""

from datetime import datetime, timedelta
from typing import Optional, List, Dict, Any, Tuple, AsyncGenerator, Coroutine
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker
from sqlalchemy import select, update, delete, func, and_, or_, desc, asc, text, Row, RowMapping
from sqlalchemy.orm import selectinload, joinedload
import structlog

from src.models.types import (
    WebhookId, TenantId, DeliveryId, TenantStatus,
    WebhookStatus, WebhookEventType, DeliveryStatus
)

from src.models.postgres.tenant_model import (
    Tenant
)
from src.models.postgres.webhook_model import (
    Webhook, WebhookDelivery,
)

from src.models.schemas.tenant_webhook_request_schemas import (
    WebhookCreateRequest, WebhookUpdateRequest, WebhookFilterParams,
    WebhookDeliveryCreateRequest, WebhookDeliveryFilterParams,
    WebhookResponse, WebhookDetailResponse, PaginatedResponse
)
from .base_repository import BaseRepository
from .exceptions import (
    RepositoryError, EntityNotFoundError, DuplicateEntityError
)

logger = structlog.get_logger(__name__)


class WebhookRepository(BaseRepository):
    """Webhook repository using SQLAlchemy ORM with dependency injection"""

    def __init__(self, session_factory: async_sessionmaker[AsyncSession]):
        """
        Initialize webhook repository

        Args:
            session_factory: SQLAlchemy async session factory
        """
        super().__init__()
        self.session_factory = session_factory
        self.logger = structlog.get_logger("WebhookRepository")

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

    async def create(self, webhook_request: WebhookCreateRequest) -> Webhook | None:
        """Create a new webhook"""
        try:
            async with self._timed_operation("create_webhook"):
                async for session in self._get_session():
                    # Check for existing webhook
                    existing = await session.get(Webhook, webhook_request.webhook_id)
                    if existing:
                        raise DuplicateEntityError("Webhook", webhook_request.webhook_id)

                    # Verify tenant exists
                    tenant = await session.get(Tenant, webhook_request.tenant_id)
                    if not tenant or tenant.status == TenantStatus.DELETED:
                        raise EntityNotFoundError("Tenant", webhook_request.tenant_id)

                    # Convert events to string list
                    webhook_data = webhook_request.model_dump()
                    webhook_data['events'] = [event.value for event in webhook_request.events]

                    # Create webhook
                    webhook = Webhook(**webhook_data)
                    session.add(webhook)

                    await session.commit()
                    await session.refresh(webhook)

                    self._log_operation(
                        "create_webhook",
                        webhook_id=webhook.webhook_id,
                        tenant_id=webhook.tenant_id,
                        url=webhook.url
                    )

                    return webhook

            return None

        except (DuplicateEntityError, EntityNotFoundError):
            raise
        except Exception as e:
            self._log_error("create_webhook", e, webhook_id=webhook_request.webhook_id)
            raise RepositoryError(f"Failed to create webhook: {e}", original_error=e)

    async def get_by_id(self, webhook_id: WebhookId, include_tenant: bool = False) -> Optional[Webhook] | None:
        """Get webhook by ID"""
        try:
            async with self._timed_operation("get_webhook"):
                async for session in self._get_session():
                    query = select(Webhook).where(Webhook.webhook_id == webhook_id)

                    if include_tenant:
                        query = query.options(selectinload(Webhook.tenant))

                    result = await session.execute(query)
                    webhook = result.scalar_one_or_none()

                    if webhook:
                        self._log_operation("get_webhook", webhook_id=webhook_id, found=True)
                    else:
                        self._log_operation("get_webhook", webhook_id=webhook_id, found=False)

                    return webhook

            return None

        except Exception as e:
            self._log_error("get_webhook", e, webhook_id=webhook_id)
            raise RepositoryError(f"Failed to get webhook: {e}", original_error=e)

    async def update(self, webhook_id: WebhookId, updates: WebhookUpdateRequest) -> Optional[Webhook]:
        """Update existing webhook"""
        try:
            async with self._timed_operation("update_webhook"):
                async for session in self._get_session():
                    webhook = await session.get(Webhook, webhook_id)
                    if not webhook:
                        raise EntityNotFoundError("Webhook", webhook_id)

                    # Update fields
                    update_data = updates.model_dump(exclude_unset=True)

                    # Convert events to string list if provided
                    if 'events' in update_data:
                        update_data['events'] = [event.value for event in updates.events]

                    for field, value in update_data.items():
                        if hasattr(webhook, field):
                            setattr(webhook, field, value)

                    webhook.updated_at = datetime.utcnow()
                    await session.commit()
                    await session.refresh(webhook)

                    self._log_operation("update_webhook", webhook_id=webhook_id)
                    return webhook


            return None

        except EntityNotFoundError:
            raise
        except Exception as e:
            self._log_error("update_webhook", e, webhook_id=webhook_id)
            raise RepositoryError(f"Failed to update webhook: {e}", original_error=e)

    async def delete(self, webhook_id: WebhookId) -> bool | None | Any:
        """Delete webhook"""
        try:
            async with self._timed_operation("delete_webhook"):
                async for session in self._get_session():
                    query = delete(Webhook).where(Webhook.webhook_id == webhook_id)
                    result = await session.execute(query)
                    await session.commit()

                    deleted = result.rowcount > 0
                    self._log_operation("delete_webhook", webhook_id=webhook_id, deleted=deleted)
                    return deleted

            return None

        except Exception as e:
            self._log_error("delete_webhook", e, webhook_id=webhook_id)
            raise RepositoryError(f"Failed to delete webhook: {e}", original_error=e)

    async def get_by_tenant(self, tenant_id: TenantId) -> list[Webhook] | None:
        """Get all webhooks for a tenant"""
        try:
            async for session in self._get_session():
                query = (
                    select(Webhook)
                    .where(Webhook.tenant_id == tenant_id)
                    .order_by(desc(Webhook.created_at))
                )

                result = await session.execute(query)
                return list(result.scalars().all())

            return None

        except Exception as e:
            self._log_error("get_webhooks_by_tenant", e, tenant_id=tenant_id)
            raise RepositoryError(f"Failed to get webhooks by tenant: {e}", original_error=e)

    async def get_active_webhooks_for_event(
            self,
            tenant_id: TenantId,
            event_type: WebhookEventType
    ) -> List[Webhook] | None:
        """Get active webhooks that subscribe to a specific event"""
        try:
            async for session in self._get_session():
                # Use PostgreSQL JSON contains operator
                query = (
                    select(Webhook)
                    .where(
                        and_(
                            Webhook.tenant_id == tenant_id,
                            Webhook.status == WebhookStatus.ACTIVE,
                            func.jsonb_exists(Webhook.events, event_type.value)
                        )
                    )
                )

                result = await session.execute(query)
                return list(result.scalars().all())

            return None

        except Exception as e:
            self._log_error("get_active_webhooks_for_event", e, tenant_id=tenant_id, event_type=event_type)
            raise RepositoryError(f"Failed to get webhooks for event: {e}", original_error=e)

    async def update_statistics(
            self,
            webhook_id: WebhookId,
            success: bool,
            response_time_ms: Optional[float] = None,
            triggered_at: Optional[datetime] = None
    ) -> bool | None:
        """Update webhook delivery statistics"""
        try:
            async for session in self._get_session():
                webhook = await session.get(Webhook, webhook_id)
                if not webhook:
                    return False

                webhook.update_statistics(success, response_time_ms)
                if triggered_at:
                    webhook.last_triggered_at = triggered_at

                await session.commit()
                return True

            return None

        except Exception as e:
            self._log_error("update_statistics", e, webhook_id=webhook_id)
            return False

    async def list_webhooks(
            self,
            filters: Optional[WebhookFilterParams] = None,
            page: int = 1,
            page_size: int = 20,
            sort_by: str = "created_at",
            sort_order: str = "desc"
    ) -> Tuple[List[Webhook], int] | None:
        """List webhooks with filters and pagination"""
        try:
            async with self._timed_operation("list_webhooks"):
                async for session in self._get_session():
                    # Base query
                    query = select(Webhook)

                    # Apply filters
                    if filters:
                        query = self._apply_webhook_filters(query, filters)

                    # Get total count
                    count_query = select(func.count()).select_from(query.subquery())
                    count_result = await session.execute(count_query)
                    total = count_result.scalar()

                    # Apply sorting
                    sort_column = getattr(Webhook, sort_by, Webhook.created_at)
                    if sort_order.lower() == "asc":
                        query = query.order_by(asc(sort_column))
                    else:
                        query = query.order_by(desc(sort_column))

                    # Apply pagination
                    offset = (page - 1) * page_size
                    query = query.offset(offset).limit(page_size)

                    # Execute query
                    result = await session.execute(query)
                    webhooks = result.scalars().all()

                    return list(webhooks), total

            return None

        except Exception as e:
            self._log_error("list_webhooks", e, filters=filters)
            raise RepositoryError(f"Failed to list webhooks: {e}", original_error=e)

    # Delivery methods
    async def create_delivery(self, delivery_request: WebhookDeliveryCreateRequest) -> WebhookDelivery | None:
        """Create a webhook delivery record"""
        try:
            async for session in self._get_session():
                delivery = WebhookDelivery(**delivery_request.model_dump())
                session.add(delivery)
                await session.commit()
                await session.refresh(delivery)
                return delivery

            return None

        except Exception as e:
            self._log_error("create_delivery", e, delivery_id=delivery_request.delivery_id)
            raise RepositoryError(f"Failed to create delivery: {e}", original_error=e)

    async def get_pending_deliveries(self, limit: int = 100) -> List[WebhookDelivery] | None:
        """Get pending webhook deliveries for processing"""
        try:
            async for session in self._get_session():
                query = (
                    select(WebhookDelivery)
                    .where(
                        and_(
                            WebhookDelivery.status.in_([DeliveryStatus.PENDING, DeliveryStatus.RETRYING]),
                            or_(
                                WebhookDelivery.scheduled_at <= datetime.utcnow(),
                                and_(
                                    WebhookDelivery.next_retry_at.isnot(None),
                                    WebhookDelivery.next_retry_at <= datetime.utcnow()
                                )
                            )
                        )
                    )
                    .order_by(asc(WebhookDelivery.scheduled_at))
                    .limit(limit)
                )

                result = await session.execute(query)
                return list(result.scalars().all())

            return None

        except Exception as e:
            self._log_error("get_pending_deliveries", e)
            raise RepositoryError(f"Failed to get pending deliveries: {e}", original_error=e)

    async def update_delivery_result(
            self,
            delivery_id: str,
            status: DeliveryStatus,
            response_status: Optional[int] = None,
            response_headers: Optional[Dict[str, str]] = None,
            response_body: Optional[str] = None,
            response_time_ms: Optional[float] = None,
            error_message: Optional[str] = None,
            error_code: Optional[str] = None,
            next_retry_at: Optional[datetime] = None
    ) -> bool | None:
        """Update delivery result"""
        try:
            async for session in self._get_session():
                delivery = await session.get(WebhookDelivery, delivery_id)
                if not delivery:
                    return False

                # Update delivery based on status
                if status == DeliveryStatus.DELIVERED:
                    delivery.mark_delivered(
                        response_status, response_headers, response_body, response_time_ms
                    )
                elif status in [DeliveryStatus.FAILED, DeliveryStatus.RETRYING]:
                    delivery.mark_failed(
                        error_message or "Unknown error",
                        error_code,
                        response_status=response_status,
                        response_headers=response_headers,
                        response_body=response_body
                    )
                    if next_retry_at:
                        delivery.next_retry_at = next_retry_at

                await session.commit()
                return True

            return None

        except Exception as e:
            self._log_error("update_delivery_result", e, delivery_id=delivery_id)
            return False

    # Helper methods
    def _apply_webhook_filters(self, query, filters: WebhookFilterParams):
        """Apply filters to webhook query"""
        if filters.tenant_id:
            query = query.where(Webhook.tenant_id == filters.tenant_id)
        if filters.status:
            query = query.where(Webhook.status == filters.status)
        if filters.event_type:
            query = query.where(
                func.jsonb_exists(Webhook.events, filters.event_type.value)
            )
        if filters.created_by:
            query = query.where(Webhook.created_by == filters.created_by)
        if filters.search:
            search_term = f"%{filters.search}%"
            query = query.where(
                or_(
                    Webhook.name.ilike(search_term),
                    Webhook.url.ilike(search_term),
                    Webhook.description.ilike(search_term)
                )
            )
        if filters.has_failures is not None:
            if filters.has_failures:
                query = query.where(Webhook.failed_deliveries > 0)
            else:
                query = query.where(Webhook.failed_deliveries == 0)
        if filters.health_status:
            if filters.health_status == "healthy":
                query = query.where(Webhook.consecutive_failures < 5)
            else:
                query = query.where(Webhook.consecutive_failures >= 5)
        return query
