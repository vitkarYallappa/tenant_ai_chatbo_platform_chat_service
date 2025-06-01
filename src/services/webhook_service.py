"""
Webhook Service Implementation
=============================

Service for webhook management, delivery processing, and event handling.
Provides comprehensive webhook lifecycle operations with delivery tracking.
"""

from datetime import datetime, timedelta
from typing import Optional, Dict, Any, List, Tuple
from uuid import uuid4
import hmac
import hashlib

from src.services.base_service import BaseService
from src.services.exceptions import (
    ServiceError, ValidationError, NotFoundError,
    ConflictError, UnauthorizedError, DeliveryError
)
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

from src.repositories.webhook_repository import WebhookRepository
from src.repositories.tenant_repository import TenantRepository
from src.services.audit_service import AuditService


class WebhookService(BaseService):
    """Service for webhook management and delivery processing"""

    def __init__(
            self,
            webhook_repo: WebhookRepository,
            tenant_repo: TenantRepository,
            audit_service: AuditService
    ):
        super().__init__()
        self.webhook_repo = webhook_repo
        self.tenant_repo = tenant_repo
        self.audit_service = audit_service

        # Business configuration
        self.max_webhooks_per_tenant = {
            "starter": 10,
            "professional": 50,
            "enterprise": 200
        }
        self.max_delivery_attempts = 5
        self.max_retry_delay_hours = 24

    async def create_webhook(
            self,
            request: WebhookCreateRequest,
            user_context: Dict[str, Any]
    ) -> WebhookResponse:
        """
        Create a new webhook with business logic validation

        Args:
            request: Webhook creation request
            user_context: User authentication context

        Returns:
            Created webhook response
        """
        try:
            await self.validate_tenant_access(request.tenant_id, user_context)

            self.log_operation(
                "create_webhook_start",
                webhook_id=request.webhook_id,
                tenant_id=request.tenant_id,
                url=request.url,
                events_count=len(request.events),
                created_by=user_context.get("user_id")
            )

            # Validate webhook creation request
            await self._validate_webhook_creation(request, user_context)

            # Check webhook limits for tenant
            await self._check_webhook_limits(request.tenant_id)

            # Verify webhook URL if requested
            if request.verify_ssl:
                await self._verify_webhook_url(request.url)

            # Create webhook
            created_webhook = await self.webhook_repo.create(request)

            # Log audit event
            await self.audit_service.log_event(
                event_type="webhook_created",
                tenant_id=request.tenant_id,
                user_id=user_context.get("user_id"),
                details={
                    "webhook_id": created_webhook.webhook_id,
                    "url": created_webhook.url,
                    "events": [event.value for event in request.events],
                    "created_by": user_context.get("user_id")
                }
            )

            self.log_operation(
                "create_webhook_complete",
                webhook_id=created_webhook.webhook_id,
                tenant_id=created_webhook.tenant_id,
                url=created_webhook.url
            )

            return WebhookResponse.model_validate(created_webhook)

        except (ValidationError, ConflictError, UnauthorizedError):
            raise
        except Exception as e:
            error = self.handle_service_error(
                e, "create_webhook",
                tenant_id=getattr(request, 'tenant_id', None),
                webhook_id=getattr(request, 'webhook_id', None)
            )
            raise error

    async def get_webhook(
            self,
            webhook_id: str,
            user_context: Dict[str, Any],
            include_tenant: bool = False
    ) -> Optional[WebhookDetailResponse]:
        """Get webhook by ID with access validation"""
        try:
            webhook = await self.webhook_repo.get_by_id(webhook_id, include_tenant)

            if not webhook:
                raise NotFoundError(f"Webhook {webhook_id} not found", "webhook", webhook_id)

            await self.validate_tenant_access(webhook.tenant_id, user_context)

            self.log_operation(
                "get_webhook",
                webhook_id=webhook_id,
                tenant_id=webhook.tenant_id,
                user_id=user_context.get("user_id")
            )

            return WebhookDetailResponse.model_validate(webhook)

        except (NotFoundError, UnauthorizedError):
            raise
        except Exception as e:
            error = self.handle_service_error(e, "get_webhook", webhook_id=webhook_id)
            raise error

    async def update_webhook(
            self,
            webhook_id: str,
            updates: WebhookUpdateRequest,
            user_context: Dict[str, Any]
    ) -> WebhookResponse:
        """Update webhook with business logic validation"""
        try:
            # Get existing webhook
            existing_webhook = await self.webhook_repo.get_by_id(webhook_id)
            if not existing_webhook:
                raise NotFoundError(f"Webhook {webhook_id} not found", "webhook", webhook_id)

            await self.validate_tenant_access(existing_webhook.tenant_id, user_context)

            # Validate updates
            await self._validate_webhook_updates(existing_webhook, updates, user_context)

            # Process updates with business logic
            processed_updates = await self._process_webhook_updates(
                existing_webhook, updates, user_context
            )

            # Update webhook
            updated_webhook = await self.webhook_repo.update(webhook_id, processed_updates)

            # Log audit event
            await self.audit_service.log_event(
                event_type="webhook_updated",
                tenant_id=existing_webhook.tenant_id,
                user_id=user_context.get("user_id"),
                details={
                    "webhook_id": webhook_id,
                    "updates": updates.model_dump(exclude_unset=True),
                    "updated_by": user_context.get("user_id")
                }
            )

            self.log_operation(
                "update_webhook",
                webhook_id=webhook_id,
                tenant_id=existing_webhook.tenant_id,
                updates_count=len(updates.model_dump(exclude_unset=True))
            )

            return WebhookResponse.model_validate(updated_webhook)

        except (NotFoundError, UnauthorizedError, ValidationError):
            raise
        except Exception as e:
            error = self.handle_service_error(e, "update_webhook", webhook_id=webhook_id)
            raise error

    async def delete_webhook(
            self,
            webhook_id: str,
            user_context: Dict[str, Any]
    ) -> bool:
        """Delete webhook with proper cleanup"""
        try:
            # Get existing webhook
            existing_webhook = await self.webhook_repo.get_by_id(webhook_id)
            if not existing_webhook:
                raise NotFoundError(f"Webhook {webhook_id} not found", "webhook", webhook_id)

            await self.validate_tenant_access(existing_webhook.tenant_id, user_context)

            # Cancel any pending deliveries
            await self._cancel_pending_deliveries(webhook_id)

            # Delete webhook
            success = await self.webhook_repo.delete(webhook_id)

            if success:
                # Log audit event
                await self.audit_service.log_event(
                    event_type="webhook_deleted",
                    tenant_id=existing_webhook.tenant_id,
                    user_id=user_context.get("user_id"),
                    details={
                        "webhook_id": webhook_id,
                        "url": existing_webhook.url,
                        "deleted_by": user_context.get("user_id")
                    }
                )

                self.log_operation(
                    "delete_webhook",
                    webhook_id=webhook_id,
                    tenant_id=existing_webhook.tenant_id,
                    deleted_by=user_context.get("user_id")
                )

            return success

        except (NotFoundError, UnauthorizedError):
            raise
        except Exception as e:
            error = self.handle_service_error(e, "delete_webhook", webhook_id=webhook_id)
            raise error

    async def list_webhooks(
            self,
            filters: Optional[WebhookFilterParams] = None,
            page: int = 1,
            page_size: int = 20,
            sort_by: str = "created_at",
            sort_order: str = "desc",
            user_context: Dict[str, Any] = None
    ) -> Tuple[List[WebhookResponse], int]:
        """List webhooks with filtering and pagination"""
        try:
            # Apply user-specific filters
            processed_filters = await self._apply_user_webhook_filters(filters, user_context)

            webhooks, total = await self.webhook_repo.list_webhooks(
                filters=processed_filters,
                page=page,
                page_size=page_size,
                sort_by=sort_by,
                sort_order=sort_order
            )

            webhook_responses = [
                WebhookResponse.model_validate(webhook) for webhook in webhooks
            ]

            self.log_operation(
                "list_webhooks",
                total=total,
                returned=len(webhook_responses),
                page=page,
                page_size=page_size,
                user_id=user_context.get("user_id") if user_context else None
            )

            return webhook_responses, total

        except Exception as e:
            error = self.handle_service_error(e, "list_webhooks")
            raise error

    async def trigger_webhook(
            self,
            tenant_id: str,
            event_type: WebhookEventType,
            event_data: Dict[str, Any],
            user_context: Optional[Dict[str, Any]] = None
    ) -> List[str]:
        """
        Trigger webhooks for a specific event

        Args:
            tenant_id: Tenant identifier
            event_type: Type of event that occurred
            event_data: Event payload data
            user_context: Optional user context

        Returns:
            List of delivery IDs created
        """
        try:
            if user_context:
                await self.validate_tenant_access(tenant_id, user_context)

            self.log_operation(
                "trigger_webhook_start",
                tenant_id=tenant_id,
                event_type=event_type.value,
                user_id=user_context.get("user_id") if user_context else None
            )

            # Get active webhooks for this event
            webhooks = await self.webhook_repo.get_active_webhooks_for_event(
                tenant_id, event_type
            )

            if not webhooks:
                self.log_operation(
                    "trigger_webhook_no_subscribers",
                    tenant_id=tenant_id,
                    event_type=event_type.value
                )
                return []

            # Create deliveries for each webhook
            delivery_ids = []
            for webhook in webhooks:
                # Check event filters
                if await self._webhook_matches_event(webhook, event_data):
                    delivery_id = await self._create_webhook_delivery(
                        webhook, event_type, event_data
                    )
                    delivery_ids.append(delivery_id)

            self.log_operation(
                "trigger_webhook_complete",
                tenant_id=tenant_id,
                event_type=event_type.value,
                webhooks_count=len(webhooks),
                deliveries_created=len(delivery_ids)
            )

            return delivery_ids

        except UnauthorizedError:
            raise
        except Exception as e:
            error = self.handle_service_error(
                e, "trigger_webhook",
                tenant_id=tenant_id,
                event_type=event_type.value if event_type else None
            )
            raise error

    async def process_webhook_deliveries(self, limit: int = 100) -> int:
        """
        Process pending webhook deliveries (background task)

        Args:
            limit: Maximum number of deliveries to process

        Returns:
            Number of deliveries processed
        """
        try:
            # Get pending deliveries
            pending_deliveries = await self.webhook_repo.get_pending_deliveries(limit)

            if not pending_deliveries:
                return 0

            processed_count = 0
            for delivery in pending_deliveries:
                try:
                    await self._process_single_delivery(delivery)
                    processed_count += 1
                except Exception as e:
                    self.logger.error(
                        "Failed to process webhook delivery",
                        delivery_id=delivery.delivery_id,
                        webhook_id=delivery.webhook_id,
                        error=str(e)
                    )

            self.log_operation(
                "process_webhook_deliveries",
                pending_count=len(pending_deliveries),
                processed_count=processed_count
            )

            return processed_count

        except Exception as e:
            error = self.handle_service_error(e, "process_webhook_deliveries")
            raise error

    async def test_webhook(
            self,
            webhook_id: str,
            test_event: Optional[WebhookEventType] = None,
            user_context: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """
        Test webhook delivery with a sample event

        Args:
            webhook_id: Webhook to test
            test_event: Optional specific event to test
            user_context: User context

        Returns:
            Test result details
        """
        try:
            # Get webhook
            webhook = await self.webhook_repo.get_by_id(webhook_id)
            if not webhook:
                raise NotFoundError(f"Webhook {webhook_id} not found", "webhook", webhook_id)

            if user_context:
                await self.validate_tenant_access(webhook.tenant_id, user_context)

            # Determine test event
            if not test_event:
                test_event = webhook.get_event_types()[
                    0] if webhook.get_event_types() else WebhookEventType.MESSAGE_RECEIVED

            # Create test payload
            test_payload = await self._create_test_payload(test_event, webhook.tenant_id)

            # Create and process test delivery
            delivery_id = await self._create_webhook_delivery(
                webhook, test_event, test_payload, is_test=True
            )

            # Get delivery for immediate processing
            test_delivery = await self.webhook_repo.get_delivery_by_id(delivery_id)
            if test_delivery:
                await self._process_single_delivery(test_delivery)

                # Get updated delivery status
                updated_delivery = await self.webhook_repo.get_delivery_by_id(delivery_id)

                test_result = {
                    "delivery_id": delivery_id,
                    "status": updated_delivery.status.value if updated_delivery else "unknown",
                    "response_status": updated_delivery.response_status if updated_delivery else None,
                    "response_time_ms": updated_delivery.response_time_ms if updated_delivery else None,
                    "error_message": updated_delivery.error_message if updated_delivery else None,
                    "test_payload": test_payload
                }
            else:
                test_result = {
                    "delivery_id": delivery_id,
                    "status": "created",
                    "test_payload": test_payload
                }

            self.log_operation(
                "test_webhook",
                webhook_id=webhook_id,
                tenant_id=webhook.tenant_id,
                test_event=test_event.value,
                result_status=test_result["status"]
            )

            return test_result

        except (NotFoundError, UnauthorizedError):
            raise
        except Exception as e:
            error = self.handle_service_error(e, "test_webhook", webhook_id=webhook_id)
            raise error

    # Private helper methods
    async def _validate_webhook_creation(
            self,
            request: WebhookCreateRequest,
            user_context: Dict[str, Any]
    ) -> None:
        """Validate webhook creation request"""
        # Check URL format
        if not request.url.startswith(('http://', 'https://')):
            raise ValidationError("Webhook URL must use HTTP or HTTPS protocol", "url")

        # Validate events
        if not request.events:
            raise ValidationError("At least one event type must be specified", "events")

        # Validate secret strength
        if len(request.secret) < 8:
            raise ValidationError("Webhook secret must be at least 8 characters", "secret")

    async def _check_webhook_limits(self, tenant_id: str) -> None:
        """Check webhook limits for tenant"""
        # Get tenant to check plan
        tenant = await self.tenant_repo.get_by_id(tenant_id, include_relationships=True)
        if not tenant:
            raise NotFoundError(f"Tenant {tenant_id} not found", "tenant", tenant_id)

        # Get current webhook count
        existing_webhooks = await self.webhook_repo.get_by_tenant(tenant_id)
        current_count = len(existing_webhooks)

        # Check limit based on plan
        plan_name = tenant.plan_type.value
        max_webhooks = self.max_webhooks_per_tenant.get(plan_name, 10)

        if current_count >= max_webhooks:
            raise ValidationError(
                f"Webhook limit reached for {plan_name} plan ({max_webhooks} webhooks maximum)",
                "webhook_limit"
            )

    async def _verify_webhook_url(self, url: str) -> None:
        """Verify webhook URL is accessible (simplified implementation)"""
        # In production, this would make an HTTP request to verify the endpoint
        # For now, just basic validation
        if "localhost" in url or "127.0.0.1" in url:
            raise ValidationError("Webhook URL cannot point to localhost", "url")

    async def _validate_webhook_updates(
            self,
            existing_webhook: Webhook,
            updates: WebhookUpdateRequest,
            user_context: Dict[str, Any]
    ) -> None:
        """Validate webhook update request"""
        # Validate URL if being updated
        if updates.url and updates.url != existing_webhook.url:
            if not updates.url.startswith(('http://', 'https://')):
                raise ValidationError("Webhook URL must use HTTP or HTTPS protocol", "url")

        # Validate events if being updated
        if updates.events is not None and not updates.events:
            raise ValidationError("At least one event type must be specified", "events")

    async def _process_webhook_updates(
            self,
            existing_webhook: Webhook,
            updates: WebhookUpdateRequest,
            user_context: Dict[str, Any]
    ) -> WebhookUpdateRequest:
        """Process webhook updates with business logic"""
        # Reset failure counters if webhook is being reactivated
        if updates.status == WebhookStatus.ACTIVE and existing_webhook.status != WebhookStatus.ACTIVE:
            # This would reset consecutive failures, etc.
            pass

        return updates

    async def _cancel_pending_deliveries(self, webhook_id: str) -> None:
        """Cancel pending deliveries for webhook"""
        # This would update all pending deliveries for the webhook to cancelled status
        pass

    async def _apply_user_webhook_filters(
            self,
            filters: Optional[WebhookFilterParams],
            user_context: Dict[str, Any]
    ) -> Optional[WebhookFilterParams]:
        """Apply user-specific webhook filters"""
        if not user_context:
            return filters

        user_role = user_context.get("role", "user")
        user_tenant_id = user_context.get("tenant_id")

        # Non-admin users can only see their tenant's webhooks
        if user_role != "admin" and user_tenant_id:
            if not filters:
                filters = WebhookFilterParams()
            filters.tenant_id = user_tenant_id

        return filters

    async def _webhook_matches_event(
            self,
            webhook: Webhook,
            event_data: Dict[str, Any]
    ) -> bool:
        """Check if webhook event filters match the event data"""
        if not webhook.event_filters:
            return True

        # Apply filter logic based on webhook.event_filters
        # This would implement complex filtering logic
        return True

    async def _create_webhook_delivery(
            self,
            webhook: Webhook,
            event_type: WebhookEventType,
            event_data: Dict[str, Any],
            is_test: bool = False
    ) -> str:
        """Create a webhook delivery record"""
        delivery_id = str(uuid4())

        delivery_request = WebhookDeliveryCreateRequest(
            delivery_id=delivery_id,
            webhook_id=webhook.webhook_id,
            tenant_id=webhook.tenant_id,
            event_type=event_type,
            event_data=event_data,
            scheduled_at=datetime.utcnow(),
            correlation_id=f"test_{delivery_id}" if is_test else None
        )

        delivery = await self.webhook_repo.create_delivery(delivery_request)
        return delivery.delivery_id

    async def _process_single_delivery(self, delivery: WebhookDelivery) -> None:
        """Process a single webhook delivery"""
        try:
            # Get webhook details
            webhook = await self.webhook_repo.get_by_id(delivery.webhook_id)
            if not webhook:
                await self.webhook_repo.update_delivery_result(
                    delivery.delivery_id,
                    DeliveryStatus.FAILED,
                    error_message="Webhook not found"
                )
                return

            # Prepare payload
            payload = await self._prepare_webhook_payload(webhook, delivery)

            # Sign payload
            signature = self._generate_webhook_signature(payload, webhook.secret)

            # Make HTTP request (simplified)
            delivery_successful = await self._deliver_webhook(
                webhook.url, payload, signature, webhook.timeout_seconds
            )

            if delivery_successful:
                # Update delivery as successful
                await self.webhook_repo.update_delivery_result(
                    delivery.delivery_id,
                    DeliveryStatus.DELIVERED,
                    response_status=200,
                    response_time_ms=100  # Mock response time
                )

                # Update webhook statistics
                await self.webhook_repo.update_statistics(
                    webhook.webhook_id, True, 100
                )
            else:
                # Handle failure
                await self._handle_delivery_failure(delivery, webhook)

        except Exception as e:
            await self._handle_delivery_failure(delivery, None, str(e))

    async def _prepare_webhook_payload(
            self,
            webhook: Webhook,
            delivery: WebhookDelivery
    ) -> str:
        """Prepare webhook payload"""
        payload_data = {
            "event_type": delivery.event_type.value,
            "event_id": delivery.delivery_id,
            "timestamp": delivery.scheduled_at.isoformat(),
            "tenant_id": delivery.tenant_id,
            "data": delivery.event_data
        }

        # Apply payload template if configured
        if webhook.payload_template:
            # Apply template transformation
            pass

        return str(payload_data)  # Would be JSON in practice

    def _generate_webhook_signature(self, payload: str, secret: str) -> str:
        """Generate webhook signature for verification"""
        return hmac.new(
            secret.encode('utf-8'),
            payload.encode('utf-8'),
            hashlib.sha256
        ).hexdigest()

    async def _deliver_webhook(
            self,
            url: str,
            payload: str,
            signature: str,
            timeout_seconds: int
    ) -> bool:
        """Deliver webhook (simplified implementation)"""
        # In production, this would make actual HTTP request
        # For now, return success simulation
        return True

    async def _handle_delivery_failure(
            self,
            delivery: WebhookDelivery,
            webhook: Optional[Webhook],
            error_message: str = "Delivery failed"
    ) -> None:
        """Handle delivery failure with retry logic"""
        # Check if can retry
        if delivery.can_retry():
            # Calculate next retry time
            if webhook:
                delay_seconds = webhook.calculate_next_retry_delay(delivery.attempt_number + 1)
            else:
                delay_seconds = 300  # 5 minutes default

            next_retry = datetime.utcnow() + timedelta(seconds=delay_seconds)

            await self.webhook_repo.update_delivery_result(
                delivery.delivery_id,
                DeliveryStatus.RETRYING,
                error_message=error_message,
                next_retry_at=next_retry
            )
        else:
            # Mark as permanently failed
            await self.webhook_repo.update_delivery_result(
                delivery.delivery_id,
                DeliveryStatus.FAILED,
                error_message=error_message
            )

            # Update webhook statistics
            if webhook:
                await self.webhook_repo.update_statistics(
                    webhook.webhook_id, False
                )

    async def _create_test_payload(
            self,
            event_type: WebhookEventType,
            tenant_id: str
    ) -> Dict[str, Any]:
        """Create test payload for webhook testing"""
        base_payload = {
            "test": True,
            "timestamp": datetime.utcnow().isoformat(),
            "tenant_id": tenant_id
        }

        # Add event-specific test data
        if event_type == WebhookEventType.MESSAGE_RECEIVED:
            base_payload.update({
                "message_id": "test_message_123",
                "conversation_id": "test_conversation_456",
                "user_id": "test_user_789",
                "content": {
                    "type": "text",
                    "text": "This is a test message for webhook testing"
                }
            })
        elif event_type == WebhookEventType.CONVERSATION_STARTED:
            base_payload.update({
                "conversation_id": "test_conversation_456",
                "user_id": "test_user_789",
                "channel": "web"
            })

        return base_payload

