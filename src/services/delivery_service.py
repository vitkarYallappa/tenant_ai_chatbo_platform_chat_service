"""
Delivery Service

Manages message delivery tracking, retries, and delivery status updates.
Handles failed deliveries, delivery confirmations, and delivery analytics.
"""

from datetime import datetime, timedelta, UTC
from typing import Dict, Any, Optional, List
from uuid import uuid4
import asyncio

from src.services.base_service import BaseService
from src.services.exceptions import ServiceError, ValidationError, DeliveryError, CircuitBreakerError
from src.models.types import TenantId, ChannelType, MessageContent, DeliveryStatus
from src.models.mongo.message_model import MessageDocument
from src.repositories.message_repository import MessageRepository
from src.core.channels.channel_factory import ChannelFactory


class DeliveryService(BaseService):
    """Service for managing message delivery and retry logic"""

    def __init__(
            self,
            message_repo: MessageRepository,
            channel_factory: ChannelFactory
    ):
        super().__init__()
        self.message_repo = message_repo
        self.channel_factory = channel_factory
        self.max_retry_attempts = 3
        self.retry_backoff_factor = 2.0
        self.delivery_timeout_seconds = 30
        self._circuit_breakers = {}

    async def deliver_message(
            self,
            tenant_id: TenantId,
            message_id: str,
            channel_type: ChannelType,
            recipient: str,
            content: MessageContent,
            metadata: Optional[Dict[str, Any]] = None,
            priority: str = "normal",
            user_context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Deliver message with retry logic and tracking

        Args:
            tenant_id: Tenant identifier
            message_id: Message identifier
            channel_type: Delivery channel
            recipient: Message recipient
            content: Message content
            metadata: Optional delivery metadata
            priority: Delivery priority (low, normal, high, urgent)
            user_context: User authentication context

        Returns:
            Delivery result with status and tracking info
        """
        try:
            await self.validate_tenant_access(tenant_id, user_context)

            delivery_id = str(uuid4())
            delivery_attempt = 1

            # Check circuit breaker for channel
            if not await self._check_circuit_breaker(channel_type, tenant_id):
                raise CircuitBreakerError(
                    f"Circuit breaker open for channel {channel_type.value}"
                )

            # Create delivery record
            delivery_record = {
                "delivery_id": delivery_id,
                "tenant_id": tenant_id,
                "message_id": message_id,
                "channel_type": channel_type.value,
                "recipient": recipient,
                "priority": priority,
                "status": DeliveryStatus.SENT,
                "attempts": [],
                "created_at": datetime.utcnow(),
                "metadata": metadata or {}
            }

            # Attempt delivery with retries
            delivery_result = await self._attempt_delivery_with_retries(
                delivery_record, content, channel_type, tenant_id
            )

            # Update message with delivery status
            await self._update_message_delivery_status(
                message_id, delivery_result
            )

            self.log_operation(
                "deliver_message",
                tenant_id=tenant_id,
                message_id=message_id,
                delivery_id=delivery_id,
                channel=channel_type.value,
                success=delivery_result["success"],
                attempts=len(delivery_result["attempts"])
            )

            return delivery_result

        except Exception as e:
            error = self.handle_service_error(
                e, "deliver_message",
                tenant_id=tenant_id,
                message_id=message_id
            )
            raise error

    async def _attempt_delivery_with_retries(
            self,
            delivery_record: Dict[str, Any],
            content: MessageContent,
            channel_type: ChannelType,
            tenant_id: TenantId
    ) -> Dict[str, Any]:
        """Attempt delivery with exponential backoff retry logic"""
        attempts = []
        last_error = None

        for attempt in range(1, self.max_retry_attempts + 1):
            attempt_start = datetime.utcnow()

            try:
                # Get channel implementation
                channel = await self.channel_factory.get_channel(channel_type)

                # Attempt delivery
                channel_response = await asyncio.wait_for(
                    channel.send_message(
                        recipient=delivery_record["recipient"],
                        content=content,
                        metadata=delivery_record["metadata"]
                    ),
                    timeout=self.delivery_timeout_seconds
                )

                # Record successful attempt
                attempt_record = {
                    "attempt_number": attempt,
                    "started_at": attempt_start.isoformat(),
                    "completed_at": datetime.utcnow().isoformat(),
                    "success": channel_response.success,
                    "channel_message_id": channel_response.message_id,
                    "delivery_status": channel_response.delivery_status,
                    "response_metadata": channel_response.dict()
                }
                attempts.append(attempt_record)

                if channel_response.success:
                    # Update circuit breaker on success
                    await self._record_circuit_breaker_success(channel_type, tenant_id)

                    return {
                        "delivery_id": delivery_record["delivery_id"],
                        "success": True,
                        "status": DeliveryStatus.DELIVERED,
                        "channel_message_id": channel_response.message_id,
                        "delivered_at": datetime.utcnow().isoformat(),
                        "attempts": attempts,
                        "total_attempts": attempt
                    }
                else:
                    last_error = f"Channel delivery failed: {channel_response.error}"

            except asyncio.TimeoutError:
                last_error = f"Delivery timeout after {self.delivery_timeout_seconds} seconds"
            except Exception as e:
                last_error = str(e)

            # Record failed attempt
            attempt_record = {
                "attempt_number": attempt,
                "started_at": attempt_start.isoformat(),
                "completed_at": datetime.utcnow().isoformat(),
                "success": False,
                "error": last_error
            }
            attempts.append(attempt_record)

            # Record circuit breaker failure
            await self._record_circuit_breaker_failure(channel_type, tenant_id)

            # If not the last attempt, wait before retrying
            if attempt < self.max_retry_attempts:
                delay = self.retry_backoff_factor ** (attempt - 1)
                await asyncio.sleep(delay)

        # All attempts failed
        return {
            "delivery_id": delivery_record["delivery_id"],
            "success": False,
            "status": DeliveryStatus.FAILED,
            "error": last_error,
            "failed_at": datetime.utcnow().isoformat(),
            "attempts": attempts,
            "total_attempts": self.max_retry_attempts
        }

    async def get_delivery_status(
            self,
            tenant_id: TenantId,
            delivery_id: str,
            user_context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Get delivery status for a specific delivery

        Args:
            tenant_id: Tenant identifier
            delivery_id: Delivery identifier
            user_context: User authentication context

        Returns:
            Delivery status information
        """
        try:
            await self.validate_tenant_access(tenant_id, user_context)

            # In a real implementation, this would query a delivery tracking database
            # For now, we'll simulate the response

            self.log_operation(
                "get_delivery_status",
                tenant_id=tenant_id,
                delivery_id=delivery_id
            )

            # TODO: Implement actual delivery status retrieval
            return {
                "delivery_id": delivery_id,
                "status": "delivered",
                "message": "Delivery status retrieved successfully"
            }

        except Exception as e:
            error = self.handle_service_error(
                e, "get_delivery_status",
                tenant_id=tenant_id
            )
            raise error

    async def retry_failed_delivery(
            self,
            tenant_id: TenantId,
            message_id: str,
            user_context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Retry a failed message delivery

        Args:
            tenant_id: Tenant identifier
            message_id: Message identifier to retry
            user_context: User authentication context

        Returns:
            Retry result
        """
        try:
            await self.validate_tenant_access(tenant_id, user_context)

            # Get the failed message
            message = await self.message_repo.get_by_id(message_id)
            if not message:
                raise ValidationError(f"Message {message_id} not found")

            if message.tenant_id != tenant_id:
                raise ValidationError(f"Message does not belong to tenant {tenant_id}")

            # Check if message actually failed delivery
            delivery_status = message.channel_metadata.get("delivery_status")
            if delivery_status != DeliveryStatus.FAILED:
                raise ValidationError(
                    f"Message delivery status is {delivery_status}, cannot retry"
                )

            # Extract delivery information from message
            content = MessageContent(**message.content)
            channel_type = ChannelType(message.channel)
            recipient = message.channel_metadata.get("recipient") or message.user_id

            # Attempt redelivery
            retry_result = await self.deliver_message(
                tenant_id=tenant_id,
                message_id=message_id,
                channel_type=channel_type,
                recipient=recipient,
                content=content,
                metadata=message.channel_metadata,
                priority="high",  # Higher priority for retries
                user_context=user_context
            )

            self.log_operation(
                "retry_failed_delivery",
                tenant_id=tenant_id,
                message_id=message_id,
                retry_success=retry_result["success"]
            )

            return retry_result

        except Exception as e:
            error = self.handle_service_error(
                e, "retry_failed_delivery",
                tenant_id=tenant_id,
                message_id=message_id
            )
            raise error

    async def get_delivery_metrics(
            self,
            tenant_id: TenantId,
            start_date: Optional[datetime] = None,
            end_date: Optional[datetime] = None,
            channel_type: Optional[ChannelType] = None,
            user_context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Get delivery metrics for a tenant

        Args:
            tenant_id: Tenant identifier
            start_date: Optional start date for metrics
            end_date: Optional end date for metrics
            channel_type: Optional channel filter
            user_context: User authentication context

        Returns:
            Delivery metrics data
        """
        try:
            await self.validate_tenant_access(tenant_id, user_context)

            # Set default date range if not provided
            if not end_date:
                end_date = datetime.now(UTC)
            if not start_date:
                start_date = end_date - timedelta(days=7)

            # Build filters
            filters = {
                "tenant_id": tenant_id,
                "timestamp": {
                    "$gte": start_date,
                    "$lte": end_date
                }
            }

            if channel_type:
                filters["channel"] = channel_type.value

            # Get delivery statistics from message repository
            metrics = await self.message_repo.get_delivery_metrics(filters)

            self.log_operation(
                "get_delivery_metrics",
                tenant_id=tenant_id,
                start_date=start_date.isoformat(),
                end_date=end_date.isoformat(),
                channel=channel_type.value if channel_type else "all"
            )

            return metrics

        except Exception as e:
            error = self.handle_service_error(
                e, "get_delivery_metrics",
                tenant_id=tenant_id
            )
            raise error

    async def process_delivery_webhook(
            self,
            tenant_id: TenantId,
            channel_type: ChannelType,
            webhook_data: Dict[str, Any],
            user_context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Process delivery status webhook from channel

        Args:
            tenant_id: Tenant identifier
            channel_type: Channel that sent the webhook
            webhook_data: Webhook payload
            user_context: User authentication context

        Returns:
            Processing result
        """
        try:
            await self.validate_tenant_access(tenant_id, user_context)

            # Extract delivery information from webhook
            channel_message_id = webhook_data.get("message_id")
            delivery_status = webhook_data.get("status")
            timestamp = webhook_data.get("timestamp")

            if not channel_message_id or not delivery_status:
                raise ValidationError("Invalid webhook data: missing message_id or status")

            # Find message by channel message ID
            message = await self.message_repo.get_by_channel_message_id(
                tenant_id, channel_message_id
            )

            if message:
                # Update delivery status
                await self._update_message_delivery_status(
                    message.message_id,
                    {
                        "status": delivery_status,
                        "delivered_at": timestamp,
                        "webhook_received_at": datetime.utcnow().isoformat()
                    }
                )

            self.log_operation(
                "process_delivery_webhook",
                tenant_id=tenant_id,
                channel=channel_type.value,
                channel_message_id=channel_message_id,
                delivery_status=delivery_status,
                message_found=bool(message)
            )

            return {
                "processed": True,
                "message_found": bool(message),
                "updated_status": delivery_status
            }

        except Exception as e:
            error = self.handle_service_error(
                e, "process_delivery_webhook",
                tenant_id=tenant_id
            )
            raise error

    async def _update_message_delivery_status(
            self,
            message_id: str,
            delivery_result: Dict[str, Any]
    ) -> None:
        """Update message with delivery status"""
        try:
            message = await self.message_repo.get_by_id(message_id)
            if message:
                # Update channel metadata with delivery information
                message.channel_metadata.update({
                    "delivery_status": delivery_result.get("status"),
                    "delivery_attempts": delivery_result.get("total_attempts"),
                    "delivered_at": delivery_result.get("delivered_at"),
                    "failed_at": delivery_result.get("failed_at"),
                    "delivery_error": delivery_result.get("error")
                })

                await self.message_repo.update(message)

        except Exception as e:
            self.logger.error(
                "Failed to update message delivery status",
                message_id=message_id,
                error=str(e)
            )

    async def _check_circuit_breaker(
            self,
            channel_type: ChannelType,
            tenant_id: TenantId
    ) -> bool:
        """Check if circuit breaker allows requests"""
        try:
            breaker_key = f"{tenant_id}:{channel_type.value}"
            breaker = self._circuit_breakers.get(breaker_key, {
                "state": "closed",
                "failure_count": 0,
                "last_failure_time": None,
                "next_attempt_time": None
            })

            now = datetime.utcnow()

            # If circuit is open, check if we should try again
            if breaker["state"] == "open":
                if breaker["next_attempt_time"] and now >= breaker["next_attempt_time"]:
                    breaker["state"] = "half_open"
                    self._circuit_breakers[breaker_key] = breaker
                    return True
                else:
                    return False

            # Circuit is closed or half-open, allow request
            return True

        except Exception as e:
            self.logger.error(
                "Circuit breaker check failed",
                channel=channel_type.value,
                tenant_id=tenant_id,
                error=str(e)
            )
            return True  # Fail open

    async def _record_circuit_breaker_failure(
            self,
            channel_type: ChannelType,
            tenant_id: TenantId
    ) -> None:
        """Record a failure for circuit breaker tracking"""
        try:
            breaker_key = f"{tenant_id}:{channel_type.value}"
            breaker = self._circuit_breakers.get(breaker_key, {
                "state": "closed",
                "failure_count": 0,
                "last_failure_time": None,
                "next_attempt_time": None
            })

            breaker["failure_count"] += 1
            breaker["last_failure_time"] = datetime.utcnow()

            # Open circuit if failure threshold exceeded
            failure_threshold = 5
            if breaker["failure_count"] >= failure_threshold:
                breaker["state"] = "open"
                # Try again in 1 minute
                breaker["next_attempt_time"] = datetime.utcnow() + timedelta(minutes=1)

            self._circuit_breakers[breaker_key] = breaker

        except Exception as e:
            self.logger.error(
                "Failed to record circuit breaker failure",
                channel=channel_type.value,
                tenant_id=tenant_id,
                error=str(e)
            )

    async def _record_circuit_breaker_success(
            self,
            channel_type: ChannelType,
            tenant_id: TenantId
    ) -> None:
        """Record a success for circuit breaker tracking"""
        try:
            breaker_key = f"{tenant_id}:{channel_type.value}"

            if breaker_key in self._circuit_breakers:
                breaker = self._circuit_breakers[breaker_key]

                # Reset circuit breaker on success
                if breaker["state"] == "half_open":
                    breaker["state"] = "closed"
                    breaker["failure_count"] = 0
                    breaker["last_failure_time"] = None
                    breaker["next_attempt_time"] = None

                    self._circuit_breakers[breaker_key] = breaker

        except Exception as e:
            self.logger.error(
                "Failed to record circuit breaker success",
                channel=channel_type.value,
                tenant_id=tenant_id,
                error=str(e)
            )

    async def get_failed_deliveries(
            self,
            tenant_id: TenantId,
            limit: int = 50,
            offset: int = 0,
            user_context: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """
        Get list of failed deliveries for retry processing

        Args:
            tenant_id: Tenant identifier
            limit: Maximum number of results
            offset: Number of results to skip
            user_context: User authentication context

        Returns:
            List of failed delivery records
        """
        try:
            await self.validate_tenant_access(tenant_id, user_context)

            # Get failed messages from repository
            filters = {
                "tenant_id": tenant_id,
                "channel_metadata.delivery_status": DeliveryStatus.FAILED
            }

            failed_messages = await self.message_repo.find_messages(
                filters, limit, offset
            )

            failed_deliveries = []
            for message in failed_messages:
                delivery_info = {
                    "message_id": message.message_id,
                    "conversation_id": message.conversation_id,
                    "channel": message.channel,
                    "recipient": message.user_id,
                    "failed_at": message.channel_metadata.get("failed_at"),
                    "error": message.channel_metadata.get("delivery_error"),
                    "attempts": message.channel_metadata.get("delivery_attempts", 0),
                    "content_type": message.message_type
                }
                failed_deliveries.append(delivery_info)

            self.log_operation(
                "get_failed_deliveries",
                tenant_id=tenant_id,
                count=len(failed_deliveries)
            )

            return failed_deliveries

        except Exception as e:
            error = self.handle_service_error(
                e, "get_failed_deliveries",
                tenant_id=tenant_id
            )
            raise error

    async def bulk_retry_failed_deliveries(
            self,
            tenant_id: TenantId,
            message_ids: List[str],
            user_context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Retry multiple failed deliveries in bulk

        Args:
            tenant_id: Tenant identifier
            message_ids: List of message IDs to retry
            user_context: User authentication context

        Returns:
            Bulk retry results
        """
        try:
            await self.validate_tenant_access(tenant_id, user_context)

            results = {
                "total_requested": len(message_ids),
                "successful_retries": 0,
                "failed_retries": 0,
                "details": []
            }

            for message_id in message_ids:
                try:
                    retry_result = await self.retry_failed_delivery(
                        tenant_id, message_id, user_context
                    )

                    if retry_result["success"]:
                        results["successful_retries"] += 1
                    else:
                        results["failed_retries"] += 1

                    results["details"].append({
                        "message_id": message_id,
                        "success": retry_result["success"],
                        "error": retry_result.get("error")
                    })

                except Exception as e:
                    results["failed_retries"] += 1
                    results["details"].append({
                        "message_id": message_id,
                        "success": False,
                        "error": str(e)
                    })

            self.log_operation(
                "bulk_retry_failed_deliveries",
                tenant_id=tenant_id,
                total_requested=results["total_requested"],
                successful=results["successful_retries"],
                failed=results["failed_retries"]
            )

            return results

        except Exception as e:
            error = self.handle_service_error(
                e, "bulk_retry_failed_deliveries",
                tenant_id=tenant_id
            )
            raise error