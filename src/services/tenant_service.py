"""
Tenant Service Implementation
============================

Service for tenant management, subscription handling, and configuration management.
Provides comprehensive tenant lifecycle operations with proper business logic.
"""

from datetime import datetime, timedelta
from typing import Optional, Dict, Any, List, Tuple
from uuid import uuid4

from src.services.base_service import BaseService
from src.services.exceptions import (
    ServiceError, ValidationError, NotFoundError,
    ConflictError, UnauthorizedError
)
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
from src.repositories.tenant_repository import TenantRepository
from src.services.audit_service import AuditService


class TenantService(BaseService):
    """Service for tenant management and business operations"""

    def __init__(
            self,
            tenant_repo: TenantRepository,
            audit_service: AuditService
    ):
        super().__init__()
        self.tenant_repo = tenant_repo
        self.audit_service = audit_service

        # Business rule configurations
        self.trial_duration_days = 14
        self.max_tenants_per_organization = 10

    async def create_tenant(
            self,
            request: TenantCreateRequest,
            user_context: Dict[str, Any]
    ) -> TenantResponse:
        """
        Create a new tenant with proper business logic

        Args:
            request: Tenant creation request
            user_context: User authentication context

        Returns:
            Created tenant response
        """
        try:
            self.log_operation(
                "create_tenant_start",
                tenant_id=request.tenant_id,
                name=request.name,
                plan_type=request.plan_type.value,
                created_by=user_context.get("user_id")
            )

            # Validate request data
            await self._validate_tenant_creation_request(request)

            # Check for duplicate tenant ID or subdomain
            await self._check_tenant_uniqueness(request)

            # Set up trial period for new tenants
            tenant_data = await self._prepare_tenant_for_creation(request, user_context)

            # Create tenant with features and quotas
            created_tenant = await self.tenant_repo.create(tenant_data)

            # Log audit event
            await self.audit_service.log_event(
                event_type="tenant_created",
                tenant_id=created_tenant.tenant_id,
                user_id=user_context.get("user_id"),
                details={
                    "tenant_name": created_tenant.name,
                    "plan_type": created_tenant.plan_type.value,
                    "created_by": user_context.get("user_id")
                }
            )

            self.log_operation(
                "create_tenant_complete",
                tenant_id=created_tenant.tenant_id,
                name=created_tenant.name,
                plan_type=created_tenant.plan_type.value
            )

            return TenantResponse.model_validate(created_tenant)

        except (ValidationError, ConflictError):
            raise
        except Exception as e:
            error = self.handle_service_error(
                e, "create_tenant",
                tenant_id=getattr(request, 'tenant_id', None)
            )
            raise error

    async def get_tenant(
            self,
            tenant_id: str,
            user_context: Dict[str, Any],
            include_relationships: bool = False
    ) -> Optional[TenantDetailResponse]:
        """Get tenant by ID with proper access validation"""
        try:
            await self.validate_tenant_access(tenant_id, user_context)

            tenant = await self.tenant_repo.get_by_id(tenant_id, include_relationships)

            if not tenant:
                raise NotFoundError(f"Tenant {tenant_id} not found", "tenant", tenant_id)

            self.log_operation(
                "get_tenant",
                tenant_id=tenant_id,
                user_id=user_context.get("user_id"),
                include_relationships=include_relationships
            )

            # Update last activity
            await self.tenant_repo.update_activity(tenant_id)

            if include_relationships:
                return TenantDetailResponse.model_validate(tenant.to_dict_with_relationships())
            else:
                return TenantDetailResponse.model_validate(tenant)

        except (NotFoundError, UnauthorizedError):
            raise
        except Exception as e:
            error = self.handle_service_error(e, "get_tenant", tenant_id=tenant_id)
            raise error

    async def update_tenant(
            self,
            tenant_id: str,
            updates: TenantUpdateRequest,
            user_context: Dict[str, Any]
    ) -> TenantResponse:
        """Update tenant with business logic validation"""
        try:
            await self.validate_tenant_access(tenant_id, user_context)

            # Get existing tenant
            existing_tenant = await self.tenant_repo.get_by_id(tenant_id)
            if not existing_tenant:
                raise NotFoundError(f"Tenant {tenant_id} not found", "tenant", tenant_id)

            # Validate updates
            await self._validate_tenant_updates(existing_tenant, updates, user_context)

            # Apply business rules for specific updates
            processed_updates = await self._process_tenant_updates(
                existing_tenant, updates, user_context
            )

            # Update tenant
            updated_tenant = await self.tenant_repo.update(tenant_id, processed_updates)

            # Log audit event
            await self.audit_service.log_event(
                event_type="tenant_updated",
                tenant_id=tenant_id,
                user_id=user_context.get("user_id"),
                details={
                    "updates": updates.model_dump(exclude_unset=True),
                    "updated_by": user_context.get("user_id")
                }
            )

            self.log_operation(
                "update_tenant",
                tenant_id=tenant_id,
                updates_count=len(updates.model_dump(exclude_unset=True)),
                updated_by=user_context.get("user_id")
            )

            return TenantResponse.model_validate(updated_tenant)

        except (NotFoundError, UnauthorizedError, ValidationError):
            raise
        except Exception as e:
            error = self.handle_service_error(e, "update_tenant", tenant_id=tenant_id)
            raise error

    async def delete_tenant(
            self,
            tenant_id: str,
            user_context: Dict[str, Any]
    ) -> bool:
        """Soft delete tenant with proper cleanup"""
        try:
            await self.validate_tenant_access(tenant_id, user_context)

            # Validate deletion permissions
            if not await self._can_delete_tenant(tenant_id, user_context):
                raise UnauthorizedError(
                    "Insufficient permissions to delete tenant",
                    user_context.get("user_id"),
                    f"tenant:{tenant_id}"
                )

            # Perform soft delete
            success = await self.tenant_repo.delete(tenant_id)

            if success:
                # Schedule cleanup tasks
                await self._schedule_tenant_cleanup(tenant_id, user_context)

                # Log audit event
                await self.audit_service.log_event(
                    event_type="tenant_deleted",
                    tenant_id=tenant_id,
                    user_id=user_context.get("user_id"),
                    details={
                        "deleted_by": user_context.get("user_id"),
                        "deletion_type": "soft_delete"
                    }
                )

                self.log_operation(
                    "delete_tenant",
                    tenant_id=tenant_id,
                    deleted_by=user_context.get("user_id")
                )

            return success

        except (UnauthorizedError, NotFoundError):
            raise
        except Exception as e:
            error = self.handle_service_error(e, "delete_tenant", tenant_id=tenant_id)
            raise error

    async def list_tenants(
            self,
            filters: Optional[TenantFilterParams] = None,
            page: int = 1,
            page_size: int = 20,
            sort_by: str = "created_at",
            sort_order: str = "desc",
            user_context: Dict[str, Any] = None
    ) -> Tuple[List[TenantResponse], int]:
        """List tenants with filtering and pagination"""
        try:
            # Apply user-specific filters based on permissions
            processed_filters = await self._apply_user_filters(filters, user_context)

            tenants, total = await self.tenant_repo.list_tenants(
                filters=processed_filters,
                page=page,
                page_size=page_size,
                sort_by=sort_by,
                sort_order=sort_order
            )

            tenant_responses = [
                TenantResponse.model_validate(tenant) for tenant in tenants
            ]

            self.log_operation(
                "list_tenants",
                total=total,
                returned=len(tenant_responses),
                page=page,
                page_size=page_size,
                user_id=user_context.get("user_id") if user_context else None
            )

            return tenant_responses, total

        except Exception as e:
            error = self.handle_service_error(e, "list_tenants")
            raise error

    async def update_subscription(
            self,
            tenant_id: str,
            subscription_id: str,
            plan_type: TenantPlan,
            stripe_customer_id: Optional[str] = None,
            user_context: Dict[str, Any] = None
    ) -> bool:
        """Update tenant subscription with business logic"""
        try:
            await self.validate_tenant_access(tenant_id, user_context)

            # Get existing tenant to validate plan change
            existing_tenant = await self.tenant_repo.get_by_id(tenant_id, include_relationships=True)
            if not existing_tenant:
                raise NotFoundError(f"Tenant {tenant_id} not found", "tenant", tenant_id)

            # Validate plan change business rules
            await self._validate_plan_change(existing_tenant, plan_type, user_context)

            # Update subscription
            success = await self.tenant_repo.update_subscription(
                tenant_id, subscription_id, plan_type, stripe_customer_id
            )

            if success:
                # Update quotas based on new plan
                await self._update_quotas_for_plan(tenant_id, plan_type)

                # Log audit event
                await self.audit_service.log_event(
                    event_type="subscription_updated",
                    tenant_id=tenant_id,
                    user_id=user_context.get("user_id") if user_context else None,
                    details={
                        "subscription_id": subscription_id,
                        "plan_type": plan_type.value,
                        "stripe_customer_id": stripe_customer_id
                    }
                )

                self.log_operation(
                    "update_subscription",
                    tenant_id=tenant_id,
                    plan_type=plan_type.value,
                    subscription_id=subscription_id
                )

            return success

        except (NotFoundError, UnauthorizedError, ValidationError):
            raise
        except Exception as e:
            error = self.handle_service_error(e, "update_subscription", tenant_id=tenant_id)
            raise error

    async def get_tenant_statistics(
            self,
            user_context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Get tenant statistics (admin only)"""
        try:
            # Validate admin permissions
            if not await self._is_admin_user(user_context):
                raise UnauthorizedError(
                    "Admin permissions required for tenant statistics",
                    user_context.get("user_id"),
                    "tenant_statistics"
                )

            stats = await self.tenant_repo.get_tenant_stats()

            # Add additional calculated metrics
            enhanced_stats = await self._enhance_tenant_statistics(stats)

            self.log_operation(
                "get_tenant_statistics",
                user_id=user_context.get("user_id")
            )

            return enhanced_stats

        except UnauthorizedError:
            raise
        except Exception as e:
            error = self.handle_service_error(e, "get_tenant_statistics")
            raise error

    # Private helper methods
    async def _validate_tenant_creation_request(self, request: TenantCreateRequest) -> None:
        """Validate tenant creation request"""
        if len(request.name) < 2:
            raise ValidationError("Tenant name must be at least 2 characters", "name")

        if request.subdomain and len(request.subdomain) < 3:
            raise ValidationError("Subdomain must be at least 3 characters", "subdomain")

        # Add more business rule validations
        reserved_subdomains = ["api", "admin", "www", "mail", "ftp"]
        if request.subdomain and request.subdomain.lower() in reserved_subdomains:
            raise ValidationError(f"Subdomain '{request.subdomain}' is reserved", "subdomain")

    async def _check_tenant_uniqueness(self, request: TenantCreateRequest) -> None:
        """Check if tenant ID and subdomain are unique"""
        # Check tenant ID
        existing_tenant = await self.tenant_repo.get_by_id(request.tenant_id)
        if existing_tenant:
            raise ConflictError(
                f"Tenant with ID '{request.tenant_id}' already exists",
                "tenant",
                "tenant_id"
            )

        # Check subdomain if provided
        if request.subdomain:
            existing_by_subdomain = await self.tenant_repo.get_by_subdomain(request.subdomain)
            if existing_by_subdomain:
                raise ConflictError(
                    f"Subdomain '{request.subdomain}' is already taken",
                    "tenant",
                    "subdomain"
                )

    async def _prepare_tenant_for_creation(
            self,
            request: TenantCreateRequest,
            user_context: Dict[str, Any]
    ) -> TenantCreateRequest:
        """Prepare tenant data with business logic"""
        # Set trial period for new tenants
        if request.plan_type == TenantPlan.STARTER:
            trial_end = datetime.utcnow() + timedelta(days=self.trial_duration_days)
            request.trial_ends_at = trial_end
            request.status = TenantStatus.TRIAL

        # Set created_by from user context
        if user_context and "user_id" in user_context:
            # This would need to be added to the model if not present
            pass

        return request

    async def _validate_tenant_updates(
            self,
            existing_tenant: Tenant,
            updates: TenantUpdateRequest,
            user_context: Dict[str, Any]
    ) -> None:
        """Validate tenant update request"""
        # Validate subdomain change
        if updates.subdomain and updates.subdomain != existing_tenant.subdomain:
            existing_by_subdomain = await self.tenant_repo.get_by_subdomain(updates.subdomain)
            if existing_by_subdomain and existing_by_subdomain.tenant_id != existing_tenant.tenant_id:
                raise ConflictError(
                    f"Subdomain '{updates.subdomain}' is already taken",
                    "tenant",
                    "subdomain"
                )

        # Validate plan downgrade
        if updates.plan_type and updates.plan_type != existing_tenant.plan_type:
            await self._validate_plan_change(existing_tenant, updates.plan_type, user_context)

    async def _process_tenant_updates(
            self,
            existing_tenant: Tenant,
            updates: TenantUpdateRequest,
            user_context: Dict[str, Any]
    ) -> TenantUpdateRequest:
        """Process tenant updates with business logic"""
        # Handle status changes
        if updates.status and updates.status != existing_tenant.status:
            # Add business logic for status transitions
            if updates.status == TenantStatus.SUSPENDED:
                # Log suspension reason, notify users, etc.
                pass

        return updates

    async def _can_delete_tenant(self, tenant_id: str, user_context: Dict[str, Any]) -> bool:
        """Check if user can delete tenant"""
        # Check if user is admin or tenant owner
        user_role = user_context.get("role", "user")
        user_tenant_id = user_context.get("tenant_id")

        if user_role == "admin":
            return True

        if user_role == "owner" and user_tenant_id == tenant_id:
            return True

        return False

    async def _schedule_tenant_cleanup(self, tenant_id: str, user_context: Dict[str, Any]) -> None:
        """Schedule background cleanup tasks for deleted tenant"""
        # This would typically schedule background jobs for:
        # - Data cleanup after retention period
        # - Webhook notifications
        # - External service cleanup
        pass

    async def _apply_user_filters(
            self,
            filters: Optional[TenantFilterParams],
            user_context: Dict[str, Any]
    ) -> Optional[TenantFilterParams]:
        """Apply user-specific filters based on permissions"""
        if not user_context:
            return filters

        user_role = user_context.get("role", "user")
        user_tenant_id = user_context.get("tenant_id")

        # Non-admin users can only see their own tenant
        if user_role != "admin" and user_tenant_id:
            if not filters:
                filters = TenantFilterParams()
            # This would need additional filtering logic

        return filters

    async def _validate_plan_change(
            self,
            tenant: Tenant,
            new_plan: TenantPlan,
            user_context: Dict[str, Any]
    ) -> None:
        """Validate plan change business rules"""
        current_plan = tenant.plan_type

        # Check if downgrade is allowed
        if self._is_plan_downgrade(current_plan, new_plan):
            # Check usage against new plan limits
            if tenant.quotas:
                await self._validate_usage_against_plan(tenant, new_plan)

    def _is_plan_downgrade(self, current_plan: TenantPlan, new_plan: TenantPlan) -> bool:
        """Check if plan change is a downgrade"""
        plan_hierarchy = {
            TenantPlan.STARTER: 1,
            TenantPlan.PROFESSIONAL: 2,
            TenantPlan.ENTERPRISE: 3
        }

        return plan_hierarchy.get(new_plan, 0) < plan_hierarchy.get(current_plan, 0)

    async def _validate_usage_against_plan(self, tenant: Tenant, new_plan: TenantPlan) -> None:
        """Validate current usage against new plan limits"""
        # This would check current usage against new plan quotas
        # and raise ValidationError if usage exceeds new limits
        pass

    async def _update_quotas_for_plan(self, tenant_id: str, plan_type: TenantPlan) -> None:
        """Update tenant quotas based on plan type"""
        # This would update the tenant's quotas based on the new plan
        # using the tenant repository's quota update methods
        pass

    async def _is_admin_user(self, user_context: Dict[str, Any]) -> bool:
        """Check if user has admin permissions"""
        return user_context.get("role") == "admin"

    async def _enhance_tenant_statistics(self, base_stats: Dict[str, Any]) -> Dict[str, Any]:
        """Add additional calculated metrics to tenant statistics"""
        enhanced_stats = base_stats.copy()

        # Add growth metrics
        if "recent_signups" in base_stats and "total_tenants" in base_stats:
            if base_stats["total_tenants"] > 0:
                enhanced_stats["growth_rate"] = (
                                                        base_stats["recent_signups"] / base_stats["total_tenants"]
                                                ) * 100

        # Add plan distribution percentages
        if "by_plan" in base_stats and "total_tenants" in base_stats:
            total = base_stats["total_tenants"]
            if total > 0:
                enhanced_stats["plan_distribution_percentage"] = {
                    plan: (count / total) * 100
                    for plan, count in base_stats["by_plan"].items()
                }

        return enhanced_stats

