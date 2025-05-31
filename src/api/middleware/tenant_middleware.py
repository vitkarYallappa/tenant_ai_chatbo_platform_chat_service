"""
Tenant Isolation Middleware
Ensures proper tenant isolation and validates tenant access for all requests.
"""

from typing import Optional, Dict, Any
from fastapi import Request, HTTPException, status, Header, Depends
import structlog

from src.api.middleware.auth_middleware import AuthContext, get_auth_context, get_optional_auth_context
from src.repositories.tenant_repository import TenantRepository
from src.dependencies import get_tenant_repository
from src.config.settings import get_settings

logger = structlog.get_logger()


class TenantContext:
    """Tenant context for request processing"""

    def __init__(
            self,
            tenant_id: str,
            tenant_data: Dict[str, Any],
            is_active: bool = True,
            plan_type: str = "starter",
            features: Dict[str, Any] = None,
            quotas: Dict[str, Any] = None
    ):
        self.tenant_id = tenant_id
        self.tenant_data = tenant_data
        self.is_active = is_active
        self.plan_type = plan_type
        self.features = features or {}
        self.quotas = quotas or {}

    def has_feature(self, feature_name: str) -> bool:
        """Check if tenant has a specific feature enabled"""
        return self.features.get(feature_name, False)

    def get_quota(self, quota_name: str, default: int = 0) -> int:
        """Get quota limit for a specific resource"""
        return self.quotas.get(quota_name, default)

    def is_within_quota(self, quota_name: str, current_usage: int) -> bool:
        """Check if current usage is within quota limits"""
        limit = self.get_quota(quota_name)
        return limit == 0 or current_usage < limit  # 0 means unlimited


async def get_tenant_context(
        tenant_id: str = Header(alias="X-Tenant-ID"),
        tenant_repo: TenantRepository = Depends(get_tenant_repository),
        auth_context: Optional[AuthContext] = Depends(get_optional_auth_context)
) -> TenantContext:
    """
    Get and validate tenant context from request

    Args:
        tenant_id: Tenant ID from header
        tenant_repo: Tenant repository for data access
        auth_context: Optional authentication context

    Returns:
        TenantContext with validated tenant data

    Raises:
        HTTPException: If tenant validation fails
    """
    try:
        # Validate tenant ID format
        if not tenant_id:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="X-Tenant-ID header is required"
            )

        # Validate tenant ID against auth context if available
        if auth_context and auth_context.tenant_id != tenant_id:
            logger.warning(
                "Tenant ID mismatch",
                auth_tenant=auth_context.tenant_id,
                header_tenant=tenant_id,
                user_id=auth_context.user_id
            )
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Access denied: tenant ID mismatch"
            )

        # Fetch tenant data from repository
        tenant_data = await tenant_repo.get_tenant_by_id(tenant_id)
        if not tenant_data:
            logger.warning("Tenant not found", tenant_id=tenant_id)
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Tenant not found"
            )

        # Check if tenant is active
        if tenant_data.get("status") != "active":
            tenant_status = tenant_data.get("status", "unknown")
            logger.warning(
                "Inactive tenant access attempt",
                tenant_id=tenant_id,
                status=tenant_status
            )

            error_messages = {
                "suspended": "Tenant account is suspended",
                "deleted": "Tenant account has been deleted",
                "trial_expired": "Trial period has expired"
            }

            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=error_messages.get(tenant_status, "Tenant account is not active")
            )

        # Create tenant context
        tenant_context = TenantContext(
            tenant_id=tenant_id,
            tenant_data=tenant_data,
            is_active=tenant_data.get("status") == "active",
            plan_type=tenant_data.get("plan_type", "starter"),
            features=tenant_data.get("features", {}),
            quotas=tenant_data.get("quotas", {})
        )

        logger.debug(
            "Tenant context established",
            tenant_id=tenant_id,
            plan_type=tenant_context.plan_type,
            is_active=tenant_context.is_active
        )

        return tenant_context

    except HTTPException:
        raise
    except Exception as e:
        logger.error("Tenant validation failed", tenant_id=tenant_id, error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Tenant validation failed"
        )


async def validate_tenant_feature(
        feature_name: str,
        tenant_context: TenantContext = Depends(get_tenant_context)
) -> bool:
    """
    Validate that tenant has access to a specific feature

    Args:
        feature_name: Name of the feature to check
        tenant_context: Tenant context

    Returns:
        True if feature is available

    Raises:
        HTTPException: If feature is not available
    """
    if not tenant_context.has_feature(feature_name):
        logger.warning(
            "Feature not available for tenant",
            tenant_id=tenant_context.tenant_id,
            feature=feature_name,
            plan_type=tenant_context.plan_type
        )
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=f"Feature '{feature_name}' is not available for your plan"
        )

    return True


def require_feature(feature_name: str):
    """
    Decorator to require a specific feature for an endpoint

    Args:
        feature_name: Name of the required feature

    Returns:
        Dependency function that validates feature access
    """

    async def feature_checker(
            tenant_context: TenantContext = Depends(get_tenant_context)
    ) -> TenantContext:
        """Check if tenant has required feature"""
        await validate_tenant_feature(feature_name, tenant_context)
        return tenant_context

    return feature_checker


def require_plan(min_plan: str):
    """
    Decorator to require a minimum plan level for an endpoint

    Args:
        min_plan: Minimum required plan level

    Returns:
        Dependency function that validates plan level
    """
    plan_hierarchy = {
        "starter": 0,
        "professional": 1,
        "enterprise": 2,
        "custom": 3
    }

    async def plan_checker(
            tenant_context: TenantContext = Depends(get_tenant_context)
    ) -> TenantContext:
        """Check if tenant has required plan level"""
        current_level = plan_hierarchy.get(tenant_context.plan_type, 0)
        required_level = plan_hierarchy.get(min_plan, 999)

        if current_level < required_level:
            logger.warning(
                "Insufficient plan level",
                tenant_id=tenant_context.tenant_id,
                current_plan=tenant_context.plan_type,
                required_plan=min_plan
            )
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Plan upgrade required. Minimum plan: {min_plan}"
            )

        return tenant_context

    return plan_checker


async def check_quota_usage(
        quota_name: str,
        current_usage: int,
        tenant_context: TenantContext = Depends(get_tenant_context)
) -> bool:
    """
    Check if current usage is within quota limits

    Args:
        quota_name: Name of the quota to check
        current_usage: Current usage amount
        tenant_context: Tenant context

    Returns:
        True if within quota

    Raises:
        HTTPException: If quota is exceeded
    """
    quota_limit = tenant_context.get_quota(quota_name)

    if quota_limit > 0 and current_usage >= quota_limit:
        logger.warning(
            "Quota exceeded",
            tenant_id=tenant_context.tenant_id,
            quota_name=quota_name,
            current_usage=current_usage,
            quota_limit=quota_limit
        )
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail=f"Quota exceeded for {quota_name}. Current: {current_usage}, Limit: {quota_limit}",
            headers={
                "X-Quota-Name": quota_name,
                "X-Quota-Limit": str(quota_limit),
                "X-Quota-Used": str(current_usage),
                "X-Quota-Remaining": str(max(0, quota_limit - current_usage))
            }
        )

    return True


def require_quota(quota_name: str, usage_increment: int = 1):
    """
    Decorator to check quota before allowing endpoint access

    Args:
        quota_name: Name of the quota to check
        usage_increment: Amount this operation will consume

    Returns:
        Dependency function that validates quota
    """

    async def quota_checker(
            tenant_context: TenantContext = Depends(get_tenant_context),
            tenant_repo: TenantRepository = Depends(get_tenant_repository)
    ) -> TenantContext:
        """Check quota before processing request"""
        # Get current usage from repository
        current_usage = await tenant_repo.get_quota_usage(
            tenant_context.tenant_id,
            quota_name
        )

        # Check if adding this usage would exceed quota
        projected_usage = current_usage + usage_increment
        await check_quota_usage(quota_name, projected_usage, tenant_context)

        return tenant_context

    return quota_checker


class TenantIsolationMiddleware:
    """Middleware to ensure tenant data isolation"""

    def __init__(self, app):
        self.app = app

    async def __call__(self, request: Request, call_next):
        """Process request through tenant isolation checks"""

        # Skip tenant checks for public endpoints
        public_paths = ["/health", "/metrics", "/docs", "/openapi.json"]
        if any(request.url.path.startswith(path) for path in public_paths):
            response = await call_next(request)
            return response

        try:
            # Extract tenant ID from header
            tenant_id = request.headers.get("X-Tenant-ID")
            if not tenant_id:
                from fastapi.responses import JSONResponse
                return JSONResponse(
                    status_code=400,
                    content={
                        "status": "error",
                        "error": {
                            "code": "MISSING_TENANT_ID",
                            "message": "X-Tenant-ID header is required"
                        }
                    }
                )

            # Add tenant ID to request state for downstream use
            request.state.tenant_id = tenant_id

            # Continue with request processing
            response = await call_next(request)

            # Add tenant isolation headers to response
            response.headers["X-Tenant-ID"] = tenant_id
            response.headers["X-Data-Residency"] = "us"  # Could be dynamic based on tenant

            return response

        except Exception as e:
            logger.error("Tenant isolation middleware error", error=str(e))
            from fastapi.responses import JSONResponse
            return JSONResponse(
                status_code=500,
                content={
                    "status": "error",
                    "error": {
                        "code": "TENANT_ISOLATION_ERROR",
                        "message": "Tenant isolation check failed"
                    }
                }
            )


async def validate_cross_tenant_access(
        resource_tenant_id: str,
        tenant_context: TenantContext = Depends(get_tenant_context)
) -> bool:
    """
    Validate that a user is not accessing resources from another tenant

    Args:
        resource_tenant_id: Tenant ID of the resource being accessed
        tenant_context: Current request tenant context

    Returns:
        True if access is allowed

    Raises:
        HTTPException: If cross-tenant access is attempted
    """
    if resource_tenant_id != tenant_context.tenant_id:
        logger.error(
            "Cross-tenant access attempt detected",
            request_tenant=tenant_context.tenant_id,
            resource_tenant=resource_tenant_id
        )
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Access denied: cross-tenant access not allowed"
        )

    return True


class DataResidencyValidator:
    """Validator for data residency requirements"""

    @staticmethod
    async def validate_data_location(
            tenant_context: TenantContext,
            operation_type: str = "read"
    ) -> bool:
        """
        Validate data residency requirements for the operation

        Args:
            tenant_context: Tenant context with residency requirements
            operation_type: Type of operation (read, write, process)

        Returns:
            True if operation is allowed in current region

        Raises:
            HTTPException: If data residency requirements are violated
        """
        tenant_residency = tenant_context.tenant_data.get("data_residency", "us")
        current_region = get_settings().AWS_REGION or "us-east-1"

        # Map AWS regions to data residency regions
        region_mapping = {
            "us-east-1": "us",
            "us-west-2": "us",
            "eu-west-1": "eu",
            "eu-central-1": "eu",
            "ap-southeast-1": "apac",
            "ap-northeast-1": "apac"
        }

        current_residency = region_mapping.get(current_region, "us")

        if tenant_residency != current_residency:
            logger.error(
                "Data residency violation",
                tenant_id=tenant_context.tenant_id,
                required_residency=tenant_residency,
                current_region=current_region,
                operation_type=operation_type
            )
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Data residency requirement violated. Required: {tenant_residency}, Current: {current_residency}"
            )

        return True


def require_data_residency_compliance(operation_type: str = "read"):
    """
    Decorator to enforce data residency compliance

    Args:
        operation_type: Type of operation being performed

    Returns:
        Dependency function that validates data residency
    """

    async def residency_checker(
            tenant_context: TenantContext = Depends(get_tenant_context)
    ) -> TenantContext:
        """Check data residency compliance"""
        await DataResidencyValidator.validate_data_location(
            tenant_context, operation_type
        )
        return tenant_context

    return residency_checker