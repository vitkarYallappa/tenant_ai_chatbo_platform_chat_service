# src/api/routes/tenants.py
"""
Tenant API Routes
================

FastAPI routes for tenant management following established patterns with
proper authentication, tenant isolation, and response formatting.
"""

from typing import List, Optional, Annotated
from fastapi import APIRouter, HTTPException, Query, Path, Header, Depends, status
from fastapi.responses import JSONResponse

from api.responses import APIResponse
from src.dependencies import (
    get_tenant_service,
    CurrentUserDep,
    is_postgresql_available
)
from src.models.schemas.tenant_webhook_request_schemas import (
    TenantCreateRequest, TenantUpdateRequest, TenantResponse,
    TenantDetailResponse, TenantFilterParams, PaginatedResponse
)
from src.services.exceptions import (
    ServiceError, ValidationError, NotFoundError,
    ConflictError, UnauthorizedError
)
from src.services.tenant_service import TenantService
from src.services.exceptions import (
    ServiceError, ValidationError, NotFoundError,
    ConflictError, UnauthorizedError
)
from src.api.middleware.auth_middleware import AuthContext, get_auth_context
from src.api.responses import create_success_response, create_error_response

router = APIRouter(prefix="/api/v1/tenants", tags=["tenants"])


@router.post(
    "",
    response_model=APIResponse[TenantResponse],
    status_code=status.HTTP_201_CREATED,
    summary="Create tenant",
    description="Create a new tenant with business logic validation and proper setup"
)
async def create_tenant(
    request: TenantCreateRequest,
    tenant_id: Annotated[str, Header(alias="X-Tenant-ID")],
    auth_context: Annotated[AuthContext, Depends(get_auth_context)],
    tenant_service: Annotated[TenantService, Depends(get_tenant_service)]
) -> APIResponse[TenantResponse]:
    """
    Create a new tenant with comprehensive business logic

    Args:
        request: Tenant creation data including plan, subdomain, and configuration
        tenant_id: Tenant identifier from header
        auth_context: User authentication context with permissions
        tenant_service: Tenant management service

    Returns:
        APIResponse containing created tenant details with features and quotas
    """
    try:
        # Validate admin permissions for tenant creation
        if not auth_context.has_permission("tenant:create"):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Insufficient permissions to create tenant"
            )

        # Create tenant with business logic
        tenant_response = await tenant_service.create_tenant(
            request=request,
            user_context=auth_context.dict()
        )

        return create_success_response(
            data=tenant_response,
            message="Tenant created successfully"
        )

    except ValidationError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    except ConflictError as e:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=str(e)
        )
    except UnauthorizedError as e:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=str(e)
        )
    except ServiceError as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to create tenant"
        )


@router.get(
    "/{target_tenant_id}",
    response_model=APIResponse[TenantDetailResponse],
    summary="Get tenant details",
    description="Retrieve comprehensive tenant information including features and quotas"
)
async def get_tenant(
    target_tenant_id: Annotated[str, Path(description="Target tenant ID to retrieve")],
    tenant_id: Annotated[str, Header(alias="X-Tenant-ID")],
    auth_context: Annotated[AuthContext, Depends(get_auth_context)],
    tenant_service: Annotated[TenantService, Depends(get_tenant_service)],
    include_relationships: Annotated[bool, Query(description="Include features and quotas")] = True
) -> APIResponse[TenantDetailResponse]:
    """
    Get tenant by ID with comprehensive information

    Args:
        target_tenant_id: Tenant identifier to retrieve
        tenant_id: Requesting tenant identifier from header
        auth_context: User authentication context
        tenant_service: Tenant management service
        include_relationships: Whether to include features and quotas data

    Returns:
        APIResponse containing tenant details with optional relationships
    """
    try:
        # Validate tenant access - users can only access their own tenant unless admin
        if auth_context.tenant_id != target_tenant_id and not auth_context.has_permission("tenant:read_all"):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Access denied to tenant resources"
            )

        # Get tenant with business logic
        tenant = await tenant_service.get_tenant(
            tenant_id=target_tenant_id,
            user_context=auth_context.dict(),
            include_relationships=include_relationships
        )

        if not tenant:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Tenant {target_tenant_id} not found"
            )

        return create_success_response(
            data=tenant,
            message="Tenant retrieved successfully"
        )

    except NotFoundError as e:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(e)
        )
    except UnauthorizedError as e:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=str(e)
        )
    except ServiceError as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve tenant"
        )


@router.get(
    "",
    response_model=APIResponse[PaginatedResponse[TenantResponse]],
    summary="List tenants",
    description="Get paginated list of tenants with filtering and sorting"
)
async def list_tenants(
    tenant_id: Annotated[str, Header(alias="X-Tenant-ID")],
    auth_context: Annotated[AuthContext, Depends(get_auth_context)],
    tenant_service: Annotated[TenantService, Depends(get_tenant_service)],
    page: Annotated[int, Query(ge=1, description="Page number")] = 1,
    page_size: Annotated[int, Query(ge=1, le=100, description="Items per page")] = 20,
    sort_by: Annotated[str, Query(description="Sort field")] = "created_at",
    sort_order: Annotated[str, Query(regex="^(asc|desc)$", description="Sort order")] = "desc",
    status: Annotated[Optional[str], Query(description="Filter by status")] = None,
    plan_type: Annotated[Optional[str], Query(description="Filter by plan")] = None,
    industry: Annotated[Optional[str], Query(description="Filter by industry")] = None,
    search: Annotated[Optional[str], Query(description="Search term")] = None
) -> APIResponse[PaginatedResponse[TenantResponse]]:
    """
    List tenants with filtering, pagination, and business logic

    Args:
        tenant_id: Requesting tenant identifier from header
        auth_context: User authentication context
        tenant_service: Tenant management service
        page: Page number for pagination
        page_size: Number of items per page
        sort_by: Field to sort by
        sort_order: Sort order (asc/desc)
        status: Optional status filter
        plan_type: Optional plan type filter
        industry: Optional industry filter
        search: Optional search term

    Returns:
        APIResponse containing paginated tenant list with metadata
    """
    try:
        # Validate list permissions
        if not auth_context.has_permission("tenant:list"):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Insufficient permissions to list tenants"
            )

        # Build filters
        filters = TenantFilterParams(
            status=status,
            plan_type=plan_type,
            industry=industry,
            search=search
        )

        # Get tenants with business logic
        tenants, total = await tenant_service.list_tenants(
            filters=filters,
            page=page,
            page_size=page_size,
            sort_by=sort_by,
            sort_order=sort_order,
            user_context=auth_context.dict()
        )

        # Create paginated response
        paginated_response = PaginatedResponse(
            items=tenants,
            total=total,
            page=page,
            page_size=page_size,
            total_pages=(total + page_size - 1) // page_size,
            has_next=page * page_size < total,
            has_previous=page > 1
        )

        return create_success_response(
            data=paginated_response,
            message=f"Retrieved {len(tenants)} tenants"
        )

    except UnauthorizedError as e:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=str(e)
        )
    except ServiceError as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to list tenants"
        )


@router.put(
    "/{target_tenant_id}",
    response_model=APIResponse[TenantResponse],
    summary="Update tenant",
    description="Update tenant configuration with business logic validation"
)
async def update_tenant(
    target_tenant_id: Annotated[str, Path(description="Target tenant ID to update")],
    request: TenantUpdateRequest,
    tenant_id: Annotated[str, Header(alias="X-Tenant-ID")],
    auth_context: Annotated[AuthContext, Depends(get_auth_context)],
    tenant_service: Annotated[TenantService, Depends(get_tenant_service)]
) -> APIResponse[TenantResponse]:
    """
    Update tenant with business logic validation

    Args:
        target_tenant_id: Tenant identifier to update
        request: Tenant update data including configuration changes
        tenant_id: Requesting tenant identifier from header
        auth_context: User authentication context
        tenant_service: Tenant management service

    Returns:
        APIResponse containing updated tenant details
    """
    try:
        # Validate tenant access
        if auth_context.tenant_id != target_tenant_id and not auth_context.has_permission("tenant:update_all"):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Access denied to tenant resources"
            )

        # Update tenant with business logic
        updated_tenant = await tenant_service.update_tenant(
            tenant_id=target_tenant_id,
            updates=request,
            user_context=auth_context.dict()
        )

        return create_success_response(
            data=updated_tenant,
            message="Tenant updated successfully"
        )

    except ValidationError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    except NotFoundError as e:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(e)
        )
    except UnauthorizedError as e:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=str(e)
        )
    except ConflictError as e:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=str(e)
        )
    except ServiceError as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to update tenant"
        )


@router.delete(
    "/{target_tenant_id}",
    response_model=APIResponse[dict],
    summary="Delete tenant",
    description="Soft delete tenant with proper authorization and cleanup"
)
async def delete_tenant(
    target_tenant_id: Annotated[str, Path(description="Target tenant ID to delete")],
    tenant_id: Annotated[str, Header(alias="X-Tenant-ID")],
    auth_context: Annotated[AuthContext, Depends(get_auth_context)],
    tenant_service: Annotated[TenantService, Depends(get_tenant_service)]
) -> APIResponse[dict]:
    """
    Delete tenant with proper authorization

    Args:
        target_tenant_id: Tenant identifier to delete
        tenant_id: Requesting tenant identifier from header
        auth_context: User authentication context
        tenant_service: Tenant management service

    Returns:
        APIResponse containing deletion confirmation
    """
    try:
        # Validate deletion permissions (strict - only admins or tenant owners)
        if not auth_context.has_permission("tenant:delete"):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Insufficient permissions to delete tenant"
            )

        # Delete tenant with business logic
        success = await tenant_service.delete_tenant(
            tenant_id=target_tenant_id,
            user_context=auth_context.dict()
        )

        if not success:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Tenant {target_tenant_id} not found"
            )

        return create_success_response(
            data={"tenant_id": target_tenant_id, "deleted": True},
            message="Tenant deleted successfully"
        )

    except NotFoundError as e:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(e)
        )
    except UnauthorizedError as e:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=str(e)
        )
    except ServiceError as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to delete tenant"
        )


@router.put(
    "/{target_tenant_id}/subscription",
    response_model=APIResponse[dict],
    summary="Update subscription",
    description="Update tenant subscription plan with business logic validation"
)
async def update_subscription(
    target_tenant_id: Annotated[str, Path(description="Target tenant ID")],
    tenant_id: Annotated[str, Header(alias="X-Tenant-ID")],
    auth_context: Annotated[AuthContext, Depends(get_auth_context)],
    tenant_service: Annotated[TenantService, Depends(get_tenant_service)],
    subscription_id: Annotated[str, Query(description="Stripe subscription ID")],
    plan_type: Annotated[str, Query(description="New plan type")],
    stripe_customer_id: Annotated[Optional[str], Query(description="Stripe customer ID")] = None
) -> APIResponse[dict]:
    """
    Update tenant subscription with business logic

    Args:
        target_tenant_id: Tenant identifier to update subscription for
        tenant_id: Requesting tenant identifier from header
        auth_context: User authentication context
        tenant_service: Tenant management service
        subscription_id: Stripe subscription identifier
        plan_type: New subscription plan type
        stripe_customer_id: Optional Stripe customer identifier

    Returns:
        APIResponse containing subscription update confirmation
    """
    try:
        # Validate tenant access
        if auth_context.tenant_id != target_tenant_id and not auth_context.has_permission("tenant:update_subscription"):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Access denied to tenant resources"
            )

        # Convert plan_type string to enum
        from src.models.types import TenantPlan
        try:
            plan_enum = TenantPlan(plan_type)
        except ValueError:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Invalid plan type: {plan_type}"
            )

        # Update subscription with business logic
        success = await tenant_service.update_subscription(
            tenant_id=target_tenant_id,
            subscription_id=subscription_id,
            plan_type=plan_enum,
            stripe_customer_id=stripe_customer_id,
            user_context=auth_context.dict()
        )

        if not success:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Tenant {target_tenant_id} not found"
            )

        return create_success_response(
            data={
                "tenant_id": target_tenant_id,
                "subscription_id": subscription_id,
                "plan_type": plan_type,
                "updated": True
            },
            message="Subscription updated successfully"
        )

    except ValidationError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    except NotFoundError as e:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(e)
        )
    except UnauthorizedError as e:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=str(e)
        )
    except ServiceError as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to update subscription"
        )


@router.get(
    "/{target_tenant_id}/features",
    response_model=APIResponse[dict],
    summary="Get tenant features",
    description="Retrieve tenant feature configuration and availability"
)
async def get_tenant_features(
    target_tenant_id: Annotated[str, Path(description="Target tenant ID")],
    tenant_id: Annotated[str, Header(alias="X-Tenant-ID")],
    auth_context: Annotated[AuthContext, Depends(get_auth_context)],
    tenant_service: Annotated[TenantService, Depends(get_tenant_service)]
) -> APIResponse[dict]:
    """
    Get tenant feature configuration

    Args:
        target_tenant_id: Tenant identifier to get features for
        tenant_id: Requesting tenant identifier from header
        auth_context: User authentication context
        tenant_service: Tenant management service

    Returns:
        APIResponse containing tenant feature configuration
    """
    try:
        # Validate tenant access
        if auth_context.tenant_id != target_tenant_id and not auth_context.has_permission("tenant:read_all"):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Access denied to tenant resources"
            )

        # Get tenant with features
        tenant = await tenant_service.get_tenant(
            tenant_id=target_tenant_id,
            user_context=auth_context.dict(),
            include_relationships=True
        )

        if not tenant:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Tenant {target_tenant_id} not found"
            )

        return create_success_response(
            data=tenant.features if hasattr(tenant, 'features') else {},
            message="Tenant features retrieved successfully"
        )

    except NotFoundError as e:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(e)
        )
    except UnauthorizedError as e:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=str(e)
        )
    except ServiceError as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve tenant features"
        )


@router.get(
    "/{target_tenant_id}/quotas",
    response_model=APIResponse[dict],
    summary="Get tenant quotas",
    description="Retrieve tenant quota configuration and current usage"
)
async def get_tenant_quotas(
    target_tenant_id: Annotated[str, Path(description="Target tenant ID")],
    tenant_id: Annotated[str, Header(alias="X-Tenant-ID")],
    auth_context: Annotated[AuthContext, Depends(get_auth_context)],
    tenant_service: Annotated[TenantService, Depends(get_tenant_service)]
) -> APIResponse[dict]:
    """
    Get tenant quota configuration and usage

    Args:
        target_tenant_id: Tenant identifier to get quotas for
        tenant_id: Requesting tenant identifier from header
        auth_context: User authentication context
        tenant_service: Tenant management service

    Returns:
        APIResponse containing tenant quota configuration and usage
    """
    try:
        # Validate tenant access
        if auth_context.tenant_id != target_tenant_id and not auth_context.has_permission("tenant:read_all"):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Access denied to tenant resources"
            )

        # Get tenant with quotas
        tenant = await tenant_service.get_tenant(
            tenant_id=target_tenant_id,
            user_context=auth_context.dict(),
            include_relationships=True
        )

        if not tenant:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Tenant {target_tenant_id} not found"
            )

        return create_success_response(
            data=tenant.quotas if hasattr(tenant, 'quotas') else {},
            message="Tenant quotas retrieved successfully"
        )

    except NotFoundError as e:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(e)
        )
    except UnauthorizedError as e:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=str(e)
        )
    except ServiceError as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve tenant quotas"
        )


@router.get(
    "/stats/overview",
    response_model=APIResponse[dict],
    summary="Get tenant statistics",
    description="Get comprehensive tenant statistics and metrics (admin only)"
)
async def get_tenant_statistics(
    tenant_id: Annotated[str, Header(alias="X-Tenant-ID")],
    auth_context: Annotated[AuthContext, Depends(get_auth_context)],
    tenant_service: Annotated[TenantService, Depends(get_tenant_service)]
) -> APIResponse[dict]:
    """
    Get comprehensive tenant statistics

    Args:
        tenant_id: Requesting tenant identifier from header
        auth_context: User authentication context
        tenant_service: Tenant management service

    Returns:
        APIResponse containing comprehensive tenant statistics and metrics
    """
    try:
        # Validate admin permissions
        if not auth_context.has_permission("tenant:stats"):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Admin permissions required for tenant statistics"
            )

        # Get statistics with business logic
        stats = await tenant_service.get_tenant_statistics(auth_context.dict())

        return create_success_response(
            data=stats,
            message="Tenant statistics retrieved successfully"
        )

    except UnauthorizedError as e:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=str(e)
        )
    except ServiceError as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve tenant statistics"
        )