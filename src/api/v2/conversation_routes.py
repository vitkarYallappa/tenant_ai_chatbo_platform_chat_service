"""
Conversation API Routes
REST API endpoints for conversation management operations.
"""

from fastapi import APIRouter, Depends, Header, HTTPException, status
from typing import Annotated, Optional

from src.api.validators.conversation_validators import (
    ConversationCreateRequest,
    ConversationUpdateRequest,
    ConversationCloseRequest,
    ConversationResponse,
    ConversationListRequest,
    ConversationListResponse,
    ConversationExportRequest,
    ConversationAnalyticsRequest,
    ConversationAnalyticsResponse,
    ConversationTransferRequest
)
from src.api.responses.api_response import APIResponse, create_success_response, create_paginated_response
from src.api.middleware.auth_middleware import get_auth_context, AuthContext, require_permissions
from src.api.middleware.tenant_middleware import get_tenant_context, TenantContext, require_feature
from src.services.conversation_service import ConversationService
from src.dependencies import get_conversation_service
from src.services.exceptions import (
    ServiceError, ValidationError, UnauthorizedError, NotFoundError, ConflictError
)

router = APIRouter(prefix="/conversations", tags=["conversations"])


@router.post(
    "",
    response_model=APIResponse[ConversationResponse],
    status_code=status.HTTP_201_CREATED,
    summary="Create a new conversation",
    description="Create a new conversation thread"
)
async def create_conversation(
        request: ConversationCreateRequest,
        tenant_id: Annotated[str, Header(alias="X-Tenant-ID")],
        auth_context: Annotated[AuthContext, Depends(get_auth_context)],
        tenant_context: Annotated[TenantContext, Depends(get_tenant_context)],
        conversation_service: Annotated[ConversationService, Depends(get_conversation_service)]
) -> APIResponse[ConversationResponse]:
    """
    Create a new conversation

    Args:
        request: Conversation creation data
        tenant_id: Tenant identifier from header
        auth_context: User authentication context
        tenant_context: Tenant context
        conversation_service: Conversation management service

    Returns:
        APIResponse containing created conversation details
    """
    try:
        # Validate tenant access
        if auth_context.tenant_id != tenant_id:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Access denied to tenant resources"
            )

        # Create conversation
        conversation = await conversation_service.create_conversation(
            request=request,
            tenant_id=tenant_id,
            user_context=auth_context.dict()
        )

        return create_success_response(
            data=conversation,
            message="Conversation created successfully"
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
    except ServiceError as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to create conversation"
        )


@router.get(
    "/{conversation_id}",
    response_model=APIResponse[ConversationResponse],
    summary="Get conversation details",
    description="Retrieve detailed information about a specific conversation"
)
async def get_conversation(
        conversation_id: str,
        tenant_id: Annotated[str, Header(alias="X-Tenant-ID")],
        auth_context: Annotated[AuthContext, Depends(get_auth_context)],
        conversation_service: Annotated[ConversationService, Depends(get_conversation_service)]
) -> APIResponse[ConversationResponse]:
    """
    Get conversation details by ID

    Args:
        conversation_id: Conversation identifier
        tenant_id: Tenant identifier from header
        auth_context: User authentication context
        conversation_service: Conversation management service

    Returns:
        APIResponse containing conversation details
    """
    try:
        # Validate tenant access
        if auth_context.tenant_id != tenant_id:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Access denied to tenant resources"
            )

        # Get conversation
        conversation = await conversation_service.get_conversation(
            conversation_id=conversation_id,
            tenant_id=tenant_id,
            user_context=auth_context.dict()
        )

        return create_success_response(
            data=conversation,
            message="Conversation retrieved successfully"
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
            detail="Failed to retrieve conversation"
        )


@router.put(
    "/{conversation_id}",
    response_model=APIResponse[ConversationResponse],
    summary="Update conversation",
    description="Update conversation details and metadata"
)
async def update_conversation(
        conversation_id: str,
        request: ConversationUpdateRequest,
        tenant_id: Annotated[str, Header(alias="X-Tenant-ID")],
        auth_context: Annotated[AuthContext, Depends(get_auth_context)],
        conversation_service: Annotated[ConversationService, Depends(get_conversation_service)]
) -> APIResponse[ConversationResponse]:
    """
    Update conversation details

    Args:
        conversation_id: Conversation identifier
        request: Conversation update data
        tenant_id: Tenant identifier from header
        auth_context: User authentication context
        conversation_service: Conversation management service

    Returns:
        APIResponse containing updated conversation details
    """
    try:
        # Validate tenant access
        if auth_context.tenant_id != tenant_id:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Access denied to tenant resources"
            )

        # Update conversation
        conversation = await conversation_service.update_conversation(
            conversation_id=conversation_id,
            request=request,
            tenant_id=tenant_id,
            user_context=auth_context.dict()
        )

        return create_success_response(
            data=conversation,
            message="Conversation updated successfully"
        )

    except NotFoundError as e:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(e)
        )
    except ValidationError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
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
            detail="Failed to update conversation"
        )


@router.post(
    "/{conversation_id}/close",
    response_model=APIResponse[ConversationResponse],
    summary="Close conversation",
    description="Close a conversation and mark it as completed"
)
async def close_conversation(
        conversation_id: str,
        request: ConversationCloseRequest,
        tenant_id: Annotated[str, Header(alias="X-Tenant-ID")],
        auth_context: Annotated[AuthContext, Depends(get_auth_context)],
        conversation_service: Annotated[ConversationService, Depends(get_conversation_service)]
) -> APIResponse[ConversationResponse]:
    """
    Close a conversation

    Args:
        conversation_id: Conversation identifier
        request: Conversation closure data
        tenant_id: Tenant identifier from header
        auth_context: User authentication context
        conversation_service: Conversation management service

    Returns:
        APIResponse containing closed conversation details
    """
    try:
        # Validate tenant access
        if auth_context.tenant_id != tenant_id:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Access denied to tenant resources"
            )

        # Close conversation
        conversation = await conversation_service.close_conversation_with_details(
            conversation_id=conversation_id,
            request=request,
            tenant_id=tenant_id,
            user_context=auth_context.dict()
        )

        return create_success_response(
            data=conversation,
            message="Conversation closed successfully"
        )

    except NotFoundError as e:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(e)
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
    except ServiceError as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to close conversation"
        )


@router.post(
    "/search",
    response_model=APIResponse[ConversationListResponse],
    summary="Search conversations",
    description="Search and filter conversations with pagination"
)
async def search_conversations(
        request: ConversationListRequest,
        tenant_id: Annotated[str, Header(alias="X-Tenant-ID")],
        auth_context: Annotated[AuthContext, Depends(get_auth_context)],
        conversation_service: Annotated[ConversationService, Depends(get_conversation_service)]
) -> APIResponse[ConversationListResponse]:
    """
    Search conversations with filters and pagination

    Args:
        request: Search and pagination parameters
        tenant_id: Tenant identifier from header
        auth_context: User authentication context
        conversation_service: Conversation management service

    Returns:
        APIResponse containing paginated conversation list
    """
    try:
        # Validate tenant access
        if auth_context.tenant_id != tenant_id:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Access denied to tenant resources"
            )

        # Search conversations
        result = await conversation_service.search_conversations(
            request=request,
            tenant_id=tenant_id,
            user_context=auth_context.dict()
        )

        return create_success_response(
            data=result,
            message="Conversations retrieved successfully"
        )

    except ValidationError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    except ServiceError as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to search conversations"
        )


@router.post(
    "/{conversation_id}/export",
    summary="Export conversation",
    description="Export conversation data in various formats"
)
async def export_conversation(
        conversation_id: str,
        request: ConversationExportRequest,
        tenant_id: Annotated[str, Header(alias="X-Tenant-ID")],
        auth_context: Annotated[AuthContext, Depends(get_auth_context)],
        conversation_service: Annotated[ConversationService, Depends(get_conversation_service)]
):
    """
    Export conversation data

    Args:
        conversation_id: Conversation identifier
        request: Export configuration
        tenant_id: Tenant identifier from header
        auth_context: User authentication context
        conversation_service: Conversation management service

    Returns:
        File download response with conversation data
    """
    try:
        # Validate tenant access
        if auth_context.tenant_id != tenant_id:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Access denied to tenant resources"
            )

        # Export conversation
        export_data = await conversation_service.export_conversation(
            conversation_id=conversation_id,
            request=request,
            tenant_id=tenant_id,
            user_context=auth_context.dict()
        )

        # Return file response
        from fastapi.responses import Response

        media_types = {
            "json": "application/json",
            "csv": "text/csv",
            "txt": "text/plain",
            "pdf": "application/pdf"
        }

        filename = f"conversation_{conversation_id}.{request.format}"

        return Response(
            content=export_data["content"],
            media_type=media_types.get(request.format, "application/octet-stream"),
            headers={
                "Content-Disposition": f"attachment; filename={filename}",
                "Content-Length": str(len(export_data["content"]))
            }
        )

    except NotFoundError as e:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(e)
        )
    except ValidationError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    except ServiceError as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to export conversation"
        )


@router.post(
    "/analytics",
    response_model=APIResponse[ConversationAnalyticsResponse],
    summary="Get conversation analytics",
    description="Retrieve analytics data for conversations",
    dependencies=[Depends(require_feature("analytics"))]
)
async def get_conversation_analytics(
        request: ConversationAnalyticsRequest,
        tenant_id: Annotated[str, Header(alias="X-Tenant-ID")],
        auth_context: Annotated[AuthContext, Depends(get_auth_context)],
        conversation_service: Annotated[ConversationService, Depends(get_conversation_service)]
) -> APIResponse[ConversationAnalyticsResponse]:
    """
    Get conversation analytics data

    Args:
        request: Analytics request parameters
        tenant_id: Tenant identifier from header
        auth_context: User authentication context
        conversation_service: Conversation management service

    Returns:
        APIResponse containing analytics data
    """
    try:
        # Validate tenant access
        if auth_context.tenant_id != tenant_id:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Access denied to tenant resources"
            )

        # Get analytics
        analytics = await conversation_service.get_conversation_analytics(
            request=request,
            tenant_id=tenant_id,
            user_context=auth_context.dict()
        )

        return create_success_response(
            data=analytics,
            message="Analytics data retrieved successfully"
        )

    except ValidationError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    except ServiceError as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve analytics"
        )


@router.post(
    "/{conversation_id}/transfer",
    response_model=APIResponse[ConversationResponse],
    summary="Transfer conversation",
    description="Transfer conversation to another agent or department",
    dependencies=[Depends(require_permissions("conversations:transfer"))]
)
async def transfer_conversation(
        conversation_id: str,
        request: ConversationTransferRequest,
        tenant_id: Annotated[str, Header(alias="X-Tenant-ID")],
        auth_context: Annotated[AuthContext, Depends(get_auth_context)],
        conversation_service: Annotated[ConversationService, Depends(get_conversation_service)]
) -> APIResponse[ConversationResponse]:
    """
    Transfer conversation to another agent or department

    Args:
        conversation_id: Conversation identifier
        request: Transfer request details
        tenant_id: Tenant identifier from header
        auth_context: User authentication context
        conversation_service: Conversation management service

    Returns:
        APIResponse containing updated conversation details
    """
    try:
        # Validate tenant access
        if auth_context.tenant_id != tenant_id:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Access denied to tenant resources"
            )

        # Transfer conversation
        conversation = await conversation_service.transfer_conversation(
            conversation_id=conversation_id,
            request=request,
            tenant_id=tenant_id,
            user_context=auth_context.dict()
        )

        return create_success_response(
            data=conversation,
            message="Conversation transferred successfully"
        )

    except NotFoundError as e:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(e)
        )
    except ValidationError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
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
            detail="Failed to transfer conversation"
        )


@router.delete(
    "/{conversation_id}",
    response_model=APIResponse[dict],
    summary="Delete conversation",
    description="Delete a conversation and all associated data",
    dependencies=[Depends(require_permissions("conversations:delete"))]
)
async def delete_conversation(
        conversation_id: str,
        tenant_id: Annotated[str, Header(alias="X-Tenant-ID")],
        auth_context: Annotated[AuthContext, Depends(get_auth_context)],
        conversation_service: Annotated[ConversationService, Depends(get_conversation_service)],
        permanent: bool = False
) -> APIResponse[dict]:
    """
    Delete a conversation

    Args:
        conversation_id: Conversation identifier
        tenant_id: Tenant identifier from header
        auth_context: User authentication context
        conversation_service: Conversation management service
        permanent: Whether to permanently delete (vs soft delete)

    Returns:
        APIResponse confirming deletion
    """
    try:
        # Validate tenant access
        if auth_context.tenant_id != tenant_id:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Access denied to tenant resources"
            )

        # Delete conversation
        await conversation_service.delete_conversation(
            conversation_id=conversation_id,
            tenant_id=tenant_id,
            permanent=permanent,
            user_context=auth_context.dict()
        )

        return create_success_response(
            data={
                "conversation_id": conversation_id,
                "deleted": True,
                "permanent": permanent
            },
            message="Conversation deleted successfully"
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
            detail="Failed to delete conversation"
        )


@router.post(
    "/{conversation_id}/restore",
    response_model=APIResponse[ConversationResponse],
    summary="Restore deleted conversation",
    description="Restore a soft-deleted conversation",
    dependencies=[Depends(require_permissions("conversations:restore"))]
)
async def restore_conversation(
        conversation_id: str,
        tenant_id: Annotated[str, Header(alias="X-Tenant-ID")],
        auth_context: Annotated[AuthContext, Depends(get_auth_context)],
        conversation_service: Annotated[ConversationService, Depends(get_conversation_service)]
) -> APIResponse[ConversationResponse]:
    """
    Restore a soft-deleted conversation

    Args:
        conversation_id: Conversation identifier
        tenant_id: Tenant identifier from header
        auth_context: User authentication context
        conversation_service: Conversation management service

    Returns:
        APIResponse containing restored conversation details
    """
    try:
        # Validate tenant access
        if auth_context.tenant_id != tenant_id:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Access denied to tenant resources"
            )

        # Restore conversation
        conversation = await conversation_service.restore_conversation(
            conversation_id=conversation_id,
            tenant_id=tenant_id,
            user_context=auth_context.dict()
        )

        return create_success_response(
            data=conversation,
            message="Conversation restored successfully"
        )

    except NotFoundError as e:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
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
            detail="Failed to restore conversation"
        )