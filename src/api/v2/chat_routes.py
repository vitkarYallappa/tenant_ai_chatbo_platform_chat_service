"""
Chat API Routes
REST API endpoints for chat message operations.
"""

from fastapi import APIRouter, Depends, Header, Query, HTTPException, status
from typing import Annotated, Optional
from datetime import datetime

from src.api.validators.message_validators import (
    SendMessageRequest, MessageResponse, ConversationHistoryResponse,
    BulkMessageRequest, BulkMessageResponse, MessageStatusUpdate,
    MessageFeedback, MessageAnalytics
)
from src.api.responses.api_response import APIResponse, create_success_response, create_error_response
from src.api.middleware.auth_middleware import get_auth_context, AuthContext
from src.api.middleware.rate_limit_middleware import check_rate_limit
from src.api.middleware.tenant_middleware import get_tenant_context, TenantContext
from src.services.message_service import MessageService
from src.services.conversation_service import ConversationService
from src.dependencies import get_message_service, get_conversation_service
from src.models.types import ChannelType
from src.services.exceptions import (
    ServiceError, ValidationError, UnauthorizedError, NotFoundError
)

router = APIRouter(prefix="/chat", tags=["chat"])


@router.post(
    "/message",
    response_model=APIResponse[MessageResponse],
    status_code=status.HTTP_200_OK,
    summary="Send a chat message",
    description="Process an incoming message and generate a bot response"
)
async def send_message(
        request: SendMessageRequest,
        tenant_id: Annotated[str, Header(alias="X-Tenant-ID")],
        auth_context: Annotated[AuthContext, Depends(get_auth_context)],
        tenant_context: Annotated[TenantContext, Depends(get_tenant_context)],
        message_service: Annotated[MessageService, Depends(get_message_service)],
        rate_limit_check: Annotated[bool, Depends(check_rate_limit)]
) -> APIResponse[MessageResponse]:
    """
    Send a message through the chat system

    This endpoint processes incoming messages, generates appropriate responses,
    and handles the complete message lifecycle including:
    - Message validation and normalization
    - Conversation context management
    - Response generation
    - Channel delivery

    Args:
        request: Message content and metadata
        tenant_id: Tenant identifier from header
        auth_context: User authentication context
        tenant_context: Tenant context and configuration
        message_service: Message processing service
        rate_limit_check: Rate limiting validation

    Returns:
        APIResponse containing MessageResponse with bot reply

    Raises:
        400: Invalid request data
        401: Authentication failed
        403: Access denied
        429: Rate limit exceeded
        500: Internal server error
    """
    try:
        # Validate tenant access
        if auth_context.tenant_id != tenant_id:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Access denied to tenant resources"
            )

        # Add tenant_id to request if not present
        if not hasattr(request, 'tenant_id') or not request.tenant_id:
            request.tenant_id = tenant_id

        # Process message through service layer
        response = await message_service.process_message(
            request=request,
            user_context=auth_context.dict()
        )

        return create_success_response(
            data=response,
            message="Message processed successfully"
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
            detail="Message processing failed"
        )


@router.get(
    "/conversations/{conversation_id}/history",
    response_model=APIResponse[ConversationHistoryResponse],
    summary="Get conversation history",
    description="Retrieve paginated conversation message history"
)
async def get_conversation_history(
        conversation_id: str,
        tenant_id: Annotated[str, Header(alias="X-Tenant-ID")],
        auth_context: Annotated[AuthContext, Depends(get_auth_context)],
        conversation_service: Annotated[ConversationService, Depends(get_conversation_service)],
        page: int = Query(default=1, ge=1, description="Page number"),
        page_size: int = Query(default=20, ge=1, le=100, description="Messages per page")
) -> APIResponse[ConversationHistoryResponse]:
    """
    Get conversation message history with pagination

    Args:
        conversation_id: Conversation identifier
        tenant_id: Tenant identifier from header
        auth_context: User authentication context
        conversation_service: Conversation management service
        page: Page number for pagination
        page_size: Number of messages per page

    Returns:
        APIResponse containing paginated conversation history
    """
    try:
        # Validate tenant access
        if auth_context.tenant_id != tenant_id:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Access denied to tenant resources"
            )

        # Get conversation history
        history = await conversation_service.get_conversation_history(
            conversation_id=conversation_id,
            tenant_id=tenant_id,
            page=page,
            page_size=page_size,
            user_context=auth_context.dict()
        )

        return create_success_response(
            data=history,
            message="Conversation history retrieved successfully"
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
            detail="Failed to retrieve conversation history"
        )


@router.get(
    "/conversations",
    response_model=APIResponse[ConversationHistoryResponse],
    summary="List conversations",
    description="Get a list of conversations for the authenticated user"
)
async def list_conversations(
        tenant_id: Annotated[str, Header(alias="X-Tenant-ID")],
        auth_context: Annotated[AuthContext, Depends(get_auth_context)],
        conversation_service: Annotated[ConversationService, Depends(get_conversation_service)],
        status_filter: Optional[str] = Query(default=None, description="Filter by conversation status"),
        channel: Optional[ChannelType] = Query(default=None, description="Filter by channel"),
        limit: int = Query(default=20, ge=1, le=100, description="Number of conversations"),
        offset: int = Query(default=0, ge=0, description="Offset for pagination")
) -> APIResponse[ConversationHistoryResponse]:
    """
    List conversations for the authenticated user

    Args:
        tenant_id: Tenant identifier from header
        auth_context: User authentication context
        conversation_service: Conversation management service
        status_filter: Optional status filter
        channel: Optional channel filter
        limit: Maximum number of conversations to return
        offset: Pagination offset

    Returns:
        APIResponse containing list of conversations
    """
    try:
        # Validate tenant access
        if auth_context.tenant_id != tenant_id:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Access denied to tenant resources"
            )

        # Build filters
        filters = {}
        if status_filter:
            filters["status"] = status_filter
        if channel:
            filters["channel"] = channel

        # Get conversations
        conversations = await conversation_service.list_user_conversations(
            tenant_id=tenant_id,
            user_id=auth_context.user_id,
            filters=filters,
            limit=limit,
            offset=offset
        )

        return create_success_response(
            data=conversations,
            message="Conversations retrieved successfully"
        )

    except UnauthorizedError as e:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=str(e)
        )
    except ServiceError as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve conversations"
        )


@router.post(
    "/conversations/{conversation_id}/close",
    response_model=APIResponse[dict],
    summary="Close conversation",
    description="Mark a conversation as completed"
)
async def close_conversation(
        conversation_id: str,
        tenant_id: Annotated[str, Header(alias="X-Tenant-ID")],
        auth_context: Annotated[AuthContext, Depends(get_auth_context)],
        conversation_service: Annotated[ConversationService, Depends(get_conversation_service)]
) -> APIResponse[dict]:
    """
    Close an active conversation

    Args:
        conversation_id: Conversation identifier
        tenant_id: Tenant identifier from header
        auth_context: User authentication context
        conversation_service: Conversation management service

    Returns:
        APIResponse confirming conversation closure
    """
    try:
        # Validate tenant access
        if auth_context.tenant_id != tenant_id:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Access denied to tenant resources"
            )

        # Close conversation
        result = await conversation_service.close_conversation(
            conversation_id=conversation_id,
            tenant_id=tenant_id,
            user_context=auth_context.dict()
        )

        return create_success_response(
            data={"conversation_id": conversation_id, "status": "closed"},
            message="Conversation closed successfully"
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
            detail="Failed to close conversation"
        )


@router.get(
    "/conversations/{conversation_id}/export",
    summary="Export conversation",
    description="Export conversation data in specified format"
)
async def export_conversation(
        conversation_id: str,
        tenant_id: Annotated[str, Header(alias="X-Tenant-ID")],
        auth_context: Annotated[AuthContext, Depends(get_auth_context)],
        conversation_service: Annotated[ConversationService, Depends(get_conversation_service)],
        format: str = Query(default="json", description="Export format (json, csv, txt)")
):
    """
    Export conversation data for download

    Args:
        conversation_id: Conversation identifier
        tenant_id: Tenant identifier from header
        auth_context: User authentication context
        conversation_service: Conversation management service
        format: Export format (json, csv, txt)

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

        # Validate format
        if format not in ["json", "csv", "txt"]:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Invalid export format. Supported: json, csv, txt"
            )

        # Export conversation
        export_data = await conversation_service.export_conversation(
            conversation_id=conversation_id,
            tenant_id=tenant_id,
            format=format,
            user_context=auth_context.dict()
        )

        # Return file response
        from fastapi.responses import Response

        media_type = {
            "json": "application/json",
            "csv": "text/csv",
            "txt": "text/plain"
        }[format]

        filename = f"conversation_{conversation_id}.{format}"

        return Response(
            content=export_data["content"],
            media_type=media_type,
            headers={
                "Content-Disposition": f"attachment; filename={filename}"
            }
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
            detail="Failed to export conversation"
        )


@router.post(
    "/bulk",
    response_model=APIResponse[BulkMessageResponse],
    summary="Send bulk messages",
    description="Process multiple messages in a single request"
)
async def send_bulk_messages(
        request: BulkMessageRequest,
        tenant_id: Annotated[str, Header(alias="X-Tenant-ID")],
        auth_context: Annotated[AuthContext, Depends(get_auth_context)],
        message_service: Annotated[MessageService, Depends(get_message_service)]
) -> APIResponse[BulkMessageResponse]:
    """
    Send multiple messages in bulk

    Args:
        request: Bulk message request with list of messages
        tenant_id: Tenant identifier from header
        auth_context: User authentication context
        message_service: Message processing service

    Returns:
        APIResponse containing bulk processing results
    """
    try:
        # Validate tenant access
        if auth_context.tenant_id != tenant_id:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Access denied to tenant resources"
            )

        # Process bulk messages
        result = await message_service.process_bulk_messages(
            request=request,
            tenant_id=tenant_id,
            user_context=auth_context.dict()
        )

        return create_success_response(
            data=result,
            message=f"Processed {result.total_messages} messages"
        )

    except ValidationError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    except ServiceError as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Bulk message processing failed"
        )


@router.patch(
    "/messages/{message_id}/status",
    response_model=APIResponse[dict],
    summary="Update message status",
    description="Update delivery status of a message"
)
async def update_message_status(
        message_id: str,
        status_update: MessageStatusUpdate,
        tenant_id: Annotated[str, Header(alias="X-Tenant-ID")],
        auth_context: Annotated[AuthContext, Depends(get_auth_context)],
        message_service: Annotated[MessageService, Depends(get_message_service)]
) -> APIResponse[dict]:
    """
    Update the delivery status of a message

    Args:
        message_id: Message identifier
        status_update: Status update information
        tenant_id: Tenant identifier from header
        auth_context: User authentication context
        message_service: Message processing service

    Returns:
        APIResponse confirming status update
    """
    try:
        # Validate tenant access
        if auth_context.tenant_id != tenant_id:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Access denied to tenant resources"
            )

        # Update message status
        await message_service.update_message_status(
            message_id=message_id,
            status_update=status_update,
            tenant_id=tenant_id
        )

        return create_success_response(
            data={"message_id": message_id, "status": status_update.status},
            message="Message status updated successfully"
        )

    except NotFoundError as e:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(e)
        )
    except ServiceError as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to update message status"
        )


@router.post(
    "/messages/{message_id}/feedback",
    response_model=APIResponse[dict],
    summary="Submit message feedback",
    description="Submit user feedback for a bot message"
)
async def submit_message_feedback(
        message_id: str,
        feedback: MessageFeedback,
        tenant_id: Annotated[str, Header(alias="X-Tenant-ID")],
        auth_context: Annotated[AuthContext, Depends(get_auth_context)],
        message_service: Annotated[MessageService, Depends(get_message_service)]
) -> APIResponse[dict]:
    """
    Submit feedback for a message

    Args:
        message_id: Message identifier
        feedback: User feedback data
        tenant_id: Tenant identifier from header
        auth_context: User authentication context
        message_service: Message processing service

    Returns:
        APIResponse confirming feedback submission
    """
    try:
        # Validate tenant access
        if auth_context.tenant_id != tenant_id:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Access denied to tenant resources"
            )

        # Submit feedback
        await message_service.submit_message_feedback(
            message_id=message_id,
            feedback=feedback,
            tenant_id=tenant_id,
            user_id=auth_context.user_id
        )

        return create_success_response(
            data={"message_id": message_id, "feedback_submitted": True},
            message="Feedback submitted successfully"
        )

    except NotFoundError as e:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(e)
        )
    except ServiceError as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to submit feedback"
        )


@router.get(
    "/messages/{message_id}/analytics",
    response_model=APIResponse[MessageAnalytics],
    summary="Get message analytics",
    description="Retrieve analytics data for a specific message"
)
async def get_message_analytics(
        message_id: str,
        tenant_id: Annotated[str, Header(alias="X-Tenant-ID")],
        auth_context: Annotated[AuthContext, Depends(get_auth_context)],
        message_service: Annotated[MessageService, Depends(get_message_service)]
) -> APIResponse[MessageAnalytics]:
    """
    Get analytics data for a message

    Args:
        message_id: Message identifier
        tenant_id: Tenant identifier from header
        auth_context: User authentication context
        message_service: Message processing service

    Returns:
        APIResponse containing message analytics
    """
    try:
        # Validate tenant access
        if auth_context.tenant_id != tenant_id:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Access denied to tenant resources"
            )

        # Get message analytics
        analytics = await message_service.get_message_analytics(
            message_id=message_id,
            tenant_id=tenant_id
        )

        return create_success_response(
            data=analytics,
            message="Message analytics retrieved successfully"
        )

    except NotFoundError as e:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(e)
        )
    except ServiceError as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve message analytics"
        )