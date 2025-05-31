"""
Session API Routes
REST API endpoints for session management operations.
"""

from fastapi import APIRouter, Depends, Header, Query, HTTPException, status
from typing import Annotated, Optional, Dict, Any
from datetime import datetime

from src.api.responses.api_response import APIResponse, create_success_response
from src.api.middleware.auth_middleware import get_auth_context, AuthContext, get_optional_auth_context
from src.api.middleware.tenant_middleware import get_tenant_context, TenantContext
from src.services.session_service import SessionService
from src.dependencies import get_session_service
from src.models.types import ChannelType
from src.services.exceptions import (
    ServiceError, ValidationError, UnauthorizedError, NotFoundError, ConflictError
)
from pydantic import BaseModel, Field

router = APIRouter(prefix="/sessions", tags=["sessions"])


class SessionCreateRequest(BaseModel):
    """Request model for creating sessions"""
    user_id: str
    channel: ChannelType
    context: Optional[Dict[str, Any]] = Field(default_factory=dict)
    metadata: Optional[Dict[str, Any]] = Field(default_factory=dict)


class SessionUpdateRequest(BaseModel):
    """Request model for updating sessions"""
    context: Optional[Dict[str, Any]] = None
    metadata: Optional[Dict[str, Any]] = None
    extend_ttl: Optional[bool] = False


class SessionResponse(BaseModel):
    """Response model for session data"""
    session_id: str
    user_id: str
    tenant_id: str
    channel: ChannelType
    context: Dict[str, Any]
    metadata: Dict[str, Any]
    created_at: datetime
    updated_at: datetime
    expires_at: datetime
    is_active: bool

    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }


@router.post(
    "",
    response_model=APIResponse[SessionResponse],
    status_code=status.HTTP_201_CREATED,
    summary="Create a new session",
    description="Create a new user session for conversation management"
)
async def create_session(
        request: SessionCreateRequest,
        tenant_id: Annotated[str, Header(alias="X-Tenant-ID")],
        auth_context: Annotated[Optional[AuthContext], Depends(get_optional_auth_context)],
        tenant_context: Annotated[TenantContext, Depends(get_tenant_context)],
        session_service: Annotated[SessionService, Depends(get_session_service)]
) -> APIResponse[SessionResponse]:
    """
    Create a new session

    Args:
        request: Session creation data
        tenant_id: Tenant identifier from header
        auth_context: Optional user authentication context
        tenant_context: Tenant context
        session_service: Session management service

    Returns:
        APIResponse containing created session details
    """
    try:
        # Validate tenant access if authenticated
        if auth_context and auth_context.tenant_id != tenant_id:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Access denied to tenant resources"
            )

        # Create session
        session = await session_service.create_session(
            tenant_id=tenant_id,
            user_id=request.user_id,
            channel=request.channel,
            context=request.context,
            metadata=request.metadata
        )

        return create_success_response(
            data=session,
            message="Session created successfully"
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
            detail="Failed to create session"
        )


@router.get(
    "/{session_id}",
    response_model=APIResponse[SessionResponse],
    summary="Get session details",
    description="Retrieve detailed information about a specific session"
)
async def get_session(
        session_id: str,
        tenant_id: Annotated[str, Header(alias="X-Tenant-ID")],
        auth_context: Annotated[Optional[AuthContext], Depends(get_optional_auth_context)],
        session_service: Annotated[SessionService, Depends(get_session_service)]
) -> APIResponse[SessionResponse]:
    """
    Get session details by ID

    Args:
        session_id: Session identifier
        tenant_id: Tenant identifier from header
        auth_context: Optional user authentication context
        session_service: Session management service

    Returns:
        APIResponse containing session details
    """
    try:
        # Validate tenant access if authenticated
        if auth_context and auth_context.tenant_id != tenant_id:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Access denied to tenant resources"
            )

        # Get session
        session = await session_service.get_session(
            session_id=session_id,
            tenant_id=tenant_id
        )

        return create_success_response(
            data=session,
            message="Session retrieved successfully"
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
            detail="Failed to retrieve session"
        )


@router.put(
    "/{session_id}",
    response_model=APIResponse[SessionResponse],
    summary="Update session",
    description="Update session context and metadata"
)
async def update_session(
        session_id: str,
        request: SessionUpdateRequest,
        tenant_id: Annotated[str, Header(alias="X-Tenant-ID")],
        auth_context: Annotated[Optional[AuthContext], Depends(get_optional_auth_context)],
        session_service: Annotated[SessionService, Depends(get_session_service)]
) -> APIResponse[SessionResponse]:
    """
    Update session details

    Args:
        session_id: Session identifier
        request: Session update data
        tenant_id: Tenant identifier from header
        auth_context: Optional user authentication context
        session_service: Session management service

    Returns:
        APIResponse containing updated session details
    """
    try:
        # Validate tenant access if authenticated
        if auth_context and auth_context.tenant_id != tenant_id:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Access denied to tenant resources"
            )

        # Update session
        session = await session_service.update_session(
            session_id=session_id,
            tenant_id=tenant_id,
            context_updates=request.context,
            metadata_updates=request.metadata,
            extend_ttl=request.extend_ttl or False
        )

        return create_success_response(
            data=session,
            message="Session updated successfully"
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
            detail="Failed to update session"
        )


@router.delete(
    "/{session_id}",
    response_model=APIResponse[dict],
    summary="Delete session",
    description="Delete a session and clear all associated data"
)
async def delete_session(
        session_id: str,
        tenant_id: Annotated[str, Header(alias="X-Tenant-ID")],
        auth_context: Annotated[Optional[AuthContext], Depends(get_optional_auth_context)],
        session_service: Annotated[SessionService, Depends(get_session_service)]
) -> APIResponse[dict]:
    """
    Delete a session

    Args:
        session_id: Session identifier
        tenant_id: Tenant identifier from header
        auth_context: Optional user authentication context
        session_service: Session management service

    Returns:
        APIResponse confirming deletion
    """
    try:
        # Validate tenant access if authenticated
        if auth_context and auth_context.tenant_id != tenant_id:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Access denied to tenant resources"
            )

        # Delete session
        await session_service.delete_session(
            session_id=session_id,
            tenant_id=tenant_id
        )

        return create_success_response(
            data={"session_id": session_id, "deleted": True},
            message="Session deleted successfully"
        )

    except NotFoundError as e:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(e)
        )
    except ServiceError as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to delete session"
        )


@router.post(
    "/{session_id}/extend",
    response_model=APIResponse[SessionResponse],
    summary="Extend session",
    description="Extend session expiration time"
)
async def extend_session(
        session_id: str,
        tenant_id: Annotated[str, Header(alias="X-Tenant-ID")],
        auth_context: Annotated[Optional[AuthContext], Depends(get_optional_auth_context)],
        session_service: Annotated[SessionService, Depends(get_session_service)],
        extend_minutes: int = Query(default=60, ge=1, le=1440, description="Minutes to extend")
) -> APIResponse[SessionResponse]:
    """
    Extend session expiration time

    Args:
        session_id: Session identifier
        tenant_id: Tenant identifier from header
        auth_context: Optional user authentication context
        session_service: Session management service
        extend_minutes: Minutes to extend the session

    Returns:
        APIResponse containing updated session details
    """
    try:
        # Validate tenant access if authenticated
        if auth_context and auth_context.tenant_id != tenant_id:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Access denied to tenant resources"
            )

        # Extend session
        session = await session_service.extend_session(
            session_id=session_id,
            tenant_id=tenant_id,
            extend_minutes=extend_minutes
        )

        return create_success_response(
            data=session,
            message=f"Session extended by {extend_minutes} minutes"
        )

    except NotFoundError as e:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(e)
        )
    except ServiceError as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to extend session"
        )


@router.get(
    "/user/{user_id}",
    response_model=APIResponse[list[SessionResponse]],
    summary="Get user sessions",
    description="Retrieve all active sessions for a specific user"
)
async def get_user_sessions(
        user_id: str,
        tenant_id: Annotated[str, Header(alias="X-Tenant-ID")],
        auth_context: Annotated[Optional[AuthContext], Depends(get_optional_auth_context)],
        session_service: Annotated[SessionService, Depends(get_session_service)],
        channel: Optional[ChannelType] = Query(default=None, description="Filter by channel"),
        active_only: bool = Query(default=True, description="Only return active sessions")
) -> APIResponse[list[SessionResponse]]:
    """
    Get all sessions for a user

    Args:
        user_id: User identifier
        tenant_id: Tenant identifier from header
        auth_context: Optional user authentication context
        session_service: Session management service
        channel: Optional channel filter
        active_only: Whether to only return active sessions

    Returns:
        APIResponse containing list of user sessions
    """
    try:
        # Validate tenant access if authenticated
        if auth_context and auth_context.tenant_id != tenant_id:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Access denied to tenant resources"
            )

        # Validate user access if authenticated
        if auth_context and auth_context.user_id != user_id:
            # Check if user has permission to view other users' sessions
            if "sessions:read_all" not in auth_context.permissions:
                raise HTTPException(
                    status_code=status.HTTP_403_FORBIDDEN,
                    detail="Access denied to other users' sessions"
                )

        # Get user sessions
        sessions = await session_service.get_user_sessions(
            user_id=user_id,
            tenant_id=tenant_id,
            channel=channel,
            active_only=active_only
        )

        return create_success_response(
            data=sessions,
            message=f"Retrieved {len(sessions)} sessions for user"
        )

    except UnauthorizedError as e:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=str(e)
        )
    except ServiceError as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve user sessions"
        )


@router.post(
    "/{session_id}/transfer",
    response_model=APIResponse[SessionResponse],
    summary="Transfer session",
    description="Transfer session to a different channel"
)
async def transfer_session(
        session_id: str,
        tenant_id: Annotated[str, Header(alias="X-Tenant-ID")],
        auth_context: Annotated[Optional[AuthContext], Depends(get_optional_auth_context)],
        session_service: Annotated[SessionService, Depends(get_session_service)],
        target_channel: ChannelType = Query(..., description="Target channel for transfer"),
        preserve_context: bool = Query(default=True, description="Whether to preserve session context")
) -> APIResponse[SessionResponse]:
    """
    Transfer session to a different channel

    Args:
        session_id: Session identifier
        tenant_id: Tenant identifier from header
        auth_context: Optional user authentication context
        session_service: Session management service
        target_channel: Channel to transfer session to
        preserve_context: Whether to preserve existing context

    Returns:
        APIResponse containing updated session details
    """
    try:
        # Validate tenant access if authenticated
        if auth_context and auth_context.tenant_id != tenant_id:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Access denied to tenant resources"
            )

        # Transfer session
        session = await session_service.transfer_session(
            session_id=session_id,
            tenant_id=tenant_id,
            target_channel=target_channel,
            preserve_context=preserve_context
        )

        return create_success_response(
            data=session,
            message=f"Session transferred to {target_channel.value}"
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
            detail="Failed to transfer session"
        )


@router.get(
    "",
    response_model=APIResponse[list[SessionResponse]],
    summary="List sessions",
    description="List sessions with optional filtering"
)
async def list_sessions(
        tenant_id: Annotated[str, Header(alias="X-Tenant-ID")],
        auth_context: Annotated[AuthContext, Depends(get_auth_context)],
        session_service: Annotated[SessionService, Depends(get_session_service)],
        channel: Optional[ChannelType] = Query(default=None, description="Filter by channel"),
        active_only: bool = Query(default=True, description="Only return active sessions"),
        limit: int = Query(default=50, ge=1, le=100, description="Maximum number of sessions"),
        offset: int = Query(default=0, ge=0, description="Offset for pagination")
) -> APIResponse[list[SessionResponse]]:
    """
    List sessions with filtering and pagination

    Args:
        tenant_id: Tenant identifier from header
        auth_context: User authentication context
        session_service: Session management service
        channel: Optional channel filter
        active_only: Whether to only return active sessions
        limit: Maximum number of sessions to return
        offset: Pagination offset

    Returns:
        APIResponse containing list of sessions
    """
    try:
        # Validate tenant access
        if auth_context.tenant_id != tenant_id:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Access denied to tenant resources"
            )

        # Check permissions for listing all sessions
        if "sessions:read_all" not in auth_context.permissions:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Insufficient permissions to list sessions"
            )

        # List sessions
        sessions = await session_service.list_sessions(
            tenant_id=tenant_id,
            channel=channel,
            active_only=active_only,
            limit=limit,
            offset=offset
        )

        return create_success_response(
            data=sessions,
            message=f"Retrieved {len(sessions)} sessions"
        )

    except UnauthorizedError as e:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=str(e)
        )
    except ServiceError as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to list sessions"
        )


@router.post(
    "/cleanup",
    response_model=APIResponse[dict],
    summary="Cleanup expired sessions",
    description="Remove expired sessions and clean up resources"
)
async def cleanup_expired_sessions(
        tenant_id: Annotated[str, Header(alias="X-Tenant-ID")],
        auth_context: Annotated[AuthContext, Depends(get_auth_context)],
        session_service: Annotated[SessionService, Depends(get_session_service)],
        force: bool = Query(default=False, description="Force cleanup even if not due")
) -> APIResponse[dict]:
    """
    Cleanup expired sessions

    Args:
        tenant_id: Tenant identifier from header
        auth_context: User authentication context
        session_service: Session management service
        force: Whether to force cleanup regardless of schedule

    Returns:
        APIResponse containing cleanup results
    """
    try:
        # Validate tenant access
        if auth_context.tenant_id != tenant_id:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Access denied to tenant resources"
            )

        # Check permissions for cleanup operations
        if "sessions:cleanup" not in auth_context.permissions:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Insufficient permissions for cleanup operations"
            )

        # Perform cleanup
        result = await session_service.cleanup_expired_sessions(
            tenant_id=tenant_id,
            force=force
        )

        return create_success_response(
            data=result,
            message=f"Cleaned up {result.get('sessions_removed', 0)} expired sessions"
        )

    except UnauthorizedError as e:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=str(e)
        )
    except ServiceError as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to cleanup sessions"
        )