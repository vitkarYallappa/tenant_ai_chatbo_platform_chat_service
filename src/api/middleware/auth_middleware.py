"""
Authentication Middleware
JWT authentication and authorization middleware for the Chat Service API.
"""

from datetime import datetime
from typing import Dict, Any, Optional, List
from fastapi import Header, HTTPException, status, Depends
from pydantic import BaseModel
import jwt
from jwt.exceptions import InvalidTokenError, ExpiredSignatureError
import structlog

from src.config.settings import get_settings
from src.services.exceptions import UnauthorizedError

logger = structlog.get_logger()


class AuthContext(BaseModel):
    """Authentication context for requests"""
    user_id: str
    tenant_id: str
    email: Optional[str] = None
    role: str = "member"
    permissions: List[str] = []
    scopes: List[str] = []

    # Token metadata
    token_type: str = "bearer"
    issued_at: Optional[datetime] = None
    expires_at: Optional[datetime] = None

    # Rate limiting info
    rate_limit_tier: str = "standard"

    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }


async def get_auth_context(
        authorization: str = Header(alias="Authorization")
) -> AuthContext:
    """
    Extract and validate authentication context from request

    Args:
        authorization: Authorization header with Bearer token

    Returns:
        AuthContext with validated user data

    Raises:
        HTTPException: If authentication fails
    """
    try:
        # Validate authorization header format
        if not authorization.startswith("Bearer "):
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid authorization header format",
                headers={"WWW-Authenticate": "Bearer"}
            )

        # Extract token
        token = authorization.split(" ")[1]

        # Verify and decode token
        payload = await verify_jwt_token(token)

        # Extract user context from token
        auth_context = AuthContext(
            user_id=payload.get("sub"),
            tenant_id=payload.get("tenant_id"),
            email=payload.get("email"),
            role=payload.get("user_role", "member"),
            permissions=payload.get("permissions", []),
            scopes=payload.get("scopes", []),
            rate_limit_tier=payload.get("rate_limit_tier", "standard"),
            issued_at=datetime.fromtimestamp(payload.get("iat", 0)) if payload.get("iat") else None,
            expires_at=datetime.fromtimestamp(payload.get("exp", 0)) if payload.get("exp") else None
        )

        logger.info(
            "Authentication successful",
            user_id=auth_context.user_id,
            tenant_id=auth_context.tenant_id,
            role=auth_context.role
        )

        return auth_context

    except InvalidTokenError as e:
        logger.warning("Invalid JWT token", error=str(e))
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or expired token",
            headers={"WWW-Authenticate": "Bearer"}
        )
    except ExpiredSignatureError:
        logger.warning("Expired JWT token")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Token has expired",
            headers={"WWW-Authenticate": "Bearer"}
        )
    except Exception as e:
        logger.error("Authentication failed", error=str(e))
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication failed",
            headers={"WWW-Authenticate": "Bearer"}
        )


async def verify_jwt_token(token: str) -> Dict[str, Any]:
    """
    Verify JWT token and extract payload

    Args:
        token: JWT token to verify

    Returns:
        Token payload if valid

    Raises:
        InvalidTokenError: If token is invalid
        ExpiredSignatureError: If token is expired
    """
    settings = get_settings()

    try:
        # Decode and verify token
        payload = jwt.decode(
            token,
            settings.JWT_SECRET_KEY,
            algorithms=[settings.JWT_ALGORITHM]
        )

        # Validate required fields
        required_fields = ["sub", "tenant_id", "exp"]
        missing_fields = [field for field in required_fields if field not in payload]

        if missing_fields:
            raise InvalidTokenError(f"Missing required fields: {missing_fields}")

        # Validate token hasn't expired
        current_timestamp = datetime.utcnow().timestamp()
        if payload.get("exp", 0) < current_timestamp:
            raise ExpiredSignatureError("Token has expired")

        return payload

    except (InvalidTokenError, ExpiredSignatureError):
        raise
    except Exception as e:
        logger.error("Token verification failed", error=str(e))
        raise InvalidTokenError(f"Token verification failed: {e}")


async def get_optional_auth_context(
        authorization: Optional[str] = Header(default=None, alias="Authorization")
) -> Optional[AuthContext]:
    """
    Get authentication context if provided, otherwise return None

    Args:
        authorization: Optional authorization header

    Returns:
        AuthContext if token provided and valid, None otherwise
    """
    if not authorization:
        return None

    try:
        return await get_auth_context(authorization)
    except HTTPException:
        return None


def require_permissions(*required_permissions: str):
    """
    Decorator to require specific permissions

    Args:
        required_permissions: List of required permissions

    Returns:
        Dependency function that validates permissions
    """

    def permission_checker(
            auth_context: AuthContext = Depends(get_auth_context)
    ) -> AuthContext:
        """Check if user has required permissions"""
        user_permissions = set(auth_context.permissions)
        missing_permissions = set(required_permissions) - user_permissions

        if missing_permissions:
            logger.warning(
                "Insufficient permissions",
                user_id=auth_context.user_id,
                required=list(required_permissions),
                missing=list(missing_permissions)
            )
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Missing required permissions: {list(missing_permissions)}"
            )

        return auth_context

    return permission_checker


def require_role(required_role: str):
    """
    Decorator to require specific role

    Args:
        required_role: Required user role

    Returns:
        Dependency function that validates role
    """

    def role_checker(
            auth_context: AuthContext = Depends(get_auth_context)
    ) -> AuthContext:
        """Check if user has required role"""
        role_hierarchy = {
            "viewer": 0,
            "member": 1,
            "manager": 2,
            "developer": 3,
            "admin": 4,
            "owner": 5
        }

        user_level = role_hierarchy.get(auth_context.role, 0)
        required_level = role_hierarchy.get(required_role, 999)

        if user_level < required_level:
            logger.warning(
                "Insufficient role",
                user_id=auth_context.user_id,
                user_role=auth_context.role,
                required_role=required_role
            )
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Required role: {required_role}"
            )

        return auth_context

    return role_checker


def require_scopes(*required_scopes: str):
    """
    Decorator to require specific OAuth scopes

    Args:
        required_scopes: List of required scopes

    Returns:
        Dependency function that validates scopes
    """

    def scope_checker(
            auth_context: AuthContext = Depends(get_auth_context)
    ) -> AuthContext:
        """Check if user has required scopes"""
        user_scopes = set(auth_context.scopes)
        missing_scopes = set(required_scopes) - user_scopes

        if missing_scopes:
            logger.warning(
                "Insufficient scopes",
                user_id=auth_context.user_id,
                required=list(required_scopes),
                missing=list(missing_scopes)
            )
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Missing required scopes: {list(missing_scopes)}"
            )

        return auth_context

    return scope_checker


async def validate_tenant_access(
        auth_context: AuthContext,
        tenant_id: str
) -> bool:
    """
    Validate that user has access to the specified tenant

    Args:
        auth_context: User authentication context
        tenant_id: Tenant ID to validate access for

    Returns:
        True if access is allowed

    Raises:
        HTTPException: If access is denied
    """
    if auth_context.tenant_id != tenant_id:
        logger.warning(
            "Tenant access denied",
            user_id=auth_context.user_id,
            user_tenant=auth_context.tenant_id,
            requested_tenant=tenant_id
        )
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Access denied to tenant resources"
        )

    return True


class APIKeyAuthContext(BaseModel):
    """Authentication context for API key requests"""
    api_key_id: str
    tenant_id: str
    permissions: List[str] = []
    rate_limit_tier: str = "standard"
    created_at: Optional[datetime] = None
    expires_at: Optional[datetime] = None


async def get_api_key_auth_context(
        x_api_key: str = Header(alias="X-API-Key")
) -> APIKeyAuthContext:
    """
    Extract and validate API key authentication context

    Args:
        x_api_key: API key from header

    Returns:
        APIKeyAuthContext with validated API key data

    Raises:
        HTTPException: If API key authentication fails
    """
    # This would integrate with the API key validation service
    # For now, implementing a basic structure
    try:
        # Validate API key format
        if not x_api_key.startswith("cb_"):
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid API key format"
            )

        # Here you would validate the API key against the database
        # and extract the associated permissions and tenant info

        # Placeholder implementation - replace with actual API key validation
        api_key_context = APIKeyAuthContext(
            api_key_id="api_key_123",
            tenant_id="tenant_123",
            permissions=["api:read", "api:write"],
            rate_limit_tier="standard"
        )

        logger.info(
            "API key authentication successful",
            api_key_id=api_key_context.api_key_id,
            tenant_id=api_key_context.tenant_id
        )

        return api_key_context

    except Exception as e:
        logger.error("API key authentication failed", error=str(e))
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid API key"
        )


async def get_auth_context_flexible(
        authorization: Optional[str] = Header(default=None, alias="Authorization"),
        x_api_key: Optional[str] = Header(default=None, alias="X-API-Key")
) -> AuthContext:
    """
    Get authentication context from either JWT token or API key

    Args:
        authorization: Optional Authorization header with Bearer token
        x_api_key: Optional API key header

    Returns:
        AuthContext with validated authentication data

    Raises:
        HTTPException: If no valid authentication is provided
    """
    if authorization:
        return await get_auth_context(authorization)
    elif x_api_key:
        api_key_context = await get_api_key_auth_context(x_api_key)
        # Convert API key context to AuthContext
        return AuthContext(
            user_id=f"api_key_{api_key_context.api_key_id}",
            tenant_id=api_key_context.tenant_id,
            role="api_user",
            permissions=api_key_context.permissions,
            scopes=[],
            token_type="api_key",
            rate_limit_tier=api_key_context.rate_limit_tier
        )
    else:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication required. Provide either Bearer token or API key.",
            headers={"WWW-Authenticate": "Bearer"}
        )