"""
Rate Limiting Middleware
Rate limiting middleware using Redis for tracking and enforcement.
"""

from datetime import datetime
from typing import Optional, Dict, Any
from fastapi import Request, HTTPException, status, Depends
from starlette.responses import Response
import structlog

from src.api.middleware.auth_middleware import AuthContext, get_auth_context, get_optional_auth_context
from src.repositories.rate_limit_repository import RateLimitRepository
from src.dependencies import get_rate_limit_repository
from src.config.settings import get_settings

logger = structlog.get_logger()

# Rate limit configuration
RATE_LIMITS = {
    "basic": {
        "requests_per_minute": 60,
        "requests_per_hour": 1000,
        "requests_per_day": 10000
    },
    "standard": {
        "requests_per_minute": 200,
        "requests_per_hour": 5000,
        "requests_per_day": 50000
    },
    "premium": {
        "requests_per_minute": 1000,
        "requests_per_hour": 20000,
        "requests_per_day": 200000
    },
    "enterprise": {
        "requests_per_minute": 5000,
        "requests_per_hour": 100000,
        "requests_per_day": 1000000
    }
}


async def check_rate_limit(
        request: Request,
        rate_limit_repo: RateLimitRepository = Depends(get_rate_limit_repository),
        auth_context: Optional[AuthContext] = Depends(get_optional_auth_context)
) -> bool:
    """
    Check and enforce rate limits for requests

    Args:
        request: FastAPI request object
        rate_limit_repo: Rate limiting repository
        auth_context: Optional authentication context

    Returns:
        True if within rate limit

    Raises:
        HTTPException: If rate limit exceeded
    """
    try:
        # Determine rate limit identifier and tier
        if auth_context:
            identifier = f"user:{auth_context.tenant_id}:{auth_context.user_id}"
            tier = auth_context.rate_limit_tier
            tenant_id = auth_context.tenant_id
        else:
            # Use IP address for unauthenticated requests
            client_ip = get_client_ip(request)
            identifier = f"ip:{client_ip}"
            tier = "basic"
            tenant_id = "anonymous"

        # Get rate limits for tier
        limits = RATE_LIMITS.get(tier, RATE_LIMITS["basic"])

        # Check rate limits for different windows
        rate_limit_results = {}

        # Check per-minute limit
        allowed_minute, count_minute, reset_minute = await rate_limit_repo.check_rate_limit(
            tenant_id=tenant_id,
            identifier=f"{identifier}:minute",
            limit=limits["requests_per_minute"],
            window_seconds=60
        )

        rate_limit_results["minute"] = {
            "allowed": allowed_minute,
            "count": count_minute,
            "limit": limits["requests_per_minute"],
            "reset": reset_minute
        }

        # Check per-hour limit
        allowed_hour, count_hour, reset_hour = await rate_limit_repo.check_rate_limit(
            tenant_id=tenant_id,
            identifier=f"{identifier}:hour",
            limit=limits["requests_per_hour"],
            window_seconds=3600
        )

        rate_limit_results["hour"] = {
            "allowed": allowed_hour,
            "count": count_hour,
            "limit": limits["requests_per_hour"],
            "reset": reset_hour
        }

        # Check per-day limit
        allowed_day, count_day, reset_day = await rate_limit_repo.check_rate_limit(
            tenant_id=tenant_id,
            identifier=f"{identifier}:day",
            limit=limits["requests_per_day"],
            window_seconds=86400
        )

        rate_limit_results["day"] = {
            "allowed": allowed_day,
            "count": count_day,
            "limit": limits["requests_per_day"],
            "reset": reset_day
        }

        # Check if any limit is exceeded
        exceeded_limits = []
        for window, result in rate_limit_results.items():
            if not result["allowed"]:
                exceeded_limits.append(window)

        if exceeded_limits:
            logger.warning(
                "Rate limit exceeded",
                identifier=identifier,
                tier=tier,
                exceeded_limits=exceeded_limits,
                rate_limit_results=rate_limit_results
            )

            # Find the most restrictive limit for error response
            most_restrictive = rate_limit_results["minute"]
            for window in ["hour", "day"]:
                if window in exceeded_limits:
                    most_restrictive = rate_limit_results[window]
                    break

            raise HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                detail=f"Rate limit exceeded. Try again in {most_restrictive['reset'] - int(datetime.utcnow().timestamp())} seconds",
                headers={
                    "X-RateLimit-Limit": str(most_restrictive["limit"]),
                    "X-RateLimit-Remaining": str(max(0, most_restrictive["limit"] - most_restrictive["count"])),
                    "X-RateLimit-Reset": str(most_restrictive["reset"]),
                    "X-RateLimit-Type": tier,
                    "Retry-After": str(most_restrictive["reset"] - int(datetime.utcnow().timestamp()))
                }
            )

        # Add rate limit headers to response
        # Note: These will be added by response middleware
        request.state.rate_limit_headers = {
            "X-RateLimit-Limit": str(limits["requests_per_minute"]),
            "X-RateLimit-Remaining": str(max(0, limits["requests_per_minute"] - count_minute)),
            "X-RateLimit-Reset": str(reset_minute),
            "X-RateLimit-Type": tier
        }

        logger.debug(
            "Rate limit check passed",
            identifier=identifier,
            tier=tier,
            counts={
                "minute": count_minute,
                "hour": count_hour,
                "day": count_day
            }
        )

        return True

    except HTTPException:
        raise
    except Exception as e:
        logger.error("Rate limit check failed", error=str(e))
        # Allow request on error to avoid blocking legitimate traffic
        return True


def get_client_ip(request: Request) -> str:
    """
    Extract client IP address from request

    Args:
        request: FastAPI request object

    Returns:
        Client IP address
    """
    # Check for IP in various headers (for reverse proxy setups)
    forwarded_for = request.headers.get("X-Forwarded-For")
    if forwarded_for:
        # Take the first IP in the chain
        return forwarded_for.split(",")[0].strip()

    real_ip = request.headers.get("X-Real-IP")
    if real_ip:
        return real_ip

    # Fall back to direct client IP
    return request.client.host if request.client else "unknown"


class RateLimitMiddleware:
    """Middleware class for rate limiting"""

    def __init__(self, app):
        self.app = app

    async def __call__(self, request: Request, call_next):
        """Process request through rate limiting"""

        # Skip rate limiting for health checks and internal endpoints
        if request.url.path in ["/health", "/metrics", "/api/v2/health"]:
            response = await call_next(request)
            return response

        try:
            # Check rate limit (this will raise exception if exceeded)
            await check_rate_limit(request)

            # Continue with request processing
            response = await call_next(request)

            # Add rate limit headers if available
            if hasattr(request.state, "rate_limit_headers"):
                for header, value in request.state.rate_limit_headers.items():
                    response.headers[header] = value

            return response

        except HTTPException as e:
            # Rate limit exceeded - return 429 response
            from fastapi.responses import JSONResponse
            return JSONResponse(
                status_code=e.status_code,
                content={
                    "status": "error",
                    "error": {
                        "code": "RATE_LIMIT_EXCEEDED",
                        "message": e.detail
                    }
                },
                headers=e.headers
            )
        except Exception as e:
            logger.error("Rate limit middleware error", error=str(e))
            # Continue with request on middleware error
            response = await call_next(request)
            return response


# Utility functions for specific rate limit checks

async def check_api_key_rate_limit(
        api_key: str,
        tenant_id: str,
        rate_limit_repo: RateLimitRepository
) -> bool:
    """Check rate limit for API key usage"""
    identifier = f"api_key:{api_key}"

    # API keys typically have higher limits
    limits = {
        "requests_per_minute": 2000,
        "requests_per_hour": 50000,
        "requests_per_day": 500000
    }

    # Check minute limit
    allowed, count, reset = await rate_limit_repo.check_rate_limit(
        tenant_id=tenant_id,
        identifier=f"{identifier}:minute",
        limit=limits["requests_per_minute"],
        window_seconds=60
    )

    if not allowed:
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail="API key rate limit exceeded",
            headers={
                "X-RateLimit-Limit": str(limits["requests_per_minute"]),
                "X-RateLimit-Remaining": str(max(0, limits["requests_per_minute"] - count)),
                "X-RateLimit-Reset": str(reset),
                "Retry-After": str(reset - int(datetime.utcnow().timestamp()))
            }
        )

    return True


async def check_webhook_rate_limit(
        source_ip: str,
        webhook_type: str,
        rate_limit_repo: RateLimitRepository
) -> bool:
    """Check rate limit for webhook endpoints"""
    identifier = f"webhook:{webhook_type}:{source_ip}"

    # Webhook limits are more permissive for legitimate webhook sources
    limits = {
        "requests_per_minute": 500,
        "requests_per_hour": 10000,
        "requests_per_day": 100000
    }

    # Check minute limit
    allowed, count, reset = await rate_limit_repo.check_rate_limit(
        tenant_id="webhooks",
        identifier=f"{identifier}:minute",
        limit=limits["requests_per_minute"],
        window_seconds=60
    )

    if not allowed:
        logger.warning(
            "Webhook rate limit exceeded",
            source_ip=source_ip,
            webhook_type=webhook_type,
            count=count,
            limit=limits["requests_per_minute"]
        )
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail="Webhook rate limit exceeded",
            headers={
                "X-RateLimit-Limit": str(limits["requests_per_minute"]),
                "X-RateLimit-Remaining": str(max(0, limits["requests_per_minute"] - count)),
                "X-RateLimit-Reset": str(reset),
                "Retry-After": str(reset - int(datetime.utcnow().timestamp()))
            }
        )

    return True


async def check_bulk_operation_rate_limit(
        auth_context: AuthContext,
        operation_type: str,
        batch_size: int,
        rate_limit_repo: RateLimitRepository
) -> bool:
    """Check rate limit for bulk operations"""
    identifier = f"bulk:{auth_context.tenant_id}:{auth_context.user_id}:{operation_type}"

    # Bulk operations have stricter limits
    base_limits = RATE_LIMITS.get(auth_context.rate_limit_tier, RATE_LIMITS["standard"])

    # Reduce limits for bulk operations
    limits = {
        "requests_per_minute": base_limits["requests_per_minute"] // 5,
        "requests_per_hour": base_limits["requests_per_hour"] // 5,
        "batch_size_per_minute": base_limits["requests_per_minute"] * 10  # Allow more items per minute
    }

    # Check request count limit
    allowed, count, reset = await rate_limit_repo.check_rate_limit(
        tenant_id=auth_context.tenant_id,
        identifier=f"{identifier}:minute",
        limit=limits["requests_per_minute"],
        window_seconds=60
    )

    if not allowed:
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail="Bulk operation rate limit exceeded",
            headers={
                "X-RateLimit-Limit": str(limits["requests_per_minute"]),
                "X-RateLimit-Remaining": str(max(0, limits["requests_per_minute"] - count)),
                "X-RateLimit-Reset": str(reset),
                "Retry-After": str(reset - int(datetime.utcnow().timestamp()))
            }
        )

    # Check batch size limit
    batch_identifier = f"{identifier}:batch_size:minute"
    allowed_batch, batch_count, batch_reset = await rate_limit_repo.check_rate_limit(
        tenant_id=auth_context.tenant_id,
        identifier=batch_identifier,
        limit=limits["batch_size_per_minute"],
        window_seconds=60,
        increment_by=batch_size
    )

    if not allowed_batch:
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail="Bulk operation batch size limit exceeded",
            headers={
                "X-RateLimit-Limit": str(limits["batch_size_per_minute"]),
                "X-RateLimit-Remaining": str(max(0, limits["batch_size_per_minute"] - batch_count)),
                "X-RateLimit-Reset": str(batch_reset),
                "Retry-After": str(batch_reset - int(datetime.utcnow().timestamp()))
            }
        )

    return True


class AdaptiveRateLimiter:
    """Adaptive rate limiting based on system load and user behavior"""

    def __init__(self, base_limits: Dict[str, int]):
        self.base_limits = base_limits
        self.load_factor = 1.0
        self.user_reputation = {}

    async def get_adjusted_limits(
            self,
            identifier: str,
            current_load: float = 1.0,
            user_reputation_score: float = 1.0
    ) -> Dict[str, int]:
        """
        Get rate limits adjusted for current system load and user reputation

        Args:
            identifier: User/API key identifier
            current_load: Current system load factor (0.1 to 2.0)
            user_reputation_score: User reputation score (0.1 to 2.0)

        Returns:
            Adjusted rate limits
        """
        # Adjust limits based on system load (higher load = lower limits)
        load_adjustment = 1.0 / max(current_load, 0.1)

        # Adjust limits based on user reputation (higher reputation = higher limits)
        reputation_adjustment = min(user_reputation_score, 2.0)

        # Combined adjustment factor
        adjustment_factor = load_adjustment * reputation_adjustment

        adjusted_limits = {}
        for window, limit in self.base_limits.items():
            adjusted_limits[window] = max(1, int(limit * adjustment_factor))

        logger.debug(
            "Adjusted rate limits",
            identifier=identifier,
            load_factor=current_load,
            reputation_score=user_reputation_score,
            adjustment_factor=adjustment_factor,
            original_limits=self.base_limits,
            adjusted_limits=adjusted_limits
        )

        return adjusted_limits