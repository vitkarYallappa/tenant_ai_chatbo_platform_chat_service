"""
Webhook API Routes
REST API endpoints for handling incoming webhooks from external platforms.
"""

from fastapi import APIRouter, Depends, Header, Request, HTTPException, status, Query
from fastapi.responses import PlainTextResponse
from typing import Annotated, Optional, Dict, Any
from datetime import datetime
import hmac
import hashlib
import structlog

from src.api.validators.message_validators import WebhookEvent
from src.api.responses.api_response import APIResponse, create_success_response
from src.api.middleware.rate_limit_middleware import check_webhook_rate_limit
from src.api.middleware.tenant_middleware import get_tenant_context, TenantContext, get_optional_auth_context
from src.services.message_service import MessageService
from src.services.webhook_service import WebhookService
from src.dependencies import get_message_service, get_webhook_service, get_rate_limit_repository
from src.config.settings import get_settings
from src.services.exceptions import (
    ServiceError, ValidationError, UnauthorizedError, NotFoundError
)
from pydantic import BaseModel

logger = structlog.get_logger()
router = APIRouter(prefix="/webhooks", tags=["webhooks"])


class WebhookVerificationRequest(BaseModel):
    """Webhook verification request for platform setup"""
    hub_mode: str
    hub_challenge: str
    hub_verify_token: str


class WebhookResponse(BaseModel):
    """Standard webhook response"""
    status: str
    message_id: Optional[str] = None
    processed_at: datetime

    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }


@router.get(
    "/whatsapp/{tenant_id}",
    response_class=PlainTextResponse,
    summary="WhatsApp webhook verification",
    description="Handle WhatsApp webhook verification challenge"
)
async def whatsapp_webhook_verification(
        tenant_id: str,
        request: Request,
        webhook_service: Annotated[WebhookService, Depends(get_webhook_service)],
        hub_mode: str = Query(alias="hub.mode"),
        hub_challenge: str = Query(alias="hub.challenge"),
        hub_verify_token: str = Query(alias="hub.verify_token"),
) -> str:
    """
    Handle WhatsApp webhook verification

    Args:
        tenant_id: Tenant identifier from URL path
        request: FastAPI request object
        hub_mode: Webhook mode from query params
        hub_challenge: Challenge string from WhatsApp
        hub_verify_token: Verification token from WhatsApp
        webhook_service: Webhook processing service

    Returns:
        Challenge string if verification succeeds

    Raises:
        HTTPException: If verification fails
    """
    try:
        logger.info(
            "WhatsApp webhook verification attempt",
            tenant_id=tenant_id,
            hub_mode=hub_mode,
            verify_token_length=len(hub_verify_token)
        )

        # Verify the webhook
        is_valid = await webhook_service.verify_whatsapp_webhook(
            tenant_id=tenant_id,
            hub_mode=hub_mode,
            hub_verify_token=hub_verify_token
        )

        if not is_valid:
            logger.warning(
                "WhatsApp webhook verification failed",
                tenant_id=tenant_id,
                hub_mode=hub_mode
            )
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Webhook verification failed"
            )

        logger.info(
            "WhatsApp webhook verified successfully",
            tenant_id=tenant_id,
            challenge_length=len(hub_challenge)
        )

        return hub_challenge

    except HTTPException:
        raise
    except Exception as e:
        logger.error(
            "WhatsApp webhook verification error",
            tenant_id=tenant_id,
            error=str(e)
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Webhook verification failed"
        )


@router.post(
    "/whatsapp/{tenant_id}",
    response_model=WebhookResponse,
    summary="WhatsApp webhook handler",
    description="Handle incoming WhatsApp messages and events"
)
async def whatsapp_webhook_handler(
        tenant_id: str,
        request: Request,
        webhook_service: Annotated[WebhookService, Depends(get_webhook_service)],
        message_service: Annotated[MessageService, Depends(get_message_service)],
        x_hub_signature_256: Optional[str] = Header(default=None, alias="X-Hub-Signature-256")
) -> WebhookResponse:
    """
    Handle incoming WhatsApp webhook events

    Args:
        tenant_id: Tenant identifier from URL path
        request: FastAPI request object containing webhook data
        webhook_service: Webhook processing service
        message_service: Message processing service
        x_hub_signature_256: Webhook signature for verification

    Returns:
        WebhookResponse confirming processing
    """
    try:
        # Rate limit check for webhooks
        client_ip = request.client.host if request.client else "unknown"
        await check_webhook_rate_limit(
            source_ip=client_ip,
            webhook_type="whatsapp",
            rate_limit_repo=await get_rate_limit_repository()
        )

        # Get raw body for signature verification
        body = await request.body()

        # Verify webhook signature
        if x_hub_signature_256:
            is_valid_signature = await webhook_service.verify_whatsapp_signature(
                tenant_id=tenant_id,
                body=body,
                signature=x_hub_signature_256
            )

            if not is_valid_signature:
                logger.warning(
                    "Invalid WhatsApp webhook signature",
                    tenant_id=tenant_id,
                    client_ip=client_ip
                )
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail="Invalid webhook signature"
                )

        # Parse webhook data
        webhook_data = await request.json()

        logger.info(
            "WhatsApp webhook received",
            tenant_id=tenant_id,
            client_ip=client_ip,
            data_keys=list(webhook_data.keys()) if webhook_data else []
        )

        # Process webhook through service
        result = await webhook_service.process_whatsapp_webhook(
            tenant_id=tenant_id,
            webhook_data=webhook_data,
            client_ip=client_ip
        )

        return WebhookResponse(
            status="processed",
            message_id=result.get("message_id"),
            processed_at=datetime.utcnow()
        )

    except ValidationError as e:
        logger.warning(
            "WhatsApp webhook validation error",
            tenant_id=tenant_id,
            error=str(e)
        )
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    except UnauthorizedError as e:
        logger.warning(
            "WhatsApp webhook unauthorized",
            tenant_id=tenant_id,
            error=str(e)
        )
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=str(e)
        )
    except Exception as e:
        logger.error(
            "WhatsApp webhook processing error",
            tenant_id=tenant_id,
            error=str(e)
        )
        # Return 200 to prevent webhook retries for unrecoverable errors
        return WebhookResponse(
            status="error",
            processed_at=datetime.utcnow()
        )


@router.get(
    "/messenger/{tenant_id}",
    response_class=PlainTextResponse,
    summary="Messenger webhook verification",
    description="Handle Facebook Messenger webhook verification"
)
async def messenger_webhook_verification(
        tenant_id: str,
        request: Request,
        webhook_service: Annotated[WebhookService, Depends(get_webhook_service)],
        hub_mode: str = Query(alias="hub.mode"),
        hub_challenge: str = Query(alias="hub.challenge"),
        hub_verify_token: str = Query(alias="hub.verify_token"),
) -> str:
    """
    Handle Facebook Messenger webhook verification

    Similar to WhatsApp verification but for Messenger platform
    """
    try:
        logger.info(
            "Messenger webhook verification attempt",
            tenant_id=tenant_id,
            hub_mode=hub_mode
        )

        is_valid = await webhook_service.verify_messenger_webhook(
            tenant_id=tenant_id,
            hub_mode=hub_mode,
            hub_verify_token=hub_verify_token
        )

        if not is_valid:
            logger.warning(
                "Messenger webhook verification failed",
                tenant_id=tenant_id
            )
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Webhook verification failed"
            )

        return hub_challenge

    except HTTPException:
        raise
    except Exception as e:
        logger.error(
            "Messenger webhook verification error",
            tenant_id=tenant_id,
            error=str(e)
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Webhook verification failed"
        )


@router.post(
    "/messenger/{tenant_id}",
    response_model=WebhookResponse,
    summary="Messenger webhook handler",
    description="Handle incoming Facebook Messenger messages and events"
)
async def messenger_webhook_handler(
        tenant_id: str,
        request: Request,
        webhook_service: Annotated[WebhookService, Depends(get_webhook_service)],
        x_hub_signature_256: Optional[str] = Header(default=None, alias="X-Hub-Signature-256")
) -> WebhookResponse:
    """
    Handle incoming Facebook Messenger webhook events
    """
    try:
        # Rate limit and signature verification
        client_ip = request.client.host if request.client else "unknown"
        await check_webhook_rate_limit(
            source_ip=client_ip,
            webhook_type="messenger",
            rate_limit_repo=await get_rate_limit_repository()
        )

        body = await request.body()

        if x_hub_signature_256:
            is_valid_signature = await webhook_service.verify_messenger_signature(
                tenant_id=tenant_id,
                body=body,
                signature=x_hub_signature_256
            )

            if not is_valid_signature:
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail="Invalid webhook signature"
                )

        webhook_data = await request.json()

        result = await webhook_service.process_messenger_webhook(
            tenant_id=tenant_id,
            webhook_data=webhook_data,
            client_ip=client_ip
        )

        return WebhookResponse(
            status="processed",
            message_id=result.get("message_id"),
            processed_at=datetime.utcnow()
        )

    except Exception as e:
        logger.error(
            "Messenger webhook processing error",
            tenant_id=tenant_id,
            error=str(e)
        )
        return WebhookResponse(
            status="error",
            processed_at=datetime.utcnow()
        )


@router.post(
    "/slack/{tenant_id}",
    response_model=WebhookResponse,
    summary="Slack webhook handler",
    description="Handle incoming Slack events and slash commands"
)
async def slack_webhook_handler(
        tenant_id: str,
        request: Request,
        webhook_service: Annotated[WebhookService, Depends(get_webhook_service)],
        x_slack_signature: Optional[str] = Header(default=None, alias="X-Slack-Signature"),
        x_slack_request_timestamp: Optional[str] = Header(default=None, alias="X-Slack-Request-Timestamp")
) -> WebhookResponse:
    """
    Handle incoming Slack webhook events

    Args:
        tenant_id: Tenant identifier from URL path
        request: FastAPI request object
        webhook_service: Webhook processing service
        x_slack_signature: Slack signature for verification
        x_slack_request_timestamp: Request timestamp from Slack

    Returns:
        WebhookResponse confirming processing
    """
    try:
        client_ip = request.client.host if request.client else "unknown"
        await check_webhook_rate_limit(
            source_ip=client_ip,
            webhook_type="slack",
            rate_limit_repo=await get_rate_limit_repository()
        )

        body = await request.body()

        # Verify Slack signature if provided
        if x_slack_signature and x_slack_request_timestamp:
            is_valid_signature = await webhook_service.verify_slack_signature(
                tenant_id=tenant_id,
                body=body,
                signature=x_slack_signature,
                timestamp=x_slack_request_timestamp
            )

            if not is_valid_signature:
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail="Invalid webhook signature"
                )

        # Parse webhook data (could be JSON or form-encoded)
        content_type = request.headers.get("content-type", "")
        if content_type.startswith("application/json"):
            webhook_data = await request.json()
        else:
            # Handle form-encoded data from Slack
            form_data = await request.form()
            webhook_data = dict(form_data)

        # Handle Slack URL verification challenge
        if webhook_data.get("type") == "url_verification":
            challenge = webhook_data.get("challenge")
            if challenge:
                from fastapi.responses import PlainTextResponse
                return PlainTextResponse(challenge)

        result = await webhook_service.process_slack_webhook(
            tenant_id=tenant_id,
            webhook_data=webhook_data,
            client_ip=client_ip
        )

        return WebhookResponse(
            status="processed",
            message_id=result.get("message_id"),
            processed_at=datetime.utcnow()
        )

    except Exception as e:
        logger.error(
            "Slack webhook processing error",
            tenant_id=tenant_id,
            error=str(e)
        )
        return WebhookResponse(
            status="error",
            processed_at=datetime.utcnow()
        )


@router.post(
    "/teams/{tenant_id}",
    response_model=WebhookResponse,
    summary="Teams webhook handler",
    description="Handle incoming Microsoft Teams messages and events"
)
async def teams_webhook_handler(
        tenant_id: str,
        request: Request,
        webhook_service: Annotated[WebhookService, Depends(get_webhook_service)],
        authorization: Optional[str] = Header(default=None, alias="Authorization")
) -> WebhookResponse:
    """
    Handle incoming Microsoft Teams webhook events

    Args:
        tenant_id: Tenant identifier from URL path
        request: FastAPI request object
        webhook_service: Webhook processing service
        authorization: Authorization header with JWT token

    Returns:
        WebhookResponse confirming processing
    """
    try:
        client_ip = request.client.host if request.client else "unknown"
        await check_webhook_rate_limit(
            source_ip=client_ip,
            webhook_type="teams",
            rate_limit_repo=await get_rate_limit_repository()
        )

        # Verify Teams JWT token if provided
        if authorization:
            is_valid_token = await webhook_service.verify_teams_token(
                tenant_id=tenant_id,
                authorization=authorization
            )

            if not is_valid_token:
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail="Invalid authorization token"
                )

        webhook_data = await request.json()

        result = await webhook_service.process_teams_webhook(
            tenant_id=tenant_id,
            webhook_data=webhook_data,
            client_ip=client_ip
        )

        return WebhookResponse(
            status="processed",
            message_id=result.get("message_id"),
            processed_at=datetime.utcnow()
        )

    except Exception as e:
        logger.error(
            "Teams webhook processing error",
            tenant_id=tenant_id,
            error=str(e)
        )
        return WebhookResponse(
            status="error",
            processed_at=datetime.utcnow()
        )


@router.post(
    "/generic/{tenant_id}",
    response_model=WebhookResponse,
    summary="Generic webhook handler",
    description="Handle webhooks from custom integrations"
)
async def generic_webhook_handler(
        tenant_id: str,
        request: Request,
        webhook_service: Annotated[WebhookService, Depends(get_webhook_service)],
        integration_id: str = Query(..., description="Integration identifier"),
        signature: Optional[str] = Header(default=None, alias="X-Webhook-Signature")
) -> WebhookResponse:
    """
    Handle generic webhook events from custom integrations

    Args:
        tenant_id: Tenant identifier from URL path
        request: FastAPI request object
        webhook_service: Webhook processing service
        integration_id: Integration identifier from query params
        signature: Optional webhook signature

    Returns:
        WebhookResponse confirming processing
    """
    try:
        client_ip = request.client.host if request.client else "unknown"
        await check_webhook_rate_limit(
            source_ip=client_ip,
            webhook_type="generic",
            rate_limit_repo=await get_rate_limit_repository()
        )

        body = await request.body()

        # Verify signature if provided
        if signature:
            is_valid_signature = await webhook_service.verify_generic_signature(
                tenant_id=tenant_id,
                integration_id=integration_id,
                body=body,
                signature=signature
            )

            if not is_valid_signature:
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail="Invalid webhook signature"
                )

        webhook_data = await request.json()

        result = await webhook_service.process_generic_webhook(
            tenant_id=tenant_id,
            integration_id=integration_id,
            webhook_data=webhook_data,
            client_ip=client_ip
        )

        return WebhookResponse(
            status="processed",
            message_id=result.get("message_id"),
            processed_at=datetime.utcnow()
        )

    except Exception as e:
        logger.error(
            "Generic webhook processing error",
            tenant_id=tenant_id,
            integration_id=integration_id,
            error=str(e)
        )
        return WebhookResponse(
            status="error",
            processed_at=datetime.utcnow()
        )


@router.get(
    "/status/{tenant_id}",
    response_model=APIResponse[Dict[str, Any]],
    summary="Get webhook status",
    description="Get webhook configuration and status for a tenant"
)
async def get_webhook_status(
        tenant_id: str,
        webhook_service: Annotated[WebhookService, Depends(get_webhook_service)],
        tenant_context: Annotated[TenantContext, Depends(get_tenant_context)]
) -> APIResponse[Dict[str, Any]]:
    """
    Get webhook status and configuration

    Args:
        tenant_id: Tenant identifier
        webhook_service: Webhook processing service
        tenant_context: Tenant context

    Returns:
        APIResponse containing webhook status information
    """
    try:
        status_data = await webhook_service.get_webhook_status(tenant_id)

        return create_success_response(
            data=status_data,
            message="Webhook status retrieved successfully"
        )

    except NotFoundError as e:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(e)
        )
    except Exception as e:
        logger.error(
            "Failed to get webhook status",
            tenant_id=tenant_id,
            error=str(e)
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve webhook status"
        )


@router.post(
    "/test/{tenant_id}",
    response_model=APIResponse[Dict[str, Any]],
    summary="Test webhook configuration",
    description="Send a test message through webhook configuration"
)
async def test_webhook(
        tenant_id: str,
        webhook_service: Annotated[WebhookService, Depends(get_webhook_service)],
        tenant_context: Annotated[TenantContext, Depends(get_tenant_context)],
        platform: str = Query(..., description="Platform to test (whatsapp, messenger, slack, teams)")
) -> APIResponse[Dict[str, Any]]:
    """
    Test webhook configuration by sending a test message

    Args:
        tenant_id: Tenant identifier
        webhook_service: Webhook processing service
        tenant_context: Tenant context
        platform: Platform to test

    Returns:
        APIResponse containing test results
    """
    try:
        if platform not in ["whatsapp", "messenger", "slack", "teams"]:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Invalid platform. Supported: whatsapp, messenger, slack, teams"
            )

        test_result = await webhook_service.test_webhook_configuration(
            tenant_id=tenant_id,
            platform=platform
        )

        return create_success_response(
            data=test_result,
            message=f"Webhook test for {platform} completed"
        )

    except ValidationError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    except Exception as e:
        logger.error(
            "Webhook test failed",
            tenant_id=tenant_id,
            platform=platform,
            error=str(e)
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Webhook test failed"
        )
