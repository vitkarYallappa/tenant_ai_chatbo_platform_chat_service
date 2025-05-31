"""
Logging Middleware
Structured logging middleware for request/response tracking and performance monitoring.
"""

import time
import json
from typing import Dict, Any, Optional, Set
from fastapi import Request, Response
from uuid import uuid4
import structlog

logger = structlog.get_logger()


class LoggingMiddleware:
    """Middleware for structured request/response logging"""

    def __init__(
            self,
            app,
            exclude_paths: Optional[Set[str]] = None,
            include_request_body: bool = False,
            include_response_body: bool = False,
            sanitize_headers: bool = True,
            max_body_size: int = 1024
    ):
        self.app = app
        self.exclude_paths = exclude_paths or {"/health", "/metrics", "/docs", "/openapi.json"}
        self.include_request_body = include_request_body
        self.include_response_body = include_response_body
        self.sanitize_headers = sanitize_headers
        self.max_body_size = max_body_size

        # Headers to sanitize for security
        self.sensitive_headers = {
            "authorization", "x-api-key", "cookie", "set-cookie",
            "x-auth-token", "x-session-id", "x-csrf-token"
        }

    async def __call__(self, request: Request, call_next):
        """Process request through logging middleware"""

        # Skip logging for excluded paths
        if request.url.path in self.exclude_paths:
            return await call_next(request)

        # Generate correlation ID if not present
        correlation_id = request.headers.get("X-Correlation-ID") or str(uuid4())
        request.state.correlation_id = correlation_id
        request.state.trace_id = getattr(request.state, "trace_id", str(uuid4()))

        # Record start time
        start_time = time.time()

        # Log incoming request
        await self._log_request(request, correlation_id)

        # Process request
        try:
            response = await call_next(request)

            # Calculate processing time
            processing_time = time.time() - start_time

            # Add correlation headers to response
            response.headers["X-Correlation-ID"] = correlation_id
            response.headers["X-Trace-ID"] = request.state.trace_id
            response.headers["X-Processing-Time"] = f"{processing_time:.3f}s"

            # Log response
            await self._log_response(request, response, processing_time)

            return response

        except Exception as e:
            # Calculate processing time for error cases
            processing_time = time.time() - start_time

            # Log error
            await self._log_error(request, e, processing_time)

            # Re-raise the exception to be handled by error middleware
            raise

    async def _log_request(self, request: Request, correlation_id: str):
        """Log incoming request details"""

        # Extract basic request info
        request_data = {
            "event": "request_started",
            "correlation_id": correlation_id,
            "trace_id": request.state.trace_id,
            "method": request.method,
            "url": str(request.url),
            "path": request.url.path,
            "query_params": dict(request.query_params),
            "client_ip": self._get_client_ip(request),
            "user_agent": request.headers.get("user-agent"),
            "content_type": request.headers.get("content-type"),
            "content_length": request.headers.get("content-length"),
            "tenant_id": getattr(request.state, "tenant_id", None)
        }

        # Add sanitized headers
        if self.sanitize_headers:
            request_data["headers"] = self._sanitize_headers(dict(request.headers))

        # Add request body if enabled and size is reasonable
        if self.include_request_body:
            request_data["body"] = await self._get_request_body(request)

        logger.info("Incoming request", **request_data)

    async def _log_response(
            self,
            request: Request,
            response: Response,
            processing_time: float
    ):
        """Log response details"""

        response_data = {
            "event": "request_completed",
            "correlation_id": request.state.correlation_id,
            "trace_id": request.state.trace_id,
            "method": request.method,
            "path": request.url.path,
            "status_code": response.status_code,
            "processing_time_ms": round(processing_time * 1000, 2),
            "response_size": len(response.body) if hasattr(response, "body") else None,
            "tenant_id": getattr(request.state, "tenant_id", None)
        }

        # Add performance metrics
        if processing_time > 5.0:  # Slow request threshold
            response_data["performance_warning"] = "slow_request"

        # Add response headers
        if self.sanitize_headers:
            response_data["response_headers"] = self._sanitize_headers(
                dict(response.headers)
            )

        # Add response body if enabled and size is reasonable
        if self.include_response_body and hasattr(response, "body"):
            response_data["response_body"] = self._get_response_body(response)

        # Log with appropriate level based on status code
        if response.status_code >= 500:
            logger.error("Request completed with server error", **response_data)
        elif response.status_code >= 400:
            logger.warning("Request completed with client error", **response_data)
        else:
            logger.info("Request completed successfully", **response_data)

    async def _log_error(
            self,
            request: Request,
            exception: Exception,
            processing_time: float
    ):
        """Log request that resulted in error"""

        error_data = {
            "event": "request_failed",
            "correlation_id": request.state.correlation_id,
            "trace_id": request.state.trace_id,
            "method": request.method,
            "path": request.url.path,
            "error_type": type(exception).__name__,
            "error_message": str(exception),
            "processing_time_ms": round(processing_time * 1000, 2),
            "tenant_id": getattr(request.state, "tenant_id", None)
        }

        logger.error("Request failed with exception", **error_data)

    def _get_client_ip(self, request: Request) -> str:
        """Extract client IP address from request"""
        # Check for IP in various headers (for reverse proxy setups)
        forwarded_for = request.headers.get("x-forwarded-for")
        if forwarded_for:
            return forwarded_for.split(",")[0].strip()

        real_ip = request.headers.get("x-real-ip")
        if real_ip:
            return real_ip

        return request.client.host if request.client else "unknown"

    def _sanitize_headers(self, headers: Dict[str, str]) -> Dict[str, str]:
        """Sanitize sensitive headers for logging"""
        sanitized = {}

        for key, value in headers.items():
            key_lower = key.lower()
            if key_lower in self.sensitive_headers:
                # Show only first and last 4 characters for sensitive headers
                if len(value) > 8:
                    sanitized[key] = f"{value[:4]}...{value[-4:]}"
                else:
                    sanitized[key] = "***"
            else:
                sanitized[key] = value

        return sanitized

    async def _get_request_body(self, request: Request) -> Optional[str]:
        """Safely extract request body for logging"""
        try:
            if request.headers.get("content-type", "").startswith("application/json"):
                body = await request.body()
                if len(body) <= self.max_body_size:
                    return body.decode("utf-8")
                else:
                    return f"<body too large: {len(body)} bytes>"
            else:
                return "<non-json body>"
        except Exception:
            return "<unable to read body>"

    def _get_response_body(self, response: Response) -> Optional[str]:
        """Safely extract response body for logging"""
        try:
            if not hasattr(response, "body"):
                return None

            if len(response.body) <= self.max_body_size:
                return response.body.decode("utf-8")
            else:
                return f"<body too large: {len(response.body)} bytes>"
        except Exception:
            return "<unable to read response body>"


class PerformanceLoggingMiddleware:
    """Middleware for detailed performance logging"""

    def __init__(
            self,
            app,
            slow_request_threshold: float = 1.0,
            enable_detailed_timing: bool = False
    ):
        self.app = app
        self.slow_request_threshold = slow_request_threshold
        self.enable_detailed_timing = enable_detailed_timing

    async def __call__(self, request: Request, call_next):
        """Process request through performance logging"""

        # Record detailed timings if enabled
        timings = {"start": time.time()} if self.enable_detailed_timing else {}

        try:
            if self.enable_detailed_timing:
                timings["middleware_start"] = time.time()

            response = await call_next(request)

            if self.enable_detailed_timing:
                timings["middleware_end"] = time.time()
                timings["total"] = timings["middleware_end"] - timings["start"]

            # Log performance metrics
            await self._log_performance(request, response, timings)

            return response

        except Exception as e:
            if self.enable_detailed_timing:
                timings["error_time"] = time.time()
                timings["total"] = timings["error_time"] - timings["start"]

            await self._log_performance_error(request, e, timings)
            raise

    async def _log_performance(
            self,
            request: Request,
            response: Response,
            timings: Dict[str, float]
    ):
        """Log performance metrics"""

        if not self.enable_detailed_timing:
            return

        total_time = timings.get("total", 0)

        performance_data = {
            "event": "performance_metrics",
            "correlation_id": getattr(request.state, "correlation_id", None),
            "trace_id": getattr(request.state, "trace_id", None),
            "method": request.method,
            "path": request.url.path,
            "status_code": response.status_code,
            "total_time_ms": round(total_time * 1000, 2),
            "is_slow_request": total_time > self.slow_request_threshold
        }

        # Add detailed timing breakdown if available
        if len(timings) > 2:  # More than just start and total
            performance_data["timing_breakdown"] = {
                key: round((value - timings["start"]) * 1000, 2)
                for key, value in timings.items()
                if key not in ["start", "total"]
            }

        if total_time > self.slow_request_threshold:
            logger.warning("Slow request detected", **performance_data)
        else:
            logger.debug("Performance metrics", **performance_data)

    async def _log_performance_error(
            self,
            request: Request,
            exception: Exception,
            timings: Dict[str, float]
    ):
        """Log performance metrics for failed requests"""

        if not self.enable_detailed_timing:
            return

        total_time = timings.get("total", 0)

        performance_data = {
            "event": "performance_metrics_error",
            "correlation_id": getattr(request.state, "correlation_id", None),
            "trace_id": getattr(request.state, "trace_id", None),
            "method": request.method,
            "path": request.url.path,
            "error_type": type(exception).__name__,
            "total_time_ms": round(total_time * 1000, 2),
            "failed_fast": total_time < 0.1  # Failed very quickly
        }

        logger.warning("Performance metrics for failed request", **performance_data)


class AuditLoggingMiddleware:
    """Middleware for audit logging of sensitive operations"""

    def __init__(
            self,
            app,
            audit_paths: Optional[Set[str]] = None,
            audit_methods: Optional[Set[str]] = None
    ):
        self.app = app
        # Paths that require audit logging
        self.audit_paths = audit_paths or {
            "/api/v2/chat/message",
            "/api/v2/conversations",
            "/api/v2/integrations",
            "/api/v2/config"
        }
        # HTTP methods that require audit logging
        self.audit_methods = audit_methods or {"POST", "PUT", "PATCH", "DELETE"}

    async def __call__(self, request: Request, call_next):
        """Process request through audit logging"""

        # Check if this request needs audit logging
        needs_audit = (
                request.method in self.audit_methods or
                any(request.url.path.startswith(path) for path in self.audit_paths)
        )

        if needs_audit:
            await self._log_audit_start(request)

        try:
            response = await call_next(request)

            if needs_audit:
                await self._log_audit_success(request, response)

            return response

        except Exception as e:
            if needs_audit:
                await self._log_audit_failure(request, e)
            raise

    async def _log_audit_start(self, request: Request):
        """Log start of auditable operation"""

        audit_data = {
            "event": "audit_operation_start",
            "correlation_id": getattr(request.state, "correlation_id", None),
            "trace_id": getattr(request.state, "trace_id", None),
            "method": request.method,
            "path": request.url.path,
            "client_ip": self._get_client_ip(request),
            "user_agent": request.headers.get("user-agent"),
            "tenant_id": getattr(request.state, "tenant_id", None),
            "timestamp": time.time()
        }

        # Add authentication context if available
        if hasattr(request.state, "auth_context"):
            auth_context = request.state.auth_context
            audit_data.update({
                "user_id": getattr(auth_context, "user_id", None),
                "user_role": getattr(auth_context, "role", None),
                "auth_method": getattr(auth_context, "token_type", None)
            })

        logger.info("Audit: Operation started", **audit_data)

    async def _log_audit_success(self, request: Request, response: Response):
        """Log successful completion of auditable operation"""

        audit_data = {
            "event": "audit_operation_success",
            "correlation_id": getattr(request.state, "correlation_id", None),
            "trace_id": getattr(request.state, "trace_id", None),
            "method": request.method,
            "path": request.url.path,
            "status_code": response.status_code,
            "tenant_id": getattr(request.state, "tenant_id", None),
            "timestamp": time.time()
        }

        logger.info("Audit: Operation completed successfully", **audit_data)

    async def _log_audit_failure(self, request: Request, exception: Exception):
        """Log failed auditable operation"""

        audit_data = {
            "event": "audit_operation_failure",
            "correlation_id": getattr(request.state, "correlation_id", None),
            "trace_id": getattr(request.state, "trace_id", None),
            "method": request.method,
            "path": request.url.path,
            "error_type": type(exception).__name__,
            "error_message": str(exception),
            "tenant_id": getattr(request.state, "tenant_id", None),
            "timestamp": time.time()
        }

        logger.warning("Audit: Operation failed", **audit_data)

    def _get_client_ip(self, request: Request) -> str:
        """Extract client IP address from request"""
        forwarded_for = request.headers.get("x-forwarded-for")
        if forwarded_for:
            return forwarded_for.split(",")[0].strip()

        real_ip = request.headers.get("x-real-ip")
        if real_ip:
            return real_ip

        return request.client.host if request.client else "unknown"


# Utility function to configure structured logging
def configure_structured_logging():
    """Configure structlog for the application"""
    import structlog

    structlog.configure(
        processors=[
            structlog.stdlib.filter_by_level,
            structlog.stdlib.add_logger_name,
            structlog.stdlib.add_log_level,
            structlog.stdlib.PositionalArgumentsFormatter(),
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            structlog.processors.UnicodeDecoder(),
            structlog.processors.JSONRenderer()
        ],
        context_class=dict,
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=True,
    )