"""
Chat Service - FastAPI Application Entry Point.

This module provides the main FastAPI application instance with all
middleware, routes, exception handlers, and lifecycle management.
"""

import asyncio
import sys
from contextlib import asynccontextmanager
from typing import Dict, Any

import structlog
import uvicorn
from fastapi import FastAPI, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.responses import JSONResponse
from prometheus_client import generate_latest, CONTENT_TYPE_LATEST

from src.config.settings import get_settings
from src.utils.logger import setup_logging, get_logger
from src.exceptions.base_exceptions import setup_exception_handlers
from src.utils.dependencies import get_health_checker
from src.config.constants import (
    SERVICE_NAME,
    API_VERSION,
    HEALTH_CHECK_INTERVAL,
    STARTUP_TIMEOUT,
    SHUTDOWN_TIMEOUT,
)

# Initialize logger
logger = get_logger(__name__)


class HealthChecker:
    """Health monitoring for service dependencies."""

    def __init__(self):
        self.is_healthy = True
        self.last_check = None
        self.dependencies_status: Dict[str, Dict[str, Any]] = {}

    async def check_dependencies(self) -> Dict[str, Any]:
        """Check health of all service dependencies."""
        import time
        from datetime import datetime

        self.last_check = datetime.utcnow()
        overall_healthy = True

        # This will be expanded in later phases with actual dependency checks
        dependencies = {
            "service": {
                "status": "healthy",
                "latency_ms": 0,
                "last_check": self.last_check.isoformat()
            }
        }

        self.dependencies_status = dependencies
        self.is_healthy = overall_healthy

        return {
            "status": "healthy" if overall_healthy else "unhealthy",
            "timestamp": self.last_check.isoformat(),
            "dependencies": dependencies
        }


# Global health checker instance
health_checker = HealthChecker()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager for startup and shutdown events."""
    # Startup
    logger.info("Starting Chat Service...", version=app.version)

    try:
        await startup_event()
        logger.info("Chat Service startup completed successfully")
        yield
    except Exception as e:
        logger.error("Chat Service startup failed", error=str(e), exc_info=True)
        sys.exit(1)
    finally:
        # Shutdown
        logger.info("Shutting down Chat Service...")
        try:
            await shutdown_event()
            logger.info("Chat Service shutdown completed successfully")
        except Exception as e:
            logger.error("Error during shutdown", error=str(e), exc_info=True)


def create_app() -> FastAPI:
    """Create and configure FastAPI application instance."""
    settings = get_settings()

    # Create FastAPI app
    app = FastAPI(
        title="Chat Service API",
        description="Multi-tenant AI chatbot platform - Chat Service",
        version="2.0.0",
        openapi_url=f"/api/{API_VERSION}/openapi.json",
        docs_url=f"/api/{API_VERSION}/docs",
        redoc_url=f"/api/{API_VERSION}/redoc",
        lifespan=lifespan,
        swagger_ui_parameters={
            "defaultModelsExpandDepth": -1,
            "syntaxHighlight": True,
            "tryItOutEnabled": True,
        }
    )

    # Setup middleware
    setup_middleware(app, settings)

    # Setup exception handlers
    setup_exception_handlers(app)

    # Setup routes
    setup_routes(app)

    return app


def setup_middleware(app: FastAPI, settings) -> None:
    """Configure application middleware."""

    # Trust proxy headers in production
    if settings.ENVIRONMENT == "production":
        app.add_middleware(
            TrustedHostMiddleware,
            allowed_hosts=settings.ALLOWED_HOSTS
        )

    # CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.ALLOWED_ORIGINS,
        allow_credentials=True,
        allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS", "HEAD"],
        allow_headers=[
            "Authorization",
            "Content-Type",
            "X-Tenant-ID",
            "X-Request-ID",
            "X-API-Key",
            "User-Agent",
            "Accept",
            "Accept-Language",
            "Accept-Encoding",
        ],
        expose_headers=[
            "X-Request-ID",
            "X-RateLimit-Limit",
            "X-RateLimit-Remaining",
            "X-RateLimit-Reset",
        ]
    )

    # Request ID middleware
    @app.middleware("http")
    async def add_request_id_middleware(request: Request, call_next):
        """Add unique request ID to all requests."""
        import uuid

        request_id = request.headers.get("X-Request-ID") or str(uuid.uuid4())

        # Add to structlog context
        structlog.contextvars.clear_contextvars()
        structlog.contextvars.bind_contextvars(request_id=request_id)

        response = await call_next(request)
        response.headers["X-Request-ID"] = request_id
        return response

    # Request timing middleware
    @app.middleware("http")
    async def add_process_time_header(request: Request, call_next):
        """Add processing time header to responses."""
        import time

        start_time = time.time()
        response = await call_next(request)
        process_time = time.time() - start_time
        response.headers["X-Process-Time"] = str(round(process_time * 1000, 2))
        return response


def setup_routes(app: FastAPI) -> None:
    """Setup application routes and endpoints."""

    @app.get("/health")
    async def health_check():
        """Basic health check endpoint."""
        return {
            "status": "healthy",
            "service": SERVICE_NAME,
            "version": app.version,
            "timestamp": health_checker.last_check.isoformat() if health_checker.last_check else None
        }

    @app.get("/health/detailed")
    async def detailed_health_check():
        """Detailed health check with dependency status."""
        import psutil
        from datetime import datetime

        health_data = await health_checker.check_dependencies()

        # Add system metrics
        try:
            memory = psutil.virtual_memory()
            cpu_percent = psutil.cpu_percent(interval=1)

            health_data["metrics"] = {
                "memory_usage_mb": round(memory.used / 1024 / 1024, 2),
                "memory_percent": round(memory.percent, 2),
                "cpu_usage_percent": round(cpu_percent, 2),
                "uptime_seconds": round(health_checker.last_check.timestamp() if health_checker.last_check else 0),
                "active_connections": 0  # Will be implemented in later phases
            }
        except ImportError:
            # psutil not available
            health_data["metrics"] = {
                "memory_usage_mb": "N/A",
                "cpu_usage_percent": "N/A",
                "uptime_seconds": "N/A"
            }

        return health_data

    @app.get("/info")
    async def service_info():
        """Service information endpoint."""
        settings = get_settings()

        return {
            "service": SERVICE_NAME,
            "version": app.version,
            "api_version": API_VERSION,
            "environment": settings.ENVIRONMENT.value,  # Convert enum to string
            "debug": settings.DEBUG,
            "docs_url": app.docs_url,
            "openapi_url": app.openapi_url,
        }

    @app.get("/metrics")
    async def prometheus_metrics():
        """Prometheus metrics endpoint."""
        return Response(
            generate_latest(),
            media_type=CONTENT_TYPE_LATEST
        )

    # API version redirect
    @app.get("/api")
    async def api_redirect():
        """Redirect to latest API version."""
        return JSONResponse({
            "message": "Chat Service API",
            "version": app.version,
            "latest_api_version": API_VERSION,
            "docs_url": f"/api/{API_VERSION}/docs",
            "openapi_url": f"/api/{API_VERSION}/openapi.json"
        })


async def startup_event() -> None:
    """Initialize services and connections on application startup."""
    settings = get_settings()

    # Setup logging
    setup_logging(settings.LOG_LEVEL.value)  # Convert enum to string

    logger.info(
        "Chat Service starting",
        version="2.0.0",
        environment=settings.ENVIRONMENT.value,  # Convert enum to string
        debug=settings.DEBUG,
        host=settings.HOST,
        port=settings.PORT
    )

    # Validate configuration
    await validate_configuration(settings)

    # Initialize health checker
    await health_checker.check_dependencies()

    # Start background tasks
    await start_background_tasks()

    logger.info("Chat Service startup completed")


async def shutdown_event() -> None:
    """Clean up resources on application shutdown."""
    logger.info("Starting graceful shutdown...")

    # Stop background tasks
    await stop_background_tasks()

    # Close connections (will be implemented in later phases)
    # await close_database_connections()
    # await close_redis_connections()

    logger.info("Chat Service shutdown complete")


async def validate_configuration(settings) -> None:
    """Validate service configuration."""
    logger.info("Validating configuration...")

    # Check required settings
    required_settings = [
        ("MONGODB_URI", settings.MONGODB_URI),
        ("REDIS_URL", str(settings.REDIS_URL)),  # Convert to string
        ("POSTGRESQL_URI", str(settings.POSTGRESQL_URI)),  # Convert to string
    ]

    missing_settings = []
    for name, value in required_settings:
        if not value or value == "":
            missing_settings.append(name)

    if missing_settings:
        error_msg = f"Missing required configuration: {', '.join(missing_settings)}"
        logger.error("Configuration validation failed", missing=missing_settings)
        raise ValueError(error_msg)

    # Validate environment-specific settings
    if settings.ENVIRONMENT.value == "production":  # Use .value for enum
        if settings.DEBUG:
            logger.warning("Debug mode is enabled in production")

        if "localhost" in settings.ALLOWED_ORIGINS:
            logger.warning("Localhost is allowed in production CORS origins")

    logger.info("Configuration validation completed successfully")


async def start_background_tasks() -> None:
    """Start background tasks and periodic jobs."""
    logger.info("Starting background tasks...")

    # Health check task
    async def periodic_health_check():
        """Periodic health check task."""
        while True:
            try:
                await health_checker.check_dependencies()
                await asyncio.sleep(HEALTH_CHECK_INTERVAL)
            except asyncio.CancelledError:
                logger.info("Health check task cancelled")
                break
            except Exception as e:
                logger.error("Health check task error", error=str(e))
                await asyncio.sleep(HEALTH_CHECK_INTERVAL)

    # Start health check task
    asyncio.create_task(periodic_health_check())

    logger.info("Background tasks started")


async def stop_background_tasks() -> None:
    """Stop all background tasks."""
    logger.info("Stopping background tasks...")

    # Cancel all running tasks
    tasks = [task for task in asyncio.all_tasks() if not task.done()]

    if tasks:
        logger.info(f"Cancelling {len(tasks)} background tasks...")
        for task in tasks:
            task.cancel()

        # Wait for tasks to complete cancellation
        await asyncio.gather(*tasks, return_exceptions=True)

    logger.info("Background tasks stopped")


def main() -> None:
    """Main entry point for running the service."""
    settings = get_settings()

    # Configure uvicorn logging to work with structlog
    log_config = {
        "version": 1,
        "disable_existing_loggers": False,
        "formatters": {
            "default": {
                "()": "uvicorn.logging.DefaultFormatter",
                "fmt": "%(levelprefix)s %(asctime)s %(message)s",
                "datefmt": "%Y-%m-%d %H:%M:%S",
            },
        },
        "handlers": {
            "default": {
                "formatter": "default",
                "class": "logging.StreamHandler",
                "stream": "ext://sys.stdout",
            },
        },
        "root": {
            "level": settings.LOG_LEVEL.value,  # Convert enum to string
            "handlers": ["default"],
        },
    }

    # Configure uvicorn server
    uvicorn_config = {
        "app": "src.main:app",
        "host": settings.HOST,
        "port": settings.PORT,
        "log_level": settings.LOG_LEVEL.value.lower(),  # Convert enum to string and lowercase
        "access_log": True,
        "log_config": log_config,
        "server_header": False,  # Security: don't expose server info
        "date_header": False,  # Performance: don't add date header
    }

    # Development-specific settings
    if settings.ENVIRONMENT.value == "development":  # Use .value for enum
        uvicorn_config.update({
            "reload": settings.DEBUG,
            "reload_dirs": ["src/"],
            "reload_excludes": ["*.pyc", "*.pyo", "__pycache__"],
            "use_colors": True,
        })

    # Production-specific settings
    elif settings.ENVIRONMENT.value == "production":  # Use .value for enum
        uvicorn_config.update({
            "workers": 1,  # Single worker for now, will be configured via deployment
            "loop": "uvloop",
            "http": "httptools",
            "backlog": 2048,
            "keepalive_timeout": 5,
        })

    logger.info(
        "Starting Chat Service server",
        host=settings.HOST,
        port=settings.PORT,
        environment=settings.ENVIRONMENT.value,  # Convert enum to string
        debug=settings.DEBUG
    )

    # Run the server
    uvicorn.run(**uvicorn_config)


# Create the FastAPI app instance
app = create_app()

if __name__ == "__main__":
    main()