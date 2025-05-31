# src/database/health_checks.py
"""
Database health check implementations.
Provides comprehensive health monitoring for all database systems.
"""

from typing import Dict, Any, List, Optional
import structlog
import asyncio
from datetime import datetime, timedelta
import time

from src.database.mongodb import mongodb_manager
from src.database.redis_client import redis_manager

logger = structlog.get_logger()


class HealthCheckResult:
    """Result of a health check operation."""

    def __init__(
            self,
            service_name: str,
            healthy: bool,
            response_time_ms: Optional[float] = None,
            error: Optional[str] = None,
            details: Optional[Dict[str, Any]] = None
    ):
        self.service_name = service_name
        self.healthy = healthy
        self.response_time_ms = response_time_ms
        self.error = error
        self.details = details or {}
        self.timestamp = datetime.utcnow()

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for API responses."""
        return {
            "service": self.service_name,
            "healthy": self.healthy,
            "response_time_ms": self.response_time_ms,
            "error": self.error,
            "details": self.details,
            "timestamp": self.timestamp.isoformat()
        }


class DatabaseHealthChecker:
    """Comprehensive database health monitoring."""

    def __init__(self):
        self.last_check_results: Dict[str, HealthCheckResult] = {}
        self.check_history: List[Dict[str, Any]] = []
        self.max_history_size = 100

    async def check_all_databases(
            self,
            include_details: bool = False,
            timeout_seconds: float = 10.0
    ) -> Dict[str, Any]:
        """
        Check health of all database connections.

        Args:
            include_details: Whether to include detailed information
            timeout_seconds: Timeout for health checks

        Returns:
            Dictionary with overall health status and individual results
        """
        start_time = time.time()

        # Run all health checks concurrently with timeout
        check_tasks = [
            self._check_mongodb_with_timeout(timeout_seconds),
            self._check_redis_with_timeout(timeout_seconds)
        ]

        results = await asyncio.gather(*check_tasks, return_exceptions=True)

        # Process results
        mongodb_result, redis_result = results

        # Store results
        if isinstance(mongodb_result, HealthCheckResult):
            self.last_check_results["mongodb"] = mongodb_result
        if isinstance(redis_result, HealthCheckResult):
            self.last_check_results["redis"] = redis_result

        # Determine overall status
        all_healthy = all(
            isinstance(result, HealthCheckResult) and result.healthy
            for result in results
        )

        total_time = (time.time() - start_time) * 1000

        health_status = {
            "status": "healthy" if all_healthy else "unhealthy",
            "timestamp": datetime.utcnow().isoformat(),
            "response_time_ms": round(total_time, 2),
            "services": {}
        }

        # Add individual service results
        for result in results:
            if isinstance(result, HealthCheckResult):
                service_data = result.to_dict()
                if not include_details:
                    # Remove detailed information for basic checks
                    service_data.pop("details", None)
                health_status["services"][result.service_name] = service_data
            elif isinstance(result, Exception):
                # Handle exceptions
                health_status["services"]["unknown"] = {
                    "healthy": False,
                    "error": str(result),
                    "timestamp": datetime.utcnow().isoformat()
                }

        # Store in history
        self._add_to_history(health_status)

        return health_status

    async def check_mongodb(self, include_details: bool = False) -> HealthCheckResult:
        """
        Check MongoDB health with detailed diagnostics.

        Args:
            include_details: Whether to include server information

        Returns:
            HealthCheckResult for MongoDB
        """
        start_time = time.time()

        try:
            # Basic connectivity check
            is_healthy = await mongodb_manager.health_check()
            response_time = (time.time() - start_time) * 1000

            if not is_healthy:
                return HealthCheckResult(
                    service_name="mongodb",
                    healthy=False,
                    response_time_ms=round(response_time, 2),
                    error="MongoDB ping failed"
                )

            details = {}
            if include_details:
                # Get server information
                server_info = await mongodb_manager.get_server_info()
                details.update(server_info)

                # Get collection statistics
                collection_stats = await mongodb_manager.get_collection_stats()
                details["collections"] = collection_stats

                # Check for any connection issues
                database = mongodb_manager.get_database()
                try:
                    # Test a simple operation
                    await database.command("serverStatus", maxTimeMS=1000)
                    details["operations_functional"] = True
                except Exception as e:
                    details["operations_functional"] = False
                    details["operations_error"] = str(e)

            return HealthCheckResult(
                service_name="mongodb",
                healthy=True,
                response_time_ms=round(response_time, 2),
                details=details
            )

        except Exception as e:
            response_time = (time.time() - start_time) * 1000
            logger.error("MongoDB health check exception", error=str(e))
            return HealthCheckResult(
                service_name="mongodb",
                healthy=False,
                response_time_ms=round(response_time, 2),
                error=str(e)
            )

    async def check_redis(self, include_details: bool = False) -> HealthCheckResult:
        """
        Check Redis health with detailed diagnostics.

        Args:
            include_details: Whether to include server information

        Returns:
            HealthCheckResult for Redis
        """
        start_time = time.time()

        try:
            # Basic connectivity check
            is_healthy = await redis_manager.health_check()
            response_time = (time.time() - start_time) * 1000

            if not is_healthy:
                return HealthCheckResult(
                    service_name="redis",
                    healthy=False,
                    response_time_ms=round(response_time, 2),
                    error="Redis ping failed"
                )

            details = {}
            if include_details:
                # Get server information
                server_info = await redis_manager.get_server_info()
                details.update(server_info)

                # Test basic operations
                try:
                    client = redis_manager.get_client()

                    # Test set/get operation
                    test_key = "health_check_test"
                    test_value = str(int(time.time()))

                    await client.set(test_key, test_value, ex=60)
                    retrieved_value = await client.get(test_key)
                    await client.delete(test_key)

                    details["operations_functional"] = retrieved_value == test_value

                    # Get memory usage
                    memory_info = await client.memory_usage("non_existent_key_for_test") or 0
                    details["test_memory_usage"] = memory_info

                except Exception as e:
                    details["operations_functional"] = False
                    details["operations_error"] = str(e)

            return HealthCheckResult(
                service_name="redis",
                healthy=True,
                response_time_ms=round(response_time, 2),
                details=details
            )

        except Exception as e:
            response_time = (time.time() - start_time) * 1000
            logger.error("Redis health check exception", error=str(e))
            return HealthCheckResult(
                service_name="redis",
                healthy=False,
                response_time_ms=round(response_time, 2),
                error=str(e)
            )

    async def _check_mongodb_with_timeout(self, timeout_seconds: float) -> HealthCheckResult:
        """Check MongoDB with timeout."""
        try:
            return await asyncio.wait_for(
                self.check_mongodb(include_details=True),
                timeout=timeout_seconds
            )
        except asyncio.TimeoutError:
            return HealthCheckResult(
                service_name="mongodb",
                healthy=False,
                error=f"Health check timeout ({timeout_seconds}s)"
            )
        except Exception as e:
            return HealthCheckResult(
                service_name="mongodb",
                healthy=False,
                error=str(e)
            )

    async def _check_redis_with_timeout(self, timeout_seconds: float) -> HealthCheckResult:
        """Check Redis with timeout."""
        try:
            return await asyncio.wait_for(
                self.check_redis(include_details=True),
                timeout=timeout_seconds
            )
        except asyncio.TimeoutError:
            return HealthCheckResult(
                service_name="redis",
                healthy=False,
                error=f"Health check timeout ({timeout_seconds}s)"
            )
        except Exception as e:
            return HealthCheckResult(
                service_name="redis",
                healthy=False,
                error=str(e)
            )

    def _add_to_history(self, health_status: Dict[str, Any]) -> None:
        """Add health check result to history."""
        self.check_history.append(health_status)

        # Keep history size limited
        if len(self.check_history) > self.max_history_size:
            self.check_history = self.check_history[-self.max_history_size:]

    def get_health_history(
            self,
            limit: int = 10,
            service_filter: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Get health check history.

        Args:
            limit: Maximum number of results to return
            service_filter: Filter by specific service name

        Returns:
            List of historical health check results
        """
        history = self.check_history[-limit:] if limit > 0 else self.check_history

        if service_filter:
            filtered_history = []
            for entry in history:
                if service_filter in entry.get("services", {}):
                    # Create filtered entry with only the requested service
                    filtered_entry = {
                        "timestamp": entry["timestamp"],
                        "service": entry["services"][service_filter]
                    }
                    filtered_history.append(filtered_entry)
            return filtered_history

        return history

    def get_uptime_statistics(
            self,
            hours: int = 24,
            service_filter: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Calculate uptime statistics for the specified period.

        Args:
            hours: Number of hours to analyze
            service_filter: Filter by specific service name

        Returns:
            Dictionary with uptime statistics
        """
        cutoff_time = datetime.utcnow() - timedelta(hours=hours)

        # Filter history by time
        recent_history = [
            entry for entry in self.check_history
            if datetime.fromisoformat(entry["timestamp"]) >= cutoff_time
        ]

        if not recent_history:
            return {"error": "No health check data available for the specified period"}

        stats = {
            "period_hours": hours,
            "total_checks": len(recent_history),
            "services": {}
        }

        # Analyze each service
        services_to_check = [service_filter] if service_filter else ["mongodb", "redis"]

        for service in services_to_check:
            service_checks = []

            for entry in recent_history:
                if service in entry.get("services", {}):
                    service_data = entry["services"][service]
                    service_checks.append({
                        "timestamp": entry["timestamp"],
                        "healthy": service_data.get("healthy", False),
                        "response_time_ms": service_data.get("response_time_ms")
                    })

            if service_checks:
                healthy_checks = sum(1 for check in service_checks if check["healthy"])
                uptime_percentage = (healthy_checks / len(service_checks)) * 100

                # Calculate average response time for healthy checks
                healthy_response_times = [
                    check["response_time_ms"] for check in service_checks
                    if check["healthy"] and check["response_time_ms"] is not None
                ]
                avg_response_time = (
                    sum(healthy_response_times) / len(healthy_response_times)
                    if healthy_response_times else None
                )

                stats["services"][service] = {
                    "total_checks": len(service_checks),
                    "healthy_checks": healthy_checks,
                    "uptime_percentage": round(uptime_percentage, 2),
                    "average_response_time_ms": (
                        round(avg_response_time, 2) if avg_response_time else None
                    ),
                    "current_status": service_checks[-1]["healthy"] if service_checks else None
                }

        return stats


# Global health checker instance
health_checker = DatabaseHealthChecker()


# Convenience functions for easy access
async def check_all_databases(include_details: bool = False) -> Dict[str, Any]:
    """
    Check health of all database connections.

    Args:
        include_details: Whether to include detailed information

    Returns:
        Dictionary with health status and individual results
    """
    return await health_checker.check_all_databases(include_details=include_details)


async def check_mongodb() -> bool:
    """
    Simple MongoDB health check.

    Returns:
        True if healthy, False otherwise
    """
    try:
        result = await health_checker.check_mongodb()
        return result.healthy
    except Exception as e:
        logger.error("MongoDB health check exception", error=str(e))
        return False


async def check_redis() -> bool:
    """
    Simple Redis health check.

    Returns:
        True if healthy, False otherwise
    """
    try:
        result = await health_checker.check_redis()
        return result.healthy
    except Exception as e:
        logger.error("Redis health check exception", error=str(e))
        return False


async def get_detailed_health_status() -> Dict[str, Any]:
    """
    Get comprehensive health status with details.

    Returns:
        Dictionary with detailed health information
    """
    return await health_checker.check_all_databases(include_details=True)


def get_health_history(limit: int = 10) -> List[Dict[str, Any]]:
    """
    Get recent health check history.

    Args:
        limit: Maximum number of results

    Returns:
        List of historical health check results
    """
    return health_checker.get_health_history(limit=limit)


def get_uptime_stats(hours: int = 24) -> Dict[str, Any]:
    """
    Get uptime statistics for the specified period.

    Args:
        hours: Number of hours to analyze

    Returns:
        Dictionary with uptime statistics
    """
    return health_checker.get_uptime_statistics(hours=hours)


# Health check scheduler for background monitoring
class HealthCheckScheduler:
    """Background health check scheduler."""

    def __init__(self, check_interval_seconds: int = 60):
        self.check_interval = check_interval_seconds
        self.running = False
        self.task: Optional[asyncio.Task] = None

    async def start_monitoring(self) -> None:
        """Start background health monitoring."""
        if self.running:
            logger.warning("Health check monitoring already running")
            return

        self.running = True
        self.task = asyncio.create_task(self._monitoring_loop())
        logger.info("Health check monitoring started", interval=self.check_interval)

    async def stop_monitoring(self) -> None:
        """Stop background health monitoring."""
        self.running = False
        if self.task and not self.task.done():
            self.task.cancel()
            try:
                await self.task
            except asyncio.CancelledError:
                pass
        logger.info("Health check monitoring stopped")

    async def _monitoring_loop(self) -> None:
        """Background monitoring loop."""
        while self.running:
            try:
                # Perform health checks
                await health_checker.check_all_databases(include_details=False)

                # Wait for next check
                await asyncio.sleep(self.check_interval)

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error("Health check monitoring error", error=str(e))
                await asyncio.sleep(self.check_interval)


# Global health check scheduler
health_scheduler = HealthCheckScheduler()