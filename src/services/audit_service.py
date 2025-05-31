"""
Audit Service

Manages compliance and audit logging for all system operations.
Handles audit trail generation, compliance monitoring, and audit reporting.
"""

from datetime import datetime, timedelta, UTC
from typing import Dict, Any, Optional, List
from uuid import uuid4
import json

from src.services.base_service import BaseService
from src.services.exceptions import ServiceError, ValidationError, NotFoundError
from src.models.types import TenantId, UserId


class AuditService(BaseService):
    """Service for audit logging and compliance monitoring"""

    def __init__(self, audit_repository=None):
        super().__init__()
        self.audit_repository = audit_repository
        # In production, this would be injected
        self._audit_buffer = []
        self._buffer_size_limit = 100

    async def log_event(
            self,
            tenant_id: TenantId,
            event_type: str,
            event_category: str,
            action: str,
            resource_type: Optional[str] = None,
            resource_id: Optional[str] = None,
            user_id: Optional[UserId] = None,
            description: Optional[str] = None,
            metadata: Optional[Dict[str, Any]] = None,
            ip_address: Optional[str] = None,
            user_agent: Optional[str] = None,
            request_id: Optional[str] = None,
            session_id: Optional[str] = None,
            result: str = "success",
            error_details: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Log an audit event

        Args:
            tenant_id: Tenant identifier
            event_type: Type of event (e.g., message.sent, user.login)
            event_category: Category (auth, config, data, admin, api)
            action: Action performed (create, read, update, delete, etc.)
            resource_type: Type of resource affected
            resource_id: ID of specific resource
            user_id: User who performed the action
            description: Human-readable description
            metadata: Additional event metadata
            ip_address: Client IP address
            user_agent: Client user agent
            request_id: Request identifier
            session_id: Session identifier
            result: Operation result (success, failure, error)
            error_details: Error information if applicable

        Returns:
            Audit log entry ID
        """
        try:
            log_id = str(uuid4())

            audit_entry = {
                "log_id": log_id,
                "tenant_id": tenant_id,
                "user_id": user_id,
                "event_type": event_type,
                "event_category": event_category,
                "event_subcategory": self._determine_subcategory(event_type),
                "resource_type": resource_type,
                "resource_id": resource_id,
                "action": action,
                "description": description or f"{action} {resource_type or 'resource'}",
                "result": result,
                "ip_address": ip_address,
                "user_agent": user_agent,
                "request_id": request_id,
                "session_id": session_id,
                "metadata": self._sanitize_metadata(metadata or {}),
                "created_at": datetime.now(UTC),
                "error_details": error_details
            }

            # Add to buffer for batch processing
            self._audit_buffer.append(audit_entry)

            # Flush buffer if it's getting full
            if len(self._audit_buffer) >= self._buffer_size_limit:
                await self._flush_audit_buffer()

            # For critical events, flush immediately
            if event_category in ["auth", "admin"] or result in ["failure", "error"]:
                await self._flush_audit_buffer()

            self.log_operation(
                "audit_event_logged",
                tenant_id=tenant_id,
                event_type=event_type,
                event_category=event_category,
                result=result
            )

            return log_id

        except Exception as e:
            # Audit logging should never fail the main operation
            self.logger.error(
                "Failed to log audit event",
                tenant_id=tenant_id,
                event_type=event_type,
                error=str(e)
            )
            return "audit_log_failed"

    async def log_user_action(
            self,
            tenant_id: TenantId,
            user_id: UserId,
            action: str,
            resource_type: str,
            resource_id: Optional[str] = None,
            description: Optional[str] = None,
            metadata: Optional[Dict[str, Any]] = None,
            request_context: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Log a user action for audit trail

        Args:
            tenant_id: Tenant identifier
            user_id: User who performed the action
            action: Action performed
            resource_type: Type of resource
            resource_id: Specific resource ID
            description: Action description
            metadata: Additional metadata
            request_context: HTTP request context

        Returns:
            Audit log entry ID
        """
        return await self.log_event(
            tenant_id=tenant_id,
            event_type=f"user.{action}",
            event_category="user_action",
            action=action,
            resource_type=resource_type,
            resource_id=resource_id,
            user_id=user_id,
            description=description,
            metadata=metadata,
            ip_address=request_context.get("ip_address") if request_context else None,
            user_agent=request_context.get("user_agent") if request_context else None,
            request_id=request_context.get("request_id") if request_context else None,
            session_id=request_context.get("session_id") if request_context else None
        )

    async def log_system_event(
            self,
            tenant_id: TenantId,
            event_type: str,
            action: str,
            description: str,
            metadata: Optional[Dict[str, Any]] = None,
            result: str = "success",
            error_details: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Log a system event

        Args:
            tenant_id: Tenant identifier
            event_type: Type of system event
            action: Action performed
            description: Event description
            metadata: Additional metadata
            result: Operation result
            error_details: Error information if applicable

        Returns:
            Audit log entry ID
        """
        return await self.log_event(
            tenant_id=tenant_id,
            event_type=event_type,
            event_category="system",
            action=action,
            description=description,
            metadata=metadata,
            result=result,
            error_details=error_details
        )

    async def log_api_access(
            self,
            tenant_id: TenantId,
            endpoint: str,
            method: str,
            user_id: Optional[UserId] = None,
            api_key_id: Optional[str] = None,
            response_status: int = 200,
            request_context: Optional[Dict[str, Any]] = None,
            processing_time_ms: Optional[int] = None
    ) -> str:
        """
        Log API access for audit and monitoring

        Args:
            tenant_id: Tenant identifier
            endpoint: API endpoint accessed
            method: HTTP method
            user_id: User making the request
            api_key_id: API key used (if applicable)
            response_status: HTTP response status
            request_context: Request context information
            processing_time_ms: Request processing time

        Returns:
            Audit log entry ID
        """
        result = "success" if 200 <= response_status < 400 else "failure"

        metadata = {
            "endpoint": endpoint,
            "method": method,
            "response_status": response_status,
            "api_key_id": api_key_id,
            "processing_time_ms": processing_time_ms
        }

        if request_context:
            metadata.update(request_context)

        return await self.log_event(
            tenant_id=tenant_id,
            event_type="api.access",
            event_category="api",
            action=f"{method} {endpoint}",
            resource_type="api_endpoint",
            resource_id=endpoint,
            user_id=user_id,
            description=f"API access: {method} {endpoint}",
            metadata=metadata,
            ip_address=request_context.get("ip_address") if request_context else None,
            user_agent=request_context.get("user_agent") if request_context else None,
            request_id=request_context.get("request_id") if request_context else None,
            result=result
        )

    async def log_data_access(
            self,
            tenant_id: TenantId,
            user_id: UserId,
            data_type: str,
            data_id: str,
            access_type: str,
            purpose: Optional[str] = None,
            metadata: Optional[Dict[str, Any]] = None,
            request_context: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Log data access for compliance (GDPR, etc.)

        Args:
            tenant_id: Tenant identifier
            user_id: User accessing the data
            data_type: Type of data accessed
            data_id: Specific data identifier
            access_type: Type of access (read, export, delete, etc.)
            purpose: Purpose of data access
            metadata: Additional metadata
            request_context: Request context

        Returns:
            Audit log entry ID
        """
        description = f"Data access: {access_type} {data_type}"
        if purpose:
            description += f" for {purpose}"

        audit_metadata = {
            "data_type": data_type,
            "access_type": access_type,
            "purpose": purpose,
            "compliance_relevant": True
        }

        if metadata:
            audit_metadata.update(metadata)

        return await self.log_event(
            tenant_id=tenant_id,
            event_type=f"data.{access_type}",
            event_category="data_access",
            action=access_type,
            resource_type=data_type,
            resource_id=data_id,
            user_id=user_id,
            description=description,
            metadata=audit_metadata,
            ip_address=request_context.get("ip_address") if request_context else None,
            user_agent=request_context.get("user_agent") if request_context else None,
            request_id=request_context.get("request_id") if request_context else None
        )

    async def get_audit_logs(
            self,
            tenant_id: TenantId,
            filters: Optional[Dict[str, Any]] = None,
            start_date: Optional[datetime] = None,
            end_date: Optional[datetime] = None,
            limit: int = 100,
            offset: int = 0,
            user_context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Retrieve audit logs with filtering

        Args:
            tenant_id: Tenant identifier
            filters: Additional filters to apply
            start_date: Start date for log retrieval
            end_date: End date for log retrieval
            limit: Maximum number of logs to return
            offset: Number of logs to skip
            user_context: User authentication context

        Returns:
            Audit logs and metadata
        """
        try:
            await self.validate_tenant_access(tenant_id, user_context)

            # Set default date range if not provided
            if not end_date:
                end_date = datetime.now(UTC)
            if not start_date:
                start_date = end_date - timedelta(days=30)

            # Build query filters
            query_filters = filters or {}
            query_filters["tenant_id"] = tenant_id
            query_filters["created_at"] = {
                "$gte": start_date,
                "$lte": end_date
            }

            # In production, this would query the audit repository
            # For now, return simulated data
            logs = await self._query_audit_logs(query_filters, limit, offset)
            total_count = await self._count_audit_logs(query_filters)

            self.log_operation(
                "get_audit_logs",
                tenant_id=tenant_id,
                filters=len(query_filters),
                result_count=len(logs),
                start_date=start_date.isoformat(),
                end_date=end_date.isoformat()
            )

            return {
                "logs": logs,
                "total_count": total_count,
                "limit": limit,
                "offset": offset,
                "has_more": (offset + len(logs)) < total_count,
                "filters_applied": query_filters
            }

        except Exception as e:
            error = self.handle_service_error(
                e, "get_audit_logs",
                tenant_id=tenant_id
            )
            raise error

    async def generate_compliance_report(
            self,
            tenant_id: TenantId,
            report_type: str,
            start_date: datetime,
            end_date: datetime,
            filters: Optional[Dict[str, Any]] = None,
            user_context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Generate compliance report

        Args:
            tenant_id: Tenant identifier
            report_type: Type of compliance report
            start_date: Report start date
            end_date: Report end date
            filters: Additional filters
            user_context: User authentication context

        Returns:
            Compliance report data
        """
        try:
            await self.validate_tenant_access(tenant_id, user_context)

            report_id = str(uuid4())

            # Generate report based on type
            if report_type == "gdpr_data_access":
                report_data = await self._generate_gdpr_report(
                    tenant_id, start_date, end_date, filters
                )
            elif report_type == "api_usage":
                report_data = await self._generate_api_usage_report(
                    tenant_id, start_date, end_date, filters
                )
            elif report_type == "user_activity":
                report_data = await self._generate_user_activity_report(
                    tenant_id, start_date, end_date, filters
                )
            else:
                raise ValidationError(f"Unknown report type: {report_type}")

            report = {
                "report_id": report_id,
                "report_type": report_type,
                "tenant_id": tenant_id,
                "generated_at": datetime.now(UTC).isoformat(),
                "period": {
                    "start_date": start_date.isoformat(),
                    "end_date": end_date.isoformat()
                },
                "generated_by": user_context.get("user_id") if user_context else "system",
                "data": report_data
            }

            # Log report generation
            await self.log_user_action(
                tenant_id=tenant_id,
                user_id=user_context.get("user_id") if user_context else "system",
                action="generate_report",
                resource_type="compliance_report",
                resource_id=report_id,
                description=f"Generated {report_type} compliance report",
                metadata={"report_type": report_type, "period_days": (end_date - start_date).days}
            )

            return report

        except Exception as e:
            error = self.handle_service_error(
                e, "generate_compliance_report",
                tenant_id=tenant_id
            )
            raise error

    async def _flush_audit_buffer(self) -> None:
        """Flush audit buffer to persistent storage"""
        if not self._audit_buffer:
            return

        try:
            # In production, this would write to the audit repository
            buffer_to_flush = self._audit_buffer.copy()
            self._audit_buffer.clear()

            # Simulate writing to audit storage
            for entry in buffer_to_flush:
                # Would persist to database here
                pass

            self.logger.debug(
                "Audit buffer flushed",
                entries_count=len(buffer_to_flush)
            )

        except Exception as e:
            self.logger.error(
                "Failed to flush audit buffer",
                entries_count=len(self._audit_buffer),
                error=str(e)
            )

    def _sanitize_metadata(self, metadata: Dict[str, Any]) -> Dict[str, Any]:
        """Remove sensitive information from metadata"""
        sanitized = {}

        for key, value in metadata.items():
            # Remove sensitive fields
            if any(sensitive in key.lower() for sensitive in
                   ["password", "token", "secret", "key", "credential"]):
                sanitized[key] = "[REDACTED]"
            elif isinstance(value, dict):
                sanitized[key] = self._sanitize_metadata(value)
            elif isinstance(value, str) and len(value) > 1000:
                # Truncate very long strings
                sanitized[key] = value[:1000] + "... [TRUNCATED]"
            else:
                sanitized[key] = value

        return sanitized

    def _determine_subcategory(self, event_type: str) -> Optional[str]:
        """Determine event subcategory from event type"""
        if "." in event_type:
            return event_type.split(".")[0]
        return None

    async def _query_audit_logs(
            self,
            filters: Dict[str, Any],
            limit: int,
            offset: int
    ) -> List[Dict[str, Any]]:
        """Query audit logs from storage"""
        # In production, this would query the actual audit repository
        # For now, return empty list
        return []

    async def _count_audit_logs(self, filters: Dict[str, Any]) -> int:
        """Count audit logs matching filters"""
        # In production, this would count from the actual audit repository
        return 0

    async def _generate_gdpr_report(
            self,
            tenant_id: TenantId,
            start_date: datetime,
            end_date: datetime,
            filters: Optional[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Generate GDPR compliance report"""
        # Simulate GDPR report generation
        return {
            "data_access_events": 0,
            "data_export_events": 0,
            "data_deletion_events": 0,
            "consent_events": 0,
            "pii_detection_events": 0,
            "summary": "No GDPR-related events found in the specified period."
        }

    async def _generate_api_usage_report(
            self,
            tenant_id: TenantId,
            start_date: datetime,
            end_date: datetime,
            filters: Optional[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Generate API usage report"""
        # Simulate API usage report generation
        return {
            "total_requests": 0,
            "successful_requests": 0,
            "failed_requests": 0,
            "unique_users": 0,
            "top_endpoints": [],
            "error_rate": 0.0,
            "average_response_time_ms": 0
        }

    async def _generate_user_activity_report(
            self,
            tenant_id: TenantId,
            start_date: datetime,
            end_date: datetime,
            filters: Optional[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Generate user activity report"""
        # Simulate user activity report generation
        return {
            "total_users": 0,
            "active_users": 0,
            "login_events": 0,
            "configuration_changes": 0,
            "data_access_events": 0,
            "top_active_users": []
        }

    async def search_audit_logs(
            self,
            tenant_id: TenantId,
            search_query: str,
            filters: Optional[Dict[str, Any]] = None,
            limit: int = 50,
            offset: int = 0,
            user_context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Search audit logs by content

        Args:
            tenant_id: Tenant identifier
            search_query: Search query string
            filters: Additional filters
            limit: Maximum results
            offset: Results to skip
            user_context: User authentication context

        Returns:
            Search results
        """
        try:
            await self.validate_tenant_access(tenant_id, user_context)

            # Build search filters
            search_filters = filters or {}
            search_filters["tenant_id"] = tenant_id

            # In production, this would perform full-text search
            logs = await self._search_audit_logs(search_query, search_filters, limit, offset)
            total_count = await self._count_search_results(search_query, search_filters)

            self.log_operation(
                "search_audit_logs",
                tenant_id=tenant_id,
                search_query=search_query,
                result_count=len(logs)
            )

            return {
                "logs": logs,
                "total_count": total_count,
                "search_query": search_query,
                "limit": limit,
                "offset": offset
            }

        except Exception as e:
            error = self.handle_service_error(
                e, "search_audit_logs",
                tenant_id=tenant_id
            )
            raise error

    async def _search_audit_logs(
            self,
            query: str,
            filters: Dict[str, Any],
            limit: int,
            offset: int
    ) -> List[Dict[str, Any]]:
        """Perform full-text search on audit logs"""
        # In production, this would use a proper search engine
        return []

    async def _count_search_results(
            self,
            query: str,
            filters: Dict[str, Any]
    ) -> int:
        """Count search results"""
        # In production, this would count actual search results
        return 0