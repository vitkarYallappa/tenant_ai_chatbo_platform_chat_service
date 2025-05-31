"""
Base Repository Pattern Implementation
=====================================

Provides abstract base classes and utilities for implementing the repository pattern
across different data stores (MongoDB, Redis, PostgreSQL).

Key Features:
- Generic type support for type safety
- Standardized pagination
- Consistent error handling
- Extensible query building
- Performance monitoring integration
"""

from abc import ABC, abstractmethod
from typing import TypeVar, Generic, Optional, List, Dict, Any, Tuple
from dataclasses import dataclass
from datetime import datetime
import structlog
import asyncio
from contextlib import asynccontextmanager

# Configure structured logging
logger = structlog.get_logger()

# Generic type variable for entities
T = TypeVar('T')


@dataclass
class Pagination:
    """
    Pagination parameters with validation and utility methods

    Attributes:
        page: Page number (1-based)
        page_size: Number of items per page
        sort_by: Field name to sort by
        sort_order: Sort direction ('asc' or 'desc')
    """
    page: int = 1
    page_size: int = 20
    sort_by: Optional[str] = None
    sort_order: str = "desc"

    @property
    def offset(self) -> int:
        """Calculate offset for database queries"""
        return (self.page - 1) * self.page_size

    @property
    def limit(self) -> int:
        """Alias for page_size for clarity"""
        return self.page_size

    def validate(self) -> None:
        """
        Validate pagination parameters

        Raises:
            ValueError: If parameters are invalid
        """
        if self.page < 1:
            raise ValueError("Page must be >= 1")
        if self.page_size < 1 or self.page_size > 100:
            raise ValueError("Page size must be between 1 and 100")
        if self.sort_order not in ["asc", "desc"]:
            raise ValueError("Sort order must be 'asc' or 'desc'")

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            "page": self.page,
            "page_size": self.page_size,
            "sort_by": self.sort_by,
            "sort_order": self.sort_order,
            "offset": self.offset
        }


@dataclass
class PaginatedResult(Generic[T]):
    """
    Container for paginated query results with metadata

    Type Parameters:
        T: Type of items in the result set
    """
    items: List[T]
    total: int
    page: int
    page_size: int
    has_next: bool
    has_previous: bool
    total_pages: int

    @classmethod
    def create(
            cls,
            items: List[T],
            total: int,
            pagination: Pagination
    ) -> "PaginatedResult[T]":
        """
        Create paginated result from items and pagination info

        Args:
            items: List of items for current page
            total: Total number of items across all pages
            pagination: Pagination parameters used

        Returns:
            PaginatedResult instance with calculated metadata
        """
        total_pages = (total + pagination.page_size - 1) // pagination.page_size
        has_next = pagination.page < total_pages
        has_previous = pagination.page > 1

        return cls(
            items=items,
            total=total,
            page=pagination.page,
            page_size=pagination.page_size,
            has_next=has_next,
            has_previous=has_previous,
            total_pages=total_pages
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for API responses"""
        return {
            "items": [
                item.to_dict() if hasattr(item, 'to_dict') else item
                for item in self.items
            ],
            "pagination": {
                "total": self.total,
                "page": self.page,
                "page_size": self.page_size,
                "total_pages": self.total_pages,
                "has_next": self.has_next,
                "has_previous": self.has_previous
            }
        }


@dataclass
class QueryFilter:
    """
    Structured query filter for repository operations
    """
    field: str
    operator: str  # eq, ne, gt, gte, lt, lte, in, nin, contains, regex
    value: Any
    case_sensitive: bool = True

    def validate(self) -> None:
        """Validate filter parameters"""
        valid_operators = ['eq', 'ne', 'gt', 'gte', 'lt', 'lte', 'in', 'nin', 'contains', 'regex']
        if self.operator not in valid_operators:
            raise ValueError(f"Invalid operator: {self.operator}")


@dataclass
class SortCriteria:
    """Sort criteria for query results"""
    field: str
    direction: int = -1  # -1 for desc, 1 for asc

    @classmethod
    def from_pagination(cls, pagination: Pagination) -> List["SortCriteria"]:
        """Create sort criteria from pagination object"""
        if not pagination.sort_by:
            return [cls("created_at", -1)]  # Default sort

        direction = 1 if pagination.sort_order == "asc" else -1
        return [cls(pagination.sort_by, direction)]


class BaseRepository(ABC, Generic[T]):
    """
    Abstract base repository class defining standard CRUD operations

    Type Parameters:
        T: Entity type this repository manages

    Features:
        - Async/await support
        - Structured logging
        - Performance monitoring
        - Error handling
        - Query building utilities
    """

    def __init__(self):
        """Initialize repository with logger"""
        self.logger = structlog.get_logger(self.__class__.__name__)
        self._performance_metrics = {}

    # Abstract CRUD Methods

    @abstractmethod
    async def create(self, entity: T) -> T:
        """
        Create a new entity

        Args:
            entity: Entity to create

        Returns:
            Created entity with generated ID

        Raises:
            RepositoryError: If creation fails
            DuplicateEntityError: If entity already exists
        """
        pass

    @abstractmethod
    async def get_by_id(self, entity_id: str) -> Optional[T]:
        """
        Get entity by ID

        Args:
            entity_id: Unique identifier

        Returns:
            Entity if found, None otherwise

        Raises:
            RepositoryError: If query fails
        """
        pass

    @abstractmethod
    async def update(self, entity: T) -> T:
        """
        Update existing entity

        Args:
            entity: Entity with updated data

        Returns:
            Updated entity

        Raises:
            RepositoryError: If update fails
            EntityNotFoundError: If entity doesn't exist
        """
        pass

    @abstractmethod
    async def delete(self, entity_id: str) -> bool:
        """
        Delete entity by ID

        Args:
            entity_id: Unique identifier

        Returns:
            True if deleted, False if not found

        Raises:
            RepositoryError: If deletion fails
        """
        pass

    @abstractmethod
    async def list(
            self,
            filters: Optional[Dict[str, Any]] = None,
            pagination: Optional[Pagination] = None
    ) -> PaginatedResult[T]:
        """
        List entities with optional filters and pagination

        Args:
            filters: Query filters
            pagination: Pagination parameters

        Returns:
            Paginated result set

        Raises:
            RepositoryError: If query fails
        """
        pass

    @abstractmethod
    async def exists(self, entity_id: str) -> bool:
        """
        Check if entity exists

        Args:
            entity_id: Unique identifier

        Returns:
            True if entity exists

        Raises:
            RepositoryError: If query fails
        """
        pass

    @abstractmethod
    async def count(self, filters: Optional[Dict[str, Any]] = None) -> int:
        """
        Count entities matching filters

        Args:
            filters: Query filters

        Returns:
            Number of matching entities

        Raises:
            RepositoryError: If query fails
        """
        pass

    # Utility Methods

    def _build_sort_criteria(self, pagination: Optional[Pagination]) -> List[Tuple[str, int]]:
        """
        Build sort criteria for database queries

        Args:
            pagination: Pagination with sort preferences

        Returns:
            List of (field, direction) tuples
        """
        if not pagination or not pagination.sort_by:
            return [("created_at", -1)]  # Default sort by creation time desc

        sort_direction = 1 if pagination.sort_order == "asc" else -1
        return [(pagination.sort_by, sort_direction)]

    def _log_operation(
            self,
            operation: str,
            duration_ms: Optional[float] = None,
            **kwargs
    ) -> None:
        """
        Log repository operation with structured data

        Args:
            operation: Operation name (create, read, update, delete)
            duration_ms: Operation duration in milliseconds
            **kwargs: Additional context data
        """
        log_data = {
            "operation": operation,
            "repository": self.__class__.__name__,
            **kwargs
        }

        if duration_ms is not None:
            log_data["duration_ms"] = round(duration_ms, 2)

        self.logger.info("Repository operation completed", **log_data)

    def _log_error(
            self,
            operation: str,
            error: Exception,
            **kwargs
    ) -> None:
        """
        Log repository error with context

        Args:
            operation: Failed operation name
            error: Exception that occurred
            **kwargs: Additional context data
        """
        self.logger.error(
            f"Repository operation failed: {operation}",
            error=str(error),
            error_type=type(error).__name__,
            repository=self.__class__.__name__,
            **kwargs
        )

    @asynccontextmanager
    async def _timed_operation(self, operation: str):
        """
        Context manager for timing and logging operations

        Args:
            operation: Operation name for logging

        Yields:
            Context for the timed operation
        """
        start_time = asyncio.get_event_loop().time()
        try:
            yield
        finally:
            duration_ms = (asyncio.get_event_loop().time() - start_time) * 1000
            self._log_operation(operation, duration_ms)

    async def _validate_pagination(self, pagination: Optional[Pagination]) -> Pagination:
        """
        Validate and return pagination object with defaults

        Args:
            pagination: Pagination parameters or None

        Returns:
            Validated pagination object

        Raises:
            ValueError: If pagination parameters are invalid
        """
        if pagination is None:
            pagination = Pagination()

        pagination.validate()
        return pagination

    def _build_query_filters(self, filters: Dict[str, Any]) -> List[QueryFilter]:
        """
        Convert simple filters dict to structured QueryFilter objects

        Args:
            filters: Simple key-value filter dictionary

        Returns:
            List of QueryFilter objects
        """
        query_filters = []

        for field, value in filters.items():
            if isinstance(value, dict) and 'operator' in value:
                # Structured filter: {"field": {"operator": "gt", "value": 100}}
                query_filter = QueryFilter(
                    field=field,
                    operator=value['operator'],
                    value=value['value'],
                    case_sensitive=value.get('case_sensitive', True)
                )
            else:
                # Simple equality filter: {"field": "value"}
                query_filter = QueryFilter(
                    field=field,
                    operator='eq',
                    value=value
                )

            query_filter.validate()
            query_filters.append(query_filter)

        return query_filters

    # Performance Monitoring

    def _record_metric(self, metric_name: str, value: float) -> None:
        """Record performance metric"""
        if metric_name not in self._performance_metrics:
            self._performance_metrics[metric_name] = []

        self._performance_metrics[metric_name].append({
            'value': value,
            'timestamp': datetime.utcnow()
        })

        # Keep only last 100 measurements
        if len(self._performance_metrics[metric_name]) > 100:
            self._performance_metrics[metric_name] = self._performance_metrics[metric_name][-100:]

    def get_performance_metrics(self) -> Dict[str, Dict[str, float]]:
        """
        Get aggregated performance metrics

        Returns:
            Dictionary with metric statistics
        """
        metrics_summary = {}

        for metric_name, measurements in self._performance_metrics.items():
            if not measurements:
                continue

            values = [m['value'] for m in measurements]
            metrics_summary[metric_name] = {
                'count': len(values),
                'avg': sum(values) / len(values),
                'min': min(values),
                'max': max(values),
                'latest': values[-1] if values else 0
            }

        return metrics_summary


# Type aliases for convenience
EntityId = str
FilterDict = Dict[str, Any]
SortOrder = Tuple[str, int]


# Repository health check interface
class RepositoryHealthCheck(ABC):
    """Interface for repository health monitoring"""

    @abstractmethod
    async def health_check(self) -> Dict[str, Any]:
        """
        Perform health check on repository

        Returns:
            Health status dictionary
        """
        pass


# Batch operations mixin
class BatchOperationsMixin:
    """Mixin for batch CRUD operations"""

    async def create_many(self, entities: List[T]) -> List[T]:
        """
        Create multiple entities in batch

        Args:
            entities: List of entities to create

        Returns:
            List of created entities
        """
        # Default implementation - override for better performance
        results = []
        for entity in entities:
            result = await self.create(entity)
            results.append(result)
        return results

    async def update_many(self, entities: List[T]) -> List[T]:
        """
        Update multiple entities in batch

        Args:
            entities: List of entities to update

        Returns:
            List of updated entities
        """
        # Default implementation - override for better performance
        results = []
        for entity in entities:
            result = await self.update(entity)
            results.append(result)
        return results

    async def delete_many(self, entity_ids: List[str]) -> int:
        """
        Delete multiple entities in batch

        Args:
            entity_ids: List of entity IDs to delete

        Returns:
            Number of entities deleted
        """
        # Default implementation - override for better performance
        deleted_count = 0
        for entity_id in entity_ids:
            if await self.delete(entity_id):
                deleted_count += 1
        return deleted_count