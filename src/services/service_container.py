"""
Service Container

Dependency injection container for managing service instances and their dependencies.
Provides centralized service instantiation and lifecycle management.
"""

from typing import Dict, Any, Optional, Type, TypeVar, Generic
from functools import lru_cache
import asyncio

from src.services.base_service import BaseService
from src.services.message_service import MessageService
from src.services.session_service import SessionService
from src.services.conversation_service import ConversationService
from src.services.channel_service import ChannelService
from src.services.delivery_service import DeliveryService
from src.services.audit_service import AuditService
from src.services.exceptions import ServiceError, ConfigurationError

T = TypeVar('T', bound=BaseService)


class ServiceContainer:
    """
    Dependency injection container for services

    Manages service instances, their dependencies, and lifecycle.
    Supports singleton and transient service lifetimes.
    """

    def __init__(self):
        self._services: Dict[str, Any] = {}
        self._service_configs: Dict[str, Dict[str, Any]] = {}
        self._service_types: Dict[str, Type[BaseService]] = {}
        self._initialized = False
        self._lock = asyncio.Lock()

    async def initialize(self, config: Dict[str, Any]) -> None:
        """
        Initialize the service container with configuration

        Args:
            config: Service configuration dictionary
        """
        async with self._lock:
            if self._initialized:
                return

            self._service_configs = config.get("services", {})

            # Register core service types
            self._register_core_services()

            self._initialized = True

    def _register_core_services(self) -> None:
        """Register core service types with the container"""
        self._service_types.update({
            "message_service": MessageService,
            "session_service": SessionService,
            "conversation_service": ConversationService,
            "channel_service": ChannelService,
            "delivery_service": DeliveryService,
            "audit_service": AuditService
        })

    async def get_service(self, service_name: str) -> BaseService:
        """
        Get a service instance by name

        Args:
            service_name: Name of the service to retrieve

        Returns:
            Service instance

        Raises:
            ServiceError: If service cannot be created or found
        """
        if not self._initialized:
            raise ServiceError("Service container not initialized")

        # Check if service is already instantiated (singleton)
        if service_name in self._services:
            return self._services[service_name]

        # Create new service instance
        service_instance = await self._create_service(service_name)

        # Store as singleton by default
        self._services[service_name] = service_instance

        return service_instance

    async def _create_service(self, service_name: str) -> BaseService:
        """
        Create a new service instance with dependency injection

        Args:
            service_name: Name of the service to create

        Returns:
            Configured service instance
        """
        if service_name not in self._service_types:
            raise ServiceError(f"Unknown service type: {service_name}")

        service_class = self._service_types[service_name]
        service_config = self._service_configs.get(service_name, {})

        # Resolve dependencies
        dependencies = await self._resolve_dependencies(service_name, service_config)

        try:
            # Create service instance with dependencies
            if dependencies:
                service_instance = service_class(**dependencies)
            else:
                service_instance = service_class()

            # Apply configuration if service supports it
            if hasattr(service_instance, 'configure'):
                await service_instance.configure(service_config)

            return service_instance

        except Exception as e:
            raise ServiceError(f"Failed to create service {service_name}: {e}")

    async def _resolve_dependencies(
            self,
            service_name: str,
            service_config: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Resolve dependencies for a service

        Args:
            service_name: Name of the service
            service_config: Service configuration

        Returns:
            Dictionary of resolved dependencies
        """
        dependencies = {}

        # Define dependency mappings for each service
        dependency_mappings = {
            "message_service": {
                "conversation_repo": "conversation_repository",
                "message_repo": "message_repository",
                "session_repo": "session_repository",
                "channel_factory": "channel_factory",
                "processor_factory": "processor_factory"
            },
            "session_service": {
                "session_repo": "session_repository"
            },
            "conversation_service": {
                "conversation_repo": "conversation_repository",
                "message_repo": "message_repository"
            },
            "channel_service": {
                "channel_factory": "channel_factory"
            },
            "delivery_service": {
                "message_repo": "message_repository",
                "channel_factory": "channel_factory"
            },
            "audit_service": {
                "audit_repository": "audit_repository"
            }
        }

        service_dependencies = dependency_mappings.get(service_name, {})

        for param_name, dependency_name in service_dependencies.items():
            try:
                # Check if dependency is another service
                if dependency_name.endswith("_service"):
                    dependency = await self.get_service(dependency_name)
                # Check if dependency is a repository
                elif dependency_name.endswith("_repository"):
                    dependency = await self._get_repository(dependency_name)
                # Check if dependency is a factory
                elif dependency_name.endswith("_factory"):
                    dependency = await self._get_factory(dependency_name)
                else:
                    # Try to get from configuration or external sources
                    dependency = await self._get_external_dependency(dependency_name)

                if dependency is not None:
                    dependencies[param_name] = dependency

            except Exception as e:
                # Log warning but continue - service might handle missing dependencies
                print(f"Warning: Could not resolve dependency {dependency_name} for {service_name}: {e}")

        return dependencies

    async def _get_repository(self, repository_name: str) -> Any:
        """
        Get repository instance

        Args:
            repository_name: Name of the repository

        Returns:
            Repository instance
        """
        # In a real implementation, this would get repositories from
        # a repository factory or dependency injection container

        # For now, return None - services should handle missing repositories gracefully
        return None

    async def _get_factory(self, factory_name: str) -> Any:
        """
        Get factory instance

        Args:
            factory_name: Name of the factory

        Returns:
            Factory instance
        """
        # In a real implementation, this would get factories from
        # a factory registry or dependency injection container

        if factory_name == "channel_factory":
            from src.core.channels.channel_factory import ChannelFactory
            return ChannelFactory()
        elif factory_name == "processor_factory":
            from src.core.processors.processor_factory import ProcessorFactory
            return ProcessorFactory()

        return None

    async def _get_external_dependency(self, dependency_name: str) -> Any:
        """
        Get external dependency (database connections, etc.)

        Args:
            dependency_name: Name of the dependency

        Returns:
            Dependency instance or None
        """
        # This would typically resolve external dependencies like
        # database connections, external service clients, etc.
        return None

    async def register_service(
            self,
            service_name: str,
            service_class: Type[BaseService],
            config: Optional[Dict[str, Any]] = None,
            singleton: bool = True
    ) -> None:
        """
        Register a new service type with the container

        Args:
            service_name: Name to register the service under
            service_class: Service class to register
            config: Optional service configuration
            singleton: Whether to create as singleton
        """
        async with self._lock:
            self._service_types[service_name] = service_class

            if config:
                self._service_configs[service_name] = config

            # If not singleton, remove any existing instance
            if not singleton and service_name in self._services:
                del self._services[service_name]

    async def register_instance(
            self,
            service_name: str,
            service_instance: BaseService
    ) -> None:
        """
        Register a pre-created service instance

        Args:
            service_name: Name to register the instance under
            service_instance: Pre-created service instance
        """
        async with self._lock:
            self._services[service_name] = service_instance
            self._service_types[service_name] = type(service_instance)

    async def remove_service(self, service_name: str) -> None:
        """
        Remove a service from the container

        Args:
            service_name: Name of the service to remove
        """
        async with self._lock:
            if service_name in self._services:
                service_instance = self._services[service_name]

                # Call cleanup if service supports it
                if hasattr(service_instance, 'cleanup'):
                    try:
                        await service_instance.cleanup()
                    except Exception as e:
                        print(f"Error during service cleanup for {service_name}: {e}")

                del self._services[service_name]

            if service_name in self._service_types:
                del self._service_types[service_name]

            if service_name in self._service_configs:
                del self._service_configs[service_name]

    async def get_service_health(self) -> Dict[str, Any]:
        """
        Get health status of all registered services

        Returns:
            Health status dictionary
        """
        health_status = {
            "container_initialized": self._initialized,
            "registered_services": list(self._service_types.keys()),
            "instantiated_services": list(self._services.keys()),
            "service_health": {}
        }

        # Check health of instantiated services
        for service_name, service_instance in self._services.items():
            try:
                if hasattr(service_instance, 'health_check'):
                    service_health = await service_instance.health_check()
                else:
                    service_health = {"status": "unknown", "message": "Health check not implemented"}

                health_status["service_health"][service_name] = service_health

            except Exception as e:
                health_status["service_health"][service_name] = {
                    "status": "error",
                    "error": str(e)
                }

        return health_status

    async def shutdown(self) -> None:
        """
        Shutdown the service container and cleanup all services
        """
        async with self._lock:
            # Cleanup all service instances
            for service_name, service_instance in self._services.items():
                try:
                    if hasattr(service_instance, 'cleanup'):
                        await service_instance.cleanup()
                except Exception as e:
                    print(f"Error during cleanup of {service_name}: {e}")

            # Clear all registrations
            self._services.clear()
            self._service_types.clear()
            self._service_configs.clear()
            self._initialized = False

    def list_services(self) -> Dict[str, Any]:
        """
        List all registered services and their status

        Returns:
            Dictionary with service information
        """
        return {
            "registered_types": list(self._service_types.keys()),
            "instantiated_services": list(self._services.keys()),
            "service_configs": {
                name: bool(config) for name, config in self._service_configs.items()
            }
        }

    async def reconfigure_service(
            self,
            service_name: str,
            new_config: Dict[str, Any]
    ) -> None:
        """
        Reconfigure an existing service

        Args:
            service_name: Name of the service to reconfigure
            new_config: New configuration to apply
        """
        async with self._lock:
            # Update stored configuration
            self._service_configs[service_name] = new_config

            # If service is instantiated, reconfigure it
            if service_name in self._services:
                service_instance = self._services[service_name]

                if hasattr(service_instance, 'configure'):
                    await service_instance.configure(new_config)
                else:
                    # Service doesn't support reconfiguration, recreate it
                    await self.remove_service(service_name)
                    # Next get_service call will create with new config


# Global service container instance
_service_container: Optional[ServiceContainer] = None


async def get_service_container() -> ServiceContainer:
    """
    Get the global service container instance

    Returns:
        Global service container
    """
    global _service_container

    if _service_container is None:
        _service_container = ServiceContainer()

    return _service_container


async def initialize_service_container(config: Dict[str, Any]) -> ServiceContainer:
    """
    Initialize the global service container

    Args:
        config: Service configuration

    Returns:
        Initialized service container
    """
    container = await get_service_container()
    await container.initialize(config)
    return container


@lru_cache()
def get_service_container_sync() -> ServiceContainer:
    """
    Get service container synchronously (for FastAPI dependencies)

    Returns:
        Service container instance
    """
    global _service_container

    if _service_container is None:
        _service_container = ServiceContainer()

    return _service_container