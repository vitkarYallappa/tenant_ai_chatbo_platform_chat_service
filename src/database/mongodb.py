# src/database/mongodb.py
"""
MongoDB connection manager and client setup.
Provides async MongoDB client with connection pooling and health monitoring.
"""

from motor.motor_asyncio import AsyncIOMotorClient, AsyncIOMotorDatabase, AsyncIOMotorCollection
from pymongo.errors import ServerSelectionTimeoutError, ConnectionFailure, OperationFailure
from pymongo import IndexModel, ASCENDING, DESCENDING, TEXT
from typing import Optional, List, Dict, Any
import structlog
import asyncio
from datetime import datetime

# Note: This would normally import from settings, but since it's not yet available:
# from src.config.settings import get_settings

logger = structlog.get_logger()


class MongoDBManager:
    """
    MongoDB connection manager with connection pooling and health monitoring.
    Handles connection lifecycle, index creation, and database operations.
    """

    def __init__(self):
        self.client: Optional[AsyncIOMotorClient] = None
        self.database: Optional[AsyncIOMotorDatabase] = None
        # self.settings = get_settings()  # Will be available after config setup

        # Default settings (will be replaced by actual config)
        self.mongodb_uri = "mongodb://localhost:27017"
        self.database_name = "chatbot_platform"
        self.max_pool_size = 100
        self.server_selection_timeout_ms = 5000
        self.connect_timeout_ms = 10000
        self.socket_timeout_ms = 20000

    async def connect(self) -> None:
        """
        Establish MongoDB connection with optimal settings for production.

        Raises:
            ConnectionError: If connection fails
        """
        try:
            self.client = AsyncIOMotorClient(
                self.mongodb_uri,
                maxPoolSize=self.max_pool_size,
                serverSelectionTimeoutMS=self.server_selection_timeout_ms,
                connectTimeoutMS=self.connect_timeout_ms,
                socketTimeoutMS=self.socket_timeout_ms,
                retryWrites=True,
                w="majority",
                readPreference="secondaryPreferred",
                readConcern="majority"
            )

            # Test connection with admin ping
            await self.client.admin.command('ping')

            self.database = self.client[self.database_name]

            logger.info(
                "MongoDB connected successfully",
                database=self.database_name,
                max_pool_size=self.max_pool_size,
                uri_host=self.mongodb_uri.split('@')[-1] if '@' in self.mongodb_uri else self.mongodb_uri
            )

            # Create indexes after connection
            await self.create_indexes()

        except ServerSelectionTimeoutError as e:
            logger.error("MongoDB server selection timeout", error=str(e))
            raise ConnectionError(f"MongoDB connection timeout: {e}")
        except ConnectionFailure as e:
            logger.error("MongoDB connection failure", error=str(e))
            raise ConnectionError(f"MongoDB connection failed: {e}")
        except Exception as e:
            logger.error("Unexpected MongoDB connection error", error=str(e))
            raise ConnectionError(f"MongoDB connection error: {e}")

    async def disconnect(self) -> None:
        """Close MongoDB connection gracefully."""
        if self.client:
            self.client.close()
            # Wait for connection to close
            await asyncio.sleep(0.1)
            logger.info("MongoDB connection closed")

    def get_database(self) -> AsyncIOMotorDatabase:
        """
        Get database instance.

        Returns:
            AsyncIOMotorDatabase instance

        Raises:
            RuntimeError: If database not connected
        """
        if not self.database:
            raise RuntimeError("MongoDB database not connected. Call connect() first.")
        return self.database

    def get_collection(self, collection_name: str) -> AsyncIOMotorCollection:
        """
        Get collection instance.

        Args:
            collection_name: Name of the collection

        Returns:
            AsyncIOMotorCollection instance
        """
        database = self.get_database()
        return database[collection_name]

    async def health_check(self) -> bool:
        """
        Check MongoDB connection health.

        Returns:
            True if healthy, False otherwise
        """
        try:
            if self.client:
                # Use a short timeout for health checks
                await self.client.admin.command('ping', maxTimeMS=1000)
                return True
            return False
        except Exception as e:
            logger.error("MongoDB health check failed", error=str(e))
            return False

    async def get_server_info(self) -> Dict[str, Any]:
        """
        Get MongoDB server information.

        Returns:
            Dictionary with server information
        """
        try:
            if not self.client:
                return {"status": "disconnected"}

            server_info = await self.client.admin.command('buildInfo')
            server_status = await self.client.admin.command('serverStatus')

            return {
                "status": "connected",
                "version": server_info.get("version"),
                "uptime": server_status.get("uptime"),
                "connections": server_status.get("connections", {}),
                "memory": server_status.get("mem", {}),
                "operations": server_status.get("opcounters", {})
            }
        except Exception as e:
            logger.error("Failed to get MongoDB server info", error=str(e))
            return {"status": "error", "error": str(e)}

    async def create_indexes(self) -> None:
        """Create required database indexes for optimal performance."""
        if not self.database:
            logger.warning("Cannot create indexes: database not connected")
            return

        try:
            # Conversations collection indexes
            conversations = self.database.conversations
            conversation_indexes = [
                IndexModel([("conversation_id", ASCENDING)], unique=True),
                IndexModel([("tenant_id", ASCENDING), ("started_at", DESCENDING)]),
                IndexModel([("tenant_id", ASCENDING), ("user_id", ASCENDING), ("started_at", DESCENDING)]),
                IndexModel([("tenant_id", ASCENDING), ("status", ASCENDING), ("last_activity_at", DESCENDING)]),
                IndexModel([("tenant_id", ASCENDING), ("channel", ASCENDING), ("started_at", DESCENDING)]),
                IndexModel([("tenant_id", ASCENDING), ("business_context.category", ASCENDING)]),
                IndexModel([("tenant_id", ASCENDING), ("business_context.outcome", ASCENDING)]),
                IndexModel([("compliance.data_retention_until", ASCENDING)]),  # For cleanup
                IndexModel([("last_activity_at", ASCENDING)]),  # For stale conversation cleanup
                IndexModel([("ai_metadata.primary_models_used", ASCENDING)]),  # For model usage analytics
            ]
            await conversations.create_indexes(conversation_indexes)

            # Messages collection indexes
            messages = self.database.messages
            message_indexes = [
                IndexModel([("message_id", ASCENDING)], unique=True),
                IndexModel([("conversation_id", ASCENDING), ("sequence_number", ASCENDING)]),
                IndexModel([("tenant_id", ASCENDING), ("timestamp", DESCENDING)]),
                IndexModel([("tenant_id", ASCENDING), ("direction", ASCENDING), ("timestamp", DESCENDING)]),
                IndexModel([("tenant_id", ASCENDING), ("ai_analysis.intent.detected_intent", ASCENDING)]),
                IndexModel([("content.type", ASCENDING), ("timestamp", DESCENDING)]),
                IndexModel([("privacy.auto_delete_at", ASCENDING)]),  # For cleanup
                IndexModel([("moderation.flagged", ASCENDING), ("timestamp", DESCENDING)]),
                IndexModel([("tenant_id", ASCENDING), ("channel", ASCENDING), ("timestamp", DESCENDING)]),
                # Text search index for message content
                IndexModel([("content.text", TEXT), ("ai_analysis.topics", TEXT)]),
            ]
            await messages.create_indexes(message_indexes)

            # Sessions collection indexes
            sessions = self.database.sessions
            session_indexes = [
                IndexModel([("session_id", ASCENDING)], unique=True),
                IndexModel([("tenant_id", ASCENDING), ("started_at", DESCENDING)]),
                IndexModel([("tenant_id", ASCENDING), ("user_id", ASCENDING), ("started_at", DESCENDING)]),
                IndexModel([("tenant_id", ASCENDING), ("status", ASCENDING)]),
                IndexModel([("expires_at", ASCENDING)]),  # For cleanup
                IndexModel([("data_retention_until", ASCENDING)]),  # For compliance cleanup
                IndexModel([("last_activity_at", ASCENDING)]),  # For idle session detection
                IndexModel([("security.suspicious_activity", ASCENDING), ("last_activity_at", DESCENDING)]),
            ]
            await sessions.create_indexes(session_indexes)

            logger.info("MongoDB indexes created successfully")

        except Exception as e:
            logger.error("Failed to create MongoDB indexes", error=str(e))
            # Don't raise exception - indexes are important but not critical for startup

    async def ensure_collection_settings(self) -> None:
        """Ensure collections have proper settings and validation rules."""
        if not self.database:
            return

        try:
            # Set up collection validation rules
            conversations_validation = {
                "validator": {
                    "$jsonSchema": {
                        "bsonType": "object",
                        "required": ["conversation_id", "tenant_id", "user_id", "channel", "status"],
                        "properties": {
                            "conversation_id": {"bsonType": "string", "minLength": 1},
                            "tenant_id": {"bsonType": "string", "minLength": 1},
                            "user_id": {"bsonType": "string", "minLength": 1},
                            "channel": {"enum": ["web", "whatsapp", "messenger", "slack", "teams", "sms", "voice"]},
                            "status": {"enum": ["active", "completed", "abandoned", "escalated", "error"]}
                        }
                    }
                }
            }

            # Apply validation (this will only work if collection doesn't exist yet)
            try:
                await self.database.create_collection("conversations", **conversations_validation)
            except OperationFailure:
                # Collection already exists, validation rules can't be changed easily
                pass

            logger.info("MongoDB collection settings configured")

        except Exception as e:
            logger.error("Failed to configure collection settings", error=str(e))

    async def cleanup_expired_data(self) -> Dict[str, int]:
        """
        Clean up expired data based on retention policies.

        Returns:
            Dictionary with cleanup counts
        """
        cleanup_results = {
            "conversations_cleaned": 0,
            "messages_cleaned": 0,
            "sessions_cleaned": 0
        }

        if not self.database:
            return cleanup_results

        try:
            current_time = datetime.utcnow()

            # Clean up conversations with expired retention
            conversations = self.database.conversations
            conv_result = await conversations.delete_many({
                "compliance.data_retention_until": {"$lt": current_time}
            })
            cleanup_results["conversations_cleaned"] = conv_result.deleted_count

            # Clean up messages with expired auto-delete
            messages = self.database.messages
            msg_result = await messages.delete_many({
                "privacy.auto_delete_at": {"$lt": current_time}
            })
            cleanup_results["messages_cleaned"] = msg_result.deleted_count

            # Clean up expired sessions
            sessions = self.database.sessions
            session_result = await sessions.delete_many({
                "$or": [
                    {"expires_at": {"$lt": current_time}},
                    {"data_retention_until": {"$lt": current_time}}
                ]
            })
            cleanup_results["sessions_cleaned"] = session_result.deleted_count

            if sum(cleanup_results.values()) > 0:
                logger.info("MongoDB cleanup completed", **cleanup_results)

        except Exception as e:
            logger.error("MongoDB cleanup failed", error=str(e))

        return cleanup_results

    async def get_collection_stats(self) -> Dict[str, Dict[str, Any]]:
        """
        Get statistics for all collections.

        Returns:
            Dictionary with collection statistics
        """
        stats = {}

        if not self.database:
            return stats

        try:
            collections = ["conversations", "messages", "sessions"]

            for collection_name in collections:
                collection = self.database[collection_name]

                # Get collection stats
                coll_stats = await self.database.command("collStats", collection_name)

                # Get document count
                doc_count = await collection.count_documents({})

                stats[collection_name] = {
                    "document_count": doc_count,
                    "storage_size": coll_stats.get("storageSize", 0),
                    "total_index_size": coll_stats.get("totalIndexSize", 0),
                    "avg_obj_size": coll_stats.get("avgObjSize", 0),
                    "indexes": coll_stats.get("nindexes", 0)
                }

        except Exception as e:
            logger.error("Failed to get collection stats", error=str(e))

        return stats


# Global MongoDB manager instance
mongodb_manager = MongoDBManager()


async def get_mongodb() -> AsyncIOMotorDatabase:
    """
    Dependency function to get MongoDB database instance.

    Returns:
        AsyncIOMotorDatabase instance
    """
    return mongodb_manager.get_database()


async def get_mongodb_collection(collection_name: str) -> AsyncIOMotorCollection:
    """
    Dependency function to get specific MongoDB collection.

    Args:
        collection_name: Name of the collection

    Returns:
        AsyncIOMotorCollection instance
    """
    return mongodb_manager.get_collection(collection_name)


# Convenience functions for specific collections
async def get_conversations_collection() -> AsyncIOMotorCollection:
    """Get conversations collection."""
    return mongodb_manager.get_collection("conversations")


async def get_messages_collection() -> AsyncIOMotorCollection:
    """Get messages collection."""
    return mongodb_manager.get_collection("messages")


async def get_sessions_collection() -> AsyncIOMotorCollection:
    """Get sessions collection."""
    return mongodb_manager.get_collection("sessions")