"""
Message Service

Core message processing service that orchestrates the entire message handling pipeline.
Handles incoming messages, processes them through various stages, generates responses,
and manages delivery through appropriate channels.
"""

from datetime import datetime
from typing import Dict, Any, Optional, List
from uuid import uuid4

from src.services.base_service import BaseService
from src.services.exceptions import ServiceError, ValidationError, ProcessingError, DeliveryError
from src.models.schemas.request_schemas import SendMessageRequest
from src.models.schemas.response_schemas import MessageResponse
from src.models.types import MessageContent, ChannelType, ConversationStatus
from src.models.mongo.conversation_model import ConversationDocument
from src.models.mongo.message_model import MessageDocument
from src.repositories.conversation_repository import ConversationRepository
from src.repositories.message_repository import MessageRepository
from src.repositories.session_repository import SessionRepository
from src.core.channels.channel_factory import ChannelFactory
from src.core.processors.processor_factory import ProcessorFactory
from src.core.processors.base_processor import ProcessingContext


class MessageService(BaseService):
    """Service for message processing and orchestration"""

    def __init__(
            self,
            conversation_repo: ConversationRepository,
            message_repo: MessageRepository,
            session_repo: SessionRepository,
            channel_factory: ChannelFactory,
            processor_factory: ProcessorFactory
    ):
        super().__init__()
        self.conversation_repo = conversation_repo
        self.message_repo = message_repo
        self.session_repo = session_repo
        self.channel_factory = channel_factory
        self.processor_factory = processor_factory

    async def process_message(
            self,
            request: SendMessageRequest,
            user_context: Dict[str, Any]
    ) -> MessageResponse:
        """
        Process incoming message and generate response

        Args:
            request: Message request data
            user_context: User authentication context

        Returns:
            MessageResponse with bot response
        """
        start_time = datetime.utcnow()

        try:
            # Validate tenant access
            await self.validate_tenant_access(
                request.tenant_id,
                user_context
            )

            self.log_operation(
                "process_message_start",
                tenant_id=request.tenant_id,
                user_id=request.user_id,
                channel=request.channel,
                message_type=request.content.type
            )

            # Get or create conversation
            conversation = await self._get_or_create_conversation(request)

            # Process incoming message
            processed_message = await self._process_incoming_message(
                request, conversation, user_context
            )

            # Store incoming message
            await self.message_repo.create(processed_message)

            # Generate bot response
            response_content = await self._generate_bot_response(
                request, conversation, processed_message
            )

            # Create and store response message
            response_message = await self._create_response_message(
                request, conversation, response_content
            )
            await self.message_repo.create(response_message)

            # Deliver response via channel
            delivery_result = await self._deliver_response(
                request, response_content
            )

            # Update conversation metrics
            await self._update_conversation_metrics(
                conversation, processed_message, response_message
            )

            # Calculate processing time
            processing_time = int(
                (datetime.utcnow() - start_time).total_seconds() * 1000
            )

            self.log_operation(
                "process_message_complete",
                tenant_id=request.tenant_id,
                conversation_id=conversation.conversation_id,
                processing_time_ms=processing_time,
                delivery_success=delivery_result.success
            )

            return MessageResponse(
                message_id=processed_message.message_id,
                conversation_id=conversation.conversation_id,
                response=response_content,
                conversation_state=conversation.context.dict(),
                processing_metadata={
                    "processing_time_ms": processing_time,
                    "delivery_status": delivery_result.delivery_status,
                    "channel_response": delivery_result.dict()
                }
            )

        except Exception as e:
            error = self.handle_service_error(
                e, "process_message",
                tenant_id=getattr(request, 'tenant_id', None)
            )
            raise error

    async def _get_or_create_conversation(
            self,
            request: SendMessageRequest
    ) -> ConversationDocument:
        """Get existing or create new conversation"""
        try:
            # Try to get existing conversation
            if request.conversation_id:
                conversation = await self.conversation_repo.get_by_id(
                    request.conversation_id
                )
                if conversation:
                    conversation.update_last_activity()
                    return conversation

            # Create new conversation
            conversation = ConversationDocument(
                conversation_id=str(uuid4()),
                tenant_id=request.tenant_id,
                user_id=request.user_id,
                session_id=request.session_id,
                channel=request.channel,
                channel_metadata=request.channel_metadata.dict() if request.channel_metadata else {},
                status=ConversationStatus.ACTIVE
            )

            return await self.conversation_repo.create(conversation)

        except Exception as e:
            raise ServiceError(f"Failed to get or create conversation: {e}")

    async def _process_incoming_message(
            self,
            request: SendMessageRequest,
            conversation: ConversationDocument,
            user_context: Dict[str, Any]
    ) -> MessageDocument:
        """Process incoming message through processing pipeline"""
        try:
            # Create processing context
            context = ProcessingContext(
                tenant_id=request.tenant_id,
                user_id=request.user_id,
                conversation_id=conversation.conversation_id,
                session_id=request.session_id,
                channel=request.channel.value,
                channel_metadata=request.channel_metadata.dict() if request.channel_metadata else {},
                user_profile=user_context,
                conversation_context=conversation.context.dict(),
                processing_hints=request.processing_hints.dict() if request.processing_hints else {},
                request_id=str(uuid4())
            )

            # Get appropriate processor
            processor = await self.processor_factory.get_processor(
                request.content.type
            )

            # Process content
            processing_result = await processor.process(request.content, context)

            # Create message document
            message_doc = MessageDocument(
                message_id=request.message_id,
                conversation_id=conversation.conversation_id,
                tenant_id=request.tenant_id,
                user_id=request.user_id,
                sequence_number=await self._get_next_sequence_number(
                    conversation.conversation_id
                ),
                direction="inbound",
                timestamp=request.timestamp,
                channel=request.channel.value,
                message_type=request.content.type.value,
                content=processing_result.processed_content.dict() if processing_result.processed_content else request.content.dict(),
                ai_analysis={
                    "entities": processing_result.entities,
                    "detected_language": processing_result.detected_language,
                    "language_confidence": processing_result.language_confidence,
                    "content_categories": processing_result.content_categories,
                    "quality_score": processing_result.quality_score,
                    "safety_flags": processing_result.safety_flags
                },
                processing={
                    "pipeline_version": "1.0",
                    "processing_time_ms": processing_result.processing_time_ms,
                    "processor_version": processing_result.processor_version
                }
            )

            return message_doc

        except Exception as e:
            raise ProcessingError(f"Failed to process incoming message: {e}")

    async def _generate_bot_response(
            self,
            request: SendMessageRequest,
            conversation: ConversationDocument,
            processed_message: MessageDocument
    ) -> MessageContent:
        """Generate bot response content"""
        try:
            # This is a simplified response generation
            # In production, this would integrate with MCP Engine

            # Extract intent from processed message
            intent = processed_message.ai_analysis.get("entities", {}).get("intent")

            # Generate appropriate response based on intent and context
            if intent == "greeting":
                response_text = "Hello! How can I help you today?"
            elif intent == "order_inquiry":
                response_text = "I'd be happy to help you with your order. Could you please provide your order number?"
            elif intent == "support":
                response_text = "I'm here to help! Please describe the issue you're experiencing."
            else:
                response_text = "Thank you for your message. How can I assist you?"

            # Create response content
            response_content = MessageContent(
                type="text",
                text=response_text,
                language=processed_message.ai_analysis.get("detected_language", "en")
            )

            return response_content

        except Exception as e:
            raise ServiceError(f"Failed to generate bot response: {e}")

    async def _create_response_message(
            self,
            request: SendMessageRequest,
            conversation: ConversationDocument,
            response_content: MessageContent
    ) -> MessageDocument:
        """Create response message document"""
        try:
            return MessageDocument(
                message_id=str(uuid4()),
                conversation_id=conversation.conversation_id,
                tenant_id=request.tenant_id,
                user_id="bot",  # Bot user
                sequence_number=await self._get_next_sequence_number(
                    conversation.conversation_id
                ),
                direction="outbound",
                timestamp=datetime.utcnow(),
                channel=request.channel.value,
                message_type=response_content.type.value,
                content=response_content.dict(),
                generation_metadata={
                    "model_provider": "internal",
                    "model_name": "rule_based",
                    "generation_time_ms": 50,
                    "template_used": "default_response"
                }
            )

        except Exception as e:
            raise ServiceError(f"Failed to create response message: {e}")

    async def _deliver_response(
            self,
            request: SendMessageRequest,
            response_content: MessageContent
    ) -> Any:  # ChannelResponse
        """Deliver response via appropriate channel"""
        try:
            # Get channel implementation
            channel = await self.channel_factory.get_channel(request.channel)

            # Determine recipient from original request
            recipient = request.user_id  # Simplified - should be channel-specific

            # Deliver message
            delivery_result = await channel.send_message(
                recipient=recipient,
                content=response_content,
                metadata=request.channel_metadata.dict() if request.channel_metadata else {}
            )

            return delivery_result

        except Exception as e:
            raise DeliveryError(f"Failed to deliver response: {e}")

    async def _update_conversation_metrics(
            self,
            conversation: ConversationDocument,
            incoming_message: MessageDocument,
            response_message: MessageDocument
    ) -> None:
        """Update conversation metrics"""
        try:
            conversation.metrics.increment_message_count(True)  # User message
            conversation.metrics.increment_message_count(False)  # Bot message

            # Update conversation context if needed
            if incoming_message.ai_analysis.get("entities"):
                conversation.context.entities.update(
                    incoming_message.ai_analysis["entities"]
                )

            # Save updated conversation
            await self.conversation_repo.update(conversation)

        except Exception as e:
            self.logger.error(
                "Failed to update conversation metrics",
                conversation_id=conversation.conversation_id,
                error=str(e)
            )
            # Don't raise - this is not critical

    async def _get_next_sequence_number(self, conversation_id: str) -> int:
        """Get next sequence number for message in conversation"""
        try:
            # Get last message in conversation
            last_message = await self.message_repo.get_last_message_in_conversation(
                conversation_id
            )

            if last_message:
                return last_message.sequence_number + 1
            else:
                return 1

        except Exception as e:
            self.logger.error(
                "Failed to get next sequence number",
                conversation_id=conversation_id,
                error=str(e)
            )
            return 1  # Default to 1 if we can't determine

    async def handle_webhook(
            self,
            channel_type: ChannelType,
            webhook_data: Dict[str, Any],
            tenant_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """Process incoming webhook from channel"""
        try:
            self.log_operation(
                "handle_webhook",
                channel=channel_type,
                tenant_id=tenant_id
            )

            # Get channel implementation
            channel = await self.channel_factory.get_channel(channel_type)

            # Process webhook
            result = await channel.process_webhook(webhook_data)

            # Handle any events from webhook
            if "events" in result:
                for event in result["events"]:
                    await self._handle_webhook_event(event, channel_type, tenant_id)

            return result

        except Exception as e:
            error = self.handle_service_error(
                e, "handle_webhook",
                channel=channel_type,
                tenant_id=tenant_id
            )
            raise error

    async def _handle_webhook_event(
            self,
            event: Dict[str, Any],
            channel_type: ChannelType,
            tenant_id: Optional[str]
    ) -> None:
        """Handle individual webhook event"""
        try:
            event_type = event.get("type")

            if event_type == "message_received":
                # Convert webhook event to message request and process
                # This would involve mapping webhook data to SendMessageRequest
                pass
            elif event_type == "delivery_status":
                # Update message delivery status
                await self._update_delivery_status(event)
            elif event_type == "read_receipt":
                # Update message read status
                await self._update_read_status(event)

        except Exception as e:
            self.logger.error(
                "Failed to handle webhook event",
                event_type=event.get("type"),
                channel=channel_type,
                error=str(e)
            )

    async def _update_delivery_status(self, event: Dict[str, Any]) -> None:
        """Update message delivery status from webhook event"""
        try:
            message_id = event.get("message_id")
            delivery_status = event.get("status")
            timestamp = event.get("timestamp")

            if message_id and delivery_status:
                message = await self.message_repo.get_by_id(message_id)
                if message:
                    message.channel_metadata["delivery_status"] = delivery_status
                    message.channel_metadata["delivery_timestamp"] = timestamp
                    await self.message_repo.update(message)

        except Exception as e:
            self.logger.error(
                "Failed to update delivery status",
                event=event,
                error=str(e)
            )

    async def _update_read_status(self, event: Dict[str, Any]) -> None:
        """Update message read status from webhook event"""
        try:
            message_id = event.get("message_id")
            read_timestamp = event.get("timestamp")

            if message_id and read_timestamp:
                message = await self.message_repo.get_by_id(message_id)
                if message:
                    message.channel_metadata["read_timestamp"] = read_timestamp
                    await self.message_repo.update(message)

        except Exception as e:
            self.logger.error(
                "Failed to update read status",
                event=event,
                error=str(e)
            )

    async def get_conversation_messages(
            self,
            tenant_id: str,
            conversation_id: str,
            limit: int = 50,
            offset: int = 0,
            user_context: Optional[Dict[str, Any]] = None
    ) -> List[MessageDocument]:
        """
        Get messages for a conversation

        Args:
            tenant_id: Tenant identifier
            conversation_id: Conversation identifier
            limit: Maximum number of messages to return
            offset: Number of messages to skip
            user_context: User authentication context

        Returns:
            List of message documents
        """
        try:
            await self.validate_tenant_access(tenant_id, user_context)

            messages = await self.message_repo.get_conversation_messages(
                conversation_id, limit, offset
            )

            self.log_operation(
                "get_conversation_messages",
                tenant_id=tenant_id,
                conversation_id=conversation_id,
                message_count=len(messages)
            )

            return messages

        except Exception as e:
            error = self.handle_service_error(
                e, "get_conversation_messages",
                tenant_id=tenant_id
            )
            raise error

    async def search_messages(
            self,
            tenant_id: str,
            query: str,
            filters: Optional[Dict[str, Any]] = None,
            limit: int = 20,
            offset: int = 0,
            user_context: Optional[Dict[str, Any]] = None
    ) -> List[MessageDocument]:
        """
        Search messages by content and filters

        Args:
            tenant_id: Tenant identifier
            query: Search query string
            filters: Additional search filters
            limit: Maximum number of results
            offset: Number of results to skip
            user_context: User authentication context

        Returns:
            List of matching message documents
        """
        try:
            await self.validate_tenant_access(tenant_id, user_context)

            search_filters = filters or {}
            search_filters["tenant_id"] = tenant_id

            messages = await self.message_repo.search_messages(
                query, search_filters, limit, offset
            )

            self.log_operation(
                "search_messages",
                tenant_id=tenant_id,
                query=query,
                result_count=len(messages)
            )

            return messages

        except Exception as e:
            error = self.handle_service_error(
                e, "search_messages",
                tenant_id=tenant_id
            )
            raise error