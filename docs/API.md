# Chat Service API Documentation

## Authentication
All API requests require authentication via JWT token or API key.

### Headers
```http
Authorization: Bearer <jwt_token>
X-Tenant-ID: <tenant_uuid>
Content-Type: application/json
```

## Endpoints

### Send Message
Send a message through the chat system.

**Request:**
```http
POST /api/v2/chat/message
```

**Body:**
```json
{
  "user_id": "user123",
  "channel": "web",
  "content": {
    "type": "text",
    "text": "Hello, I need help"
  }
}
```

**Response:**
```json
{
  "status": "success",
  "data": {
    "message_id": "msg_uuid",
    "conversation_id": "conv_uuid",
    "response": {
      "type": "text",
      "text": "Hi! How can I help you today?"
    }
  }
}
```

### Get Conversation
Retrieve conversation history.

**Request:**
```http
GET /api/v2/chat/conversations/{conversation_id}
```

**Response:**
```json
{
  "status": "success",
  "data": {
    "conversation_id": "conv_uuid",
    "messages": [
      {
        "message_id": "msg_1",
        "direction": "inbound",
        "content": {
          "type": "text",
          "text": "Hello"
        },
        "timestamp": "2025-05-30T10:00:00Z"
      }
    ]
  }
}
```

For complete API documentation, see the [API Specifications](../02-API-Specifications.md).