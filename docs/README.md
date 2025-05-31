# Chat Service Documentation

## Overview

The Chat Service is a core component of the Multi-Tenant AI Chatbot Platform responsible for:

- Message ingestion and normalization across channels
- Real-time conversation management
- Message delivery and routing
- Session state management
- Integration with MCP Engine for response processing

## Architecture

``` mermaid
┌─────────────────────────────────────────────────────────────────┐
│                      CHAT SERVICE ARCHITECTURE                  │
└─────────────────────────────────────────────────────────────────┘

┌──────────────┐    ┌──────────────┐    ┌──────────────┐
│   Web Chat   │    │   WhatsApp   │    │  Messenger   │
│   Channel    │    │   Channel    │    │   Channel    │
└──────┬───────┘    └──────┬───────┘    └──────┬───────┘
       │                   │                   │
       └───────────────────┼───────────────────┘
                           │
              ┌────────────▼────────────┐
              │      API Gateway        │
              │   (Rate Limiting &      │
              │   Authentication)       │ 
              └────────────┬────────────┘
                           │
              ┌────────────▼────────────┐
              │     Chat Service        │
              │                         │
              │  ┌─────────────────┐    │
              │  │ Message Router  │    │
              │  └─────────────────┘    │
              │  ┌─────────────────┐    │
              │  │ State Manager   │    │
              │  └─────────────────┘    │
              │  ┌─────────────────┐    │
              │  │ Session Cache   │    │
              │  └─────────────────┘    │
              └────────────┬────────────┘
                           │
       ┌───────────────────┼───────────────────┐
       │                   │                   │
┌──────▼──────┐   ┌────────▼────────┐   ┌──────▼──────┐
│  MongoDB    │   │   MCP Engine    │   │   Redis     │
│(Conversations│  │  (Processing)  │   │  (Cache)    │
│ & Messages) │   │                 │   │             │
└─────────────┘   └─────────────────┘   └─────────────┘
```
## Features

### Multi-Channel Support
- **Web Chat**: Real-time browser-based conversations
- **WhatsApp Business API**: Rich messaging with media support
- **Facebook Messenger**: Full platform integration
- **Slack**: Workspace and DM conversations
- **Microsoft Teams**: Enterprise chat integration
- **Voice**: Speech-to-text and text-to-speech
- **SMS**: Basic text messaging fallback

### Message Processing
- **Content Normalization**: Standardize messages across channels
- **Media Handling**: Images, files, audio, video processing
- **Rich Content Support**: Buttons, carousels, quick replies
- **Location Services**: Geographic data processing
- **Template Management**: Reusable message templates

### Session Management
- **Distributed Sessions**: Redis-backed session storage
- **Context Preservation**: Maintain conversation state
- **Multi-Device Support**: Seamless cross-device experiences
- **Session Analytics**: Usage patterns and metrics

### Performance & Reliability
- **Horizontal Scaling**: Kubernetes-based auto-scaling
- **Circuit Breakers**: Fault tolerance for external services
- **Rate Limiting**: Per-tenant and per-user limits
- **Health Monitoring**: Comprehensive health checks
- **Caching Strategy**: Multi-layer caching for performance

## Quick Start

### Prerequisites
- Python 3.11+
- Docker & Docker Compose
- MongoDB 7+
- Redis 7+
- PostgreSQL 15+

### Local Development Setup

1. **Clone and Setup Environment**
   ```bash
   git clone <repository_url>
   cd chat-service
   chmod +x scripts/setup.sh
   ./scripts/setup.sh
   ```

2. **Configure Environment**
   ```bash
   cp .env.example .env
   # Edit .env with your local configuration
   ```

3. **Start Dependencies**
   ```bash
   docker-compose -f docker/docker-compose.dev.yml up -d
   ```

4. **Run the Service**
   ```bash
   source venv/bin/activate
   ./scripts/start.sh
   ```

5. **Verify Installation**
   ```bash
   curl http://localhost:8001/health
   ./scripts/health_check.sh
   ```

### API Usage

#### Send a Message
```bash
curl -X POST http://localhost:8001/api/v2/chat/message \
  -H "Content-Type: application/json" \
  -H "X-Tenant-ID: your-tenant-id" \
  -H "Authorization: Bearer your-jwt-token" \
  -d '{
    "user_id": "user123",
    "channel": "web",
    "content": {
      "type": "text",
      "text": "Hello, I need help with my order"
    }
  }'
```

#### Get Conversation History
```bash
curl http://localhost:8001/api/v2/chat/conversations/conv-id \
  -H "X-Tenant-ID: your-tenant-id" \
  -H "Authorization: Bearer your-jwt-token"
```

## Configuration

### Environment Variables

| Variable | Description | Default | Required |
|----------|-------------|---------|----------|
| `ENVIRONMENT` | Deployment environment | `development` | No |
| `HOST` | Service host address | `0.0.0.0` | No |
| `PORT` | Service port | `8001` | No |
| `MONGODB_URI` | MongoDB connection string | `mongodb://localhost:27017` | Yes |
| `REDIS_URL` | Redis connection string | `redis://localhost:6379` | Yes |
| `POSTGRESQL_URI` | PostgreSQL connection string | See settings.py | Yes |
| `KAFKA_BROKERS` | Kafka broker list | `["localhost:9092"]` | Yes |
| `MCP_ENGINE_URL` | MCP Engine gRPC endpoint | `localhost:50051` | Yes |
| `SECURITY_HUB_URL` | Security Hub gRPC endpoint | `localhost:50052` | Yes |

### Performance Tuning

```python
# Connection Pool Settings
MAX_CONNECTIONS_MONGO = 100
MAX_CONNECTIONS_REDIS = 50
REQUEST_TIMEOUT_MS = 30000

# Cache TTL Settings
SESSION_TTL = 3600  # 1 hour
CONFIG_CACHE_TTL = 300  # 5 minutes
RESPONSE_CACHE_TTL = 1800  # 30 minutes

# Rate Limiting
DEFAULT_RATE_LIMIT = 1000  # requests per minute
BURST_LIMIT = 2000  # burst capacity
```

## Testing

### Unit Tests
```bash
pytest tests/unit/ -v --cov=src/
```

### Integration Tests
```bash
pytest tests/integration/ -v
```

### End-to-End Tests
```bash
pytest tests/e2e/ -v --slow
```

### Performance Tests
```bash
locust -f tests/performance/locustfile.py --host=http://localhost:8001
```

## Monitoring & Observability

### Health Endpoints
- `/health` - Basic health check
- `/health/detailed` - Comprehensive system status
- `/metrics` - Prometheus metrics
- `/info` - Service information

### Logging
- **Structured Logging**: JSON format with correlation IDs
- **Log Levels**: DEBUG, INFO, WARNING, ERROR, CRITICAL
- **Log Aggregation**: Compatible with ELK stack
- **Correlation**: Request tracking across services

### Metrics
- **Request Metrics**: Latency, throughput, error rates
- **Business Metrics**: Conversations, completion rates
- **System Metrics**: Memory, CPU, connection pools
- **Custom Metrics**: Tenant-specific KPIs

## Deployment

### Kubernetes
```bash
kubectl apply -f k8s/namespace.yaml
kubectl apply -f k8s/configmap.yaml
kubectl apply -f k8s/secrets.yaml
kubectl apply -f k8s/deployment.yaml
kubectl apply -f k8s/service.yaml
kubectl apply -f k8s/ingress.yaml
```

### Docker
```bash
docker build -t chat-service:latest -f docker/Dockerfile .
docker run -p 8001:8001 --env-file .env chat-service:latest
```

### Scaling Considerations
- **Horizontal Scaling**: Stateless design allows easy scaling
- **Database Sharding**: MongoDB sharding by tenant_id
- **Cache Partitioning**: Redis cluster for distributed caching
- **Load Balancing**: Round-robin with health checks

## Troubleshooting

### Common Issues

#### High Memory Usage
```bash
# Check memory usage
docker stats chat-service
# Adjust memory limits in k8s/deployment.yaml
```

#### Database Connection Issues
```bash
# Check MongoDB connectivity
mongosh "mongodb://your-connection-string"
# Verify Redis connectivity
redis-cli -u "redis://your-redis-url" ping
```

#### Message Processing Delays
```bash
# Check queue depths
redis-cli llen "queue:messages"
# Monitor processing times
curl http://localhost:8001/metrics | grep processing_time
```

### Performance Tuning
1. **Optimize Database Queries**: Use proper indexes
2. **Tune Connection Pools**: Adjust based on load
3. **Enable Caching**: Use Redis for frequently accessed data
4. **Scale Horizontally**: Add more service instances

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for development guidelines.

## Security

- **Authentication**: JWT tokens and API keys
- **Authorization**: Role-based access control
- **Data Encryption**: TLS in transit, encryption at rest
- **Input Validation**: Comprehensive request validation
- **Rate Limiting**: DDoS protection and fair usage

## Support

- **Documentation**: [API Docs](API.md)
- **Issues**: GitHub Issues
- **Slack**: #chat-service-support
- **Email**: chatbot-platform-support@company.com
```