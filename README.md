# Chat Service

Multi-tenant AI chatbot platform - Chat Service component for message processing and conversation management.

## 🚀 Quick Start

### Prerequisites
- Python 3.11+
- Docker & Docker Compose
- MongoDB 7+
- Redis 7+
- PostgreSQL 15+

### Installation
```bash
# Clone repository
git clone <repository_url>
cd chat-service

# Setup development environment
chmod +x scripts/setup.sh
./scripts/setup.sh

# Start services
./scripts/start.sh


### Health Check
```bash
curl http://localhost:8001/health
```

## 📁 Project Structure
```
chat-service/
├── src/                    # Source code
│   ├── api/               # API layer
│   ├── core/              # Business logic
│   ├── models/            # Data models
│   ├── services/          # Service layer
│   ├── repositories/      # Data access
│   ├── utils/             # Utilities
│   ├── config/            # Configuration
│   └── exceptions/        # Custom exceptions
├── tests/                 # Test suite
├── docs/                  # Documentation
├── scripts/               # Utility scripts
├── docker/                # Docker configuration
└── k8s/                   # Kubernetes manifests
```

## 🔧 Configuration

### Environment Variables
Copy `.env.example` to `.env` and configure:

```env
ENVIRONMENT=development
MONGODB_URI=mongodb://localhost:27017
REDIS_URL=redis://localhost:6379
POSTGRESQL_URI=postgresql://postgres:postgres@localhost:5432/chatbot_config
MCP_ENGINE_URL=localhost:50051
SECURITY_HUB_URL=localhost:50052
```

## 🧪 Testing
```bash
# Unit tests
pytest tests/unit/ -v

# Integration tests
pytest tests/integration/ -v

# All tests with coverage
pytest --cov=src/
```

## 📚 Documentation
- [API Documentation](docs/API.md)
- [Contributing Guidelines](docs/CONTRIBUTING.md)
- [Deployment Guide](docs/DEPLOYMENT.md)

## 🤝 Contributing
See [CONTRIBUTING.md](docs/CONTRIBUTING.md) for development guidelines.

## 📄 License
License - see LICENSE file for details.
