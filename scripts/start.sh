#!/bin/bash
set -e

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Load environment variables
if [ -f .env ]; then
    export $(cat .env | grep -v '^#' | xargs)
    print_status "Loaded configuration from .env file"
else
    print_warning ".env file not found, using default configuration"
fi

# Set default values
HOST=${HOST:-"0.0.0.0"}
PORT=${PORT:-8001}
ENVIRONMENT=${ENVIRONMENT:-"development"}
LOG_LEVEL=${LOG_LEVEL:-"INFO"}
RELOAD_ON_CHANGE=${RELOAD_ON_CHANGE:-"true"}

print_status "Starting Chat Service..."
print_status "Environment: $ENVIRONMENT"
print_status "Host: $HOST"
print_status "Port: $PORT"
print_status "Log Level: $LOG_LEVEL"

# Check if virtual environment is activated
if [ -z "$VIRTUAL_ENV" ]; then
    if [ -d "venv" ]; then
        print_status "Activating virtual environment..."
        source venv/bin/activate
    else
        print_error "Virtual environment not found. Run ./scripts/setup.sh first"
        exit 1
    fi
fi

# Validate environment configuration
print_status "Validating configuration..."

# Check required environment variables
required_vars=("MONGODB_URI" "REDIS_URL" "POSTGRESQL_URI")
missing_vars=()

for var in "${required_vars[@]}"; do
    if [ -z "${!var}" ]; then
        missing_vars+=("$var")
    fi
done

if [ ${#missing_vars[@]} -ne 0 ]; then
    print_error "Missing required environment variables:"
    for var in "${missing_vars[@]}"; do
        echo "  - $var"
    done
    print_status "Please update your .env file or set these variables"
    exit 1
fi

# Test database connections
print_status "Testing database connections..."

# Test MongoDB connection
if command -v mongosh &> /dev/null; then
    if mongosh "$MONGODB_URI" --eval "db.runCommand('ping')" --quiet &> /dev/null; then
        print_success "MongoDB connection successful"
    else
        print_warning "MongoDB connection failed. Make sure MongoDB is running"
    fi
else
    print_warning "mongosh not found. Skipping MongoDB connection test"
fi

# Test Redis connection
if command -v redis-cli &> /dev/null; then
    redis_host=$(echo $REDIS_URL | sed -n 's/redis:\/\/\([^:]*\).*/\1/p')
    redis_port=$(echo $REDIS_URL | sed -n 's/redis:\/\/[^:]*:\([0-9]*\).*/\1/p')
    redis_host=${redis_host:-"localhost"}
    redis_port=${redis_port:-"6379"}
    
    if redis-cli -h "$redis_host" -p "$redis_port" ping &> /dev/null; then
        print_success "Redis connection successful"
    else
        print_warning "Redis connection failed. Make sure Redis is running"
    fi
else
    print_warning "redis-cli not found. Skipping Redis connection test"
fi

# Test PostgreSQL connection
if command -v psql &> /dev/null; then
    if psql "$POSTGRESQL_URI" -c "SELECT 1;" &> /dev/null; then
        print_success "PostgreSQL connection successful"
    else
        print_warning "PostgreSQL connection failed. Make sure PostgreSQL is running"
    fi
else
    print_warning "psql not found. Skipping PostgreSQL connection test"
fi

# Check if port is available
if lsof -Pi :$PORT -sTCP:LISTEN -t >/dev/null; then
    print_error "Port $PORT is already in use"
    print_status "Please stop the service using that port or change the PORT in .env"
    exit 1
fi

# Create logs directory if it doesn't exist
mkdir -p logs

# Determine uvicorn command based on environment
if [ "$ENVIRONMENT" = "development" ]; then
    if [ "$RELOAD_ON_CHANGE" = "true" ]; then
        reload_flag="--reload"
        print_status "Auto-reload enabled for development"
    else
        reload_flag=""
    fi
    
    # Development mode with detailed logging
    uvicorn_cmd="uvicorn src.main:app \
        --host $HOST \
        --port $PORT \
        --log-level ${LOG_LEVEL,,} \
        --access-log \
        --use-colors \
        $reload_flag"
else
    # Production-like mode
    uvicorn_cmd="uvicorn src.main:app \
        --host $HOST \
        --port $PORT \
        --log-level ${LOG_LEVEL,,} \
        --access-log \
        --workers 1"
fi

print_status "Starting server with command: $uvicorn_cmd"
print_success "Chat Service will be available at: http://$HOST:$PORT"
print_status "Health check endpoint: http://$HOST:$PORT/health"
print_status "API documentation: http://$HOST:$PORT/docs"
print_status "Press Ctrl+C to stop the server"

# Function to handle cleanup on exit
cleanup() {
    print_status "Shutting down Chat Service..."
    print_success "Service stopped successfully"
}

# Set trap to call cleanup function on script exit
trap cleanup EXIT

# Start the server
exec $uvicorn_cmd