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

# Default configuration
HOST=${HOST:-"localhost"}
PORT=${PORT:-8001}
TIMEOUT=${TIMEOUT:-30}
DETAILED=${DETAILED:-false}
WAIT_FOR_READY=${WAIT_FOR_READY:-false}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --host)
            HOST="$2"
            shift 2
            ;;
        --port)
            PORT="$2"
            shift 2
            ;;
        --timeout)
            TIMEOUT="$2"
            shift 2
            ;;
        --detailed)
            DETAILED=true
            shift
            ;;
        --wait)
            WAIT_FOR_READY=true
            shift
            ;;
        -h|--help)
            echo "Usage: $0 [OPTIONS]"
            echo "Options:"
            echo "  --host HOST      Service host (default: localhost)"
            echo "  --port PORT      Service port (default: 8001)"
            echo "  --timeout SEC    Timeout in seconds (default: 30)"
            echo "  --detailed       Show detailed health information"
            echo "  --wait           Wait for service to be ready"
            echo "  -h, --help       Show this help message"
            exit 0
            ;;
        *)
            print_error "Unknown option: $1"
            exit 1
            ;;
    esac
done

SERVICE_URL="http://$HOST:$PORT"
HEALTH_URL="$SERVICE_URL/health"
DETAILED_HEALTH_URL="$SERVICE_URL/health/detailed"

print_status "Chat Service Health Check"
print_status "Service URL: $SERVICE_URL"

# Function to check if service is responding
check_basic_health() {
    if curl -f -s "$HEALTH_URL" --max-time $TIMEOUT > /dev/null 2>&1; then
        return 0
    else
        return 1
    fi
}

# Function to get basic health status
get_basic_health() {
    local response
    response=$(curl -f -s "$HEALTH_URL" --max-time $TIMEOUT 2>/dev/null)
    echo "$response"
}

# Function to get detailed health status
get_detailed_health() {
    local response
    response=$(curl -f -s "$DETAILED_HEALTH_URL" --max-time $TIMEOUT 2>/dev/null)
    echo "$response"
}

# Function to wait for service to be ready
wait_for_service() {
    local attempts=0
    local max_attempts=$((TIMEOUT))
    
    print_status "Waiting for service to be ready (timeout: ${TIMEOUT}s)..."
    
    while [ $attempts -lt $max_attempts ]; do
        if check_basic_health; then
            print_success "Service is ready!"
            return 0
        fi
        
        sleep 1
        ((attempts++))
        
        # Show progress every 5 seconds
        if [ $((attempts % 5)) -eq 0 ]; then
            print_status "Still waiting... (${attempts}s elapsed)"
        fi
    done
    
    print_error "Service did not become ready within ${TIMEOUT} seconds"
    return 1
}

# Function to parse and display health information
display_health_info() {
    local health_data="$1"
    local is_detailed="$2"
    
    # Check if response is valid JSON
    if ! echo "$health_data" | python3 -m json.tool > /dev/null 2>&1; then
        print_error "Invalid JSON response from health endpoint"
        echo "Response: $health_data"
        return 1
    fi
    
    # Extract basic information
    local status=$(echo "$health_data" | python3 -c "import sys, json; data=json.load(sys.stdin); print(data.get('status', 'unknown'))")
    local timestamp=$(echo "$health_data" | python3 -c "import sys, json; data=json.load(sys.stdin); print(data.get('timestamp', 'unknown'))")
    local version=$(echo "$health_data" | python3 -c "import sys, json; data=json.load(sys.stdin); print(data.get('version', 'unknown'))")
    
    echo "┌─────────────────────────────────────────────┐"
    echo "│              Service Health                 │"
    echo "└─────────────────────────────────────────────┘"
    echo "Status:    $status"
    echo "Version:   $version"
    echo "Timestamp: $timestamp"
    
    if [ "$status" = "healthy" ]; then
        print_success "Service is healthy ✅"
    elif [ "$status" = "degraded" ]; then
        print_warning "Service is degraded ⚠️"
    else
        print_error "Service is unhealthy ❌"
    fi
    
    # Display detailed information if available
    if [ "$is_detailed" = true ]; then
        echo
        echo "┌─────────────────────────────────────────────┐"
        echo "│            Detailed Information             │"
        echo "└─────────────────────────────────────────────┘"
        
        # Dependencies status
        local dependencies=$(echo "$health_data" | python3 -c "
import sys, json
data = json.load(sys.stdin)
deps = data.get('dependencies', {})
for name, info in deps.items():
    status = info.get('status', 'unknown')
    latency = info.get('latency_ms', 'N/A')
    print(f'{name:15} {status:10} {latency:>6}ms')
" 2>/dev/null)
        
        if [ -n "$dependencies" ]; then
            echo "Dependencies:"
            echo "$dependencies"
        fi
        
        # System metrics
        local metrics=$(echo "$health_data" | python3 -c "
import sys, json
data = json.load(sys.stdin)
metrics = data.get('metrics', {})
if metrics:
    print(f\"Memory Usage:   {metrics.get('memory_usage_mb', 'N/A')} MB\")
    print(f\"CPU Usage:      {metrics.get('cpu_usage_percent', 'N/A')}%\")
    print(f\"Uptime:         {metrics.get('uptime_seconds', 'N/A')}s\")
    print(f\"Active Conns:   {metrics.get('active_connections', 'N/A')}\")
" 2>/dev/null)
        
        if [ -n "$metrics" ]; then
            echo
            echo "System Metrics:"
            echo "$metrics"
        fi
    fi
}

# Main health check logic
main() {
    # Wait for service if requested
    if [ "$WAIT_FOR_READY" = true ]; then
        if ! wait_for_service; then
            exit 1
        fi
    fi
    
    # Perform health check
    print_status "Checking service health..."
    
    if check_basic_health; then
        # Get health information
        if [ "$DETAILED" = true ]; then
            print_status "Fetching detailed health information..."
            health_data=$(get_detailed_health)
            if [ $? -eq 0 ] && [ -n "$health_data" ]; then
                display_health_info "$health_data" true
            else
                print_warning "Could not fetch detailed health information, falling back to basic check"
                health_data=$(get_basic_health)
                display_health_info "$health_data" false
            fi
        else
            health_data=$(get_basic_health)
            display_health_info "$health_data" false
        fi
        
        # Additional endpoint checks
        print_status "Checking additional endpoints..."
        
        # Check API docs
        if curl -f -s "$SERVICE_URL/docs" --max-time 5 > /dev/null 2>&1; then
            print_success "API documentation: ✅ $SERVICE_URL/docs"
        else
            print_warning "API documentation: ❌ $SERVICE_URL/docs"
        fi
        
        # Check OpenAPI spec
        if curl -f -s "$SERVICE_URL/openapi.json" --max-time 5 > /dev/null 2>&1; then
            print_success "OpenAPI spec: ✅ $SERVICE_URL/openapi.json"
        else
            print_warning "OpenAPI spec: ❌ $SERVICE_URL/openapi.json"
        fi
        
        # Check metrics endpoint (if available)
        if curl -f -s "$SERVICE_URL/metrics" --max-time 5 > /dev/null 2>&1; then
            print_success "Metrics endpoint: ✅ $SERVICE_URL/metrics"
        else
            print_status "Metrics endpoint: Not available"
        fi
        
        exit 0
    else
        print_error "Service health check failed"
        print_status "Service is not responding at $HEALTH_URL"
        
        # Additional debugging information
        print_status "Debugging information:"
        echo "- Check if the service is running"
        echo "- Verify the host and port are correct"
        echo "- Check firewall and network connectivity"
        echo "- Review service logs for errors"
        
        # Try to get more information about the connection failure
        if command -v nc &> /dev/null; then
            if nc -z "$HOST" "$PORT" 2>/dev/null; then
                print_status "Port $PORT is open but service is not responding to HTTP requests"
            else
                print_status "Cannot connect to port $PORT on $HOST"
            fi
        fi
        
        exit 1
    fi
}

# Run main function
main