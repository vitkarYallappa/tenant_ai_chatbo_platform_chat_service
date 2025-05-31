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

# Default values
TEST_TYPE="all"
COVERAGE_REPORT="term-missing"
VERBOSE=false
PARALLEL=false
BENCHMARK=false
MARKERS=""

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -t|--type)
            TEST_TYPE="$2"
            shift 2
            ;;
        -c|--coverage)
            COVERAGE_REPORT="$2"
            shift 2
            ;;
        -v|--verbose)
            VERBOSE=true
            shift
            ;;
        -p|--parallel)
            PARALLEL=true
            shift
            ;;
        -b|--benchmark)
            BENCHMARK=true
            shift
            ;;
        -m|--markers)
            MARKERS="$2"
            shift 2
            ;;
        -h|--help)
            echo "Usage: $0 [OPTIONS]"
            echo "Options:"
            echo "  -t, --type TYPE        Test type: unit, integration, e2e, all (default: all)"
            echo "  -c, --coverage FORMAT  Coverage report format: term, term-missing, html, xml (default: term-missing)"
            echo "  -v, --verbose          Verbose output"
            echo "  -p, --parallel         Run tests in parallel"
            echo "  -b, --benchmark        Include benchmark tests"
            echo "  -m, --markers MARKERS  Run tests with specific markers"
            echo "  -h, --help             Show this help message"
            echo ""
            echo "Examples:"
            echo "  $0                     # Run all tests"
            echo "  $0 -t unit -v          # Run unit tests with verbose output"
            echo "  $0 -t integration -p   # Run integration tests in parallel"
            echo "  $0 -m \"not slow\"       # Run tests excluding slow tests"
            exit 0
            ;;
        *)
            print_error "Unknown option: $1"
            exit 1
            ;;
    esac
done

print_status "Running Chat Service Test Suite"
print_status "Test type: $TEST_TYPE"

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

# Set test environment variables
export ENVIRONMENT=testing
export TESTING=true
export MONGODB_URI=${TEST_MONGODB_URI:-"mongodb://localhost:27017/chatbot_test"}
export REDIS_URL=${TEST_REDIS_URL:-"redis://localhost:6379/1"}
export POSTGRESQL_URI=${TEST_POSTGRESQL_URI:-"postgresql://postgres:postgres@localhost:5432/chatbot_config_test"}

# Create logs directory for test runs
mkdir -p logs/tests

# Set up pytest command
PYTEST_CMD="pytest"

# Add coverage options
if [ "$TEST_TYPE" != "benchmark" ]; then
    PYTEST_CMD="$PYTEST_CMD --cov=src --cov-report=$COVERAGE_REPORT"
    
    # Add HTML coverage report for local development
    if [ "$COVERAGE_REPORT" = "html" ] || [ "$COVERAGE_REPORT" = "term-missing" ]; then
        PYTEST_CMD="$PYTEST_CMD --cov-report=html:htmlcov"
    fi
    
    # Add XML coverage for CI
    PYTEST_CMD="$PYTEST_CMD --cov-report=xml:coverage.xml"
fi

# Add verbose flag
if [ "$VERBOSE" = true ]; then
    PYTEST_CMD="$PYTEST_CMD -v"
fi

# Add parallel execution
if [ "$PARALLEL" = true ]; then
    # Use number of CPU cores
    num_cores=$(nproc 2>/dev/null || sysctl -n hw.ncpu 2>/dev/null || echo 4)
    PYTEST_CMD="$PYTEST_CMD -n $num_cores"
    print_status "Running tests in parallel with $num_cores workers"
fi

# Add markers
if [ -n "$MARKERS" ]; then
    PYTEST_CMD="$PYTEST_CMD -m \"$MARKERS\""
fi

# Add benchmark options
if [ "$BENCHMARK" = true ]; then
    PYTEST_CMD="$PYTEST_CMD --benchmark-only --benchmark-sort=mean"
fi

# Add test path based on type
case $TEST_TYPE in
    unit)
        TEST_PATH="tests/unit/"
        print_status "Running unit tests..."
        ;;
    integration)
        TEST_PATH="tests/integration/"
        print_status "Running integration tests..."
        # Start test dependencies if needed
        if command -v docker-compose &> /dev/null; then
            print_status "Starting test dependencies..."
            docker-compose -f docker/docker-compose.test.yml up -d
            sleep 10  # Wait for services to be ready
        fi
        ;;
    e2e)
        TEST_PATH="tests/e2e/"
        print_status "Running end-to-end tests..."
        # Start full test environment
        if command -v docker-compose &> /dev/null; then
            print_status "Starting test environment..."
            docker-compose -f docker/docker-compose.test.yml up -d
            sleep 30  # Wait longer for full environment
        fi
        ;;
    benchmark)
        TEST_PATH="tests/"
        PYTEST_CMD="$PYTEST_CMD --benchmark-only"
        print_status "Running benchmark tests..."
        ;;
    all)
        TEST_PATH="tests/"
        print_status "Running all tests..."
        # Start test dependencies
        if command -v docker-compose &> /dev/null; then
            print_status "Starting test dependencies..."
            docker-compose -f docker/docker-compose.test.yml up -d
            sleep 15
        fi
        ;;
    *)
        print_error "Invalid test type: $TEST_TYPE"
        print_status "Valid types: unit, integration, e2e, benchmark, all"
        exit 1
        ;;
esac

# Add JUnit XML output for CI
PYTEST_CMD="$PYTEST_CMD --junitxml=logs/tests/junit-$TEST_TYPE.xml"

# Add final test path
PYTEST_CMD="$PYTEST_CMD $TEST_PATH"

print_status "Executing: $PYTEST_CMD"

# Function to cleanup test environment
cleanup_test_env() {
    if [ "$TEST_TYPE" = "integration" ] || [ "$TEST_TYPE" = "e2e" ] || [ "$TEST_TYPE" = "all" ]; then
        if command -v docker-compose &> /dev/null; then
            print_status "Cleaning up test environment..."
            docker-compose -f docker/docker-compose.test.yml down -v --remove-orphans
        fi
    fi
}

# Set trap to cleanup on exit
trap cleanup_test_env EXIT

# Pre-test validation
print_status "Validating test environment..."

# Check if pytest is available
if ! command -v pytest &> /dev/null; then
    print_error "pytest not found. Install development dependencies: pip install -r requirements-dev.txt"
    exit 1
fi

# Check test configuration
if [ ! -f "pyproject.toml" ]; then
    print_error "pyproject.toml not found. Make sure you're in the project root"
    exit 1
fi

# Run pre-test checks for integration/e2e tests
if [ "$TEST_TYPE" = "integration" ] || [ "$TEST_TYPE" = "e2e" ] || [ "$TEST_TYPE" = "all" ]; then
    print_status "Waiting for test services to be ready..."
    
    # Wait for MongoDB
    timeout=30
    while ! mongosh "$MONGODB_URI" --eval "db.runCommand('ping')" --quiet &> /dev/null && [ $timeout -gt 0 ]; do
        sleep 1
        ((timeout--))
    done
    
    if [ $timeout -eq 0 ]; then
        print_warning "MongoDB not ready, some tests may fail"
    else
        print_success "MongoDB is ready"
    fi
    
    # Wait for Redis
    redis_host=$(echo $REDIS_URL | sed -n 's/redis:\/\/\([^:]*\).*/\1/p')
    redis_port=$(echo $REDIS_URL | sed -n 's/redis:\/\/[^:]*:\([0-9]*\).*/\1/p')
    redis_host=${redis_host:-"localhost"}
    redis_port=${redis_port:-"6379"}
    
    timeout=30
    while ! redis-cli -h "$redis_host" -p "$redis_port" ping &> /dev/null && [ $timeout -gt 0 ]; do
        sleep 1
        ((timeout--))
    done
    
    if [ $timeout -eq 0 ]; then
        print_warning "Redis not ready, some tests may fail"
    else
        print_success "Redis is ready"
    fi
fi

# Run the tests
print_status "Starting test execution..."
start_time=$(date +%s)

if eval $PYTEST_CMD; then
    end_time=$(date +%s)
    duration=$((end_time - start_time))
    print_success "All tests passed! ✅"
    print_status "Test execution time: ${duration}s"
    
    # Show coverage report location if HTML was generated
    if [ "$COVERAGE_REPORT" = "html" ] || [ "$COVERAGE_REPORT" = "term-missing" ]; then
        if [ -d "htmlcov" ]; then
            print_status "HTML coverage report available at: htmlcov/index.html"
        fi
    fi
    
    # Show benchmark results location if benchmarks were run
    if [ "$BENCHMARK" = true ]; then
        print_status "Benchmark results saved to .benchmarks/"
    fi
    
    exit 0
else
    end_time=$(date +%s)
    duration=$((end_time - start_time))
    print_error "Tests failed! ❌"
    print_status "Test execution time: ${duration}s"
    print_status "Check the output above for details"
    
    # Show useful debugging information
    print_status "Debugging information:"
    print_status "- Test logs: logs/tests/"
    print_status "- Coverage report: htmlcov/index.html (if generated)"
    print_status "- JUnit XML: logs/tests/junit-$TEST_TYPE.xml"
    
    exit 1
fi