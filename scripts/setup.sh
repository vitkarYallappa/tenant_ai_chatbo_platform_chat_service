---

## **Scripts**

### `scripts/setup.sh`
```bash
#!/bin/bash
set -e

echo "🚀 Setting up Chat Service development environment..."

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
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

# Check if script is run from project root
if [ ! -f "pyproject.toml" ]; then
    print_error "Please run this script from the project root directory"
    exit 1
fi

# Check Python version
print_status "Checking Python version..."
python_version=$(python3 --version 2>&1 | cut -d' ' -f2)
required_version="3.11"

if ! python3 -c "import sys; exit(0 if sys.version_info >= (3, 11) else 1)" 2>/dev/null; then
    print_error "Python 3.11+ is required. Current version: $python_version"
    print_status "Please install Python 3.11 or higher and try again"
    exit 1
fi

print_success "Python version $python_version is compatible"

# Check if virtual environment already exists
if [ -d "venv" ]; then
    print_warning "Virtual environment already exists"
    read -p "Do you want to recreate it? (y/N): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        print_status "Removing existing virtual environment..."
        rm -rf venv
    else
        print_status "Using existing virtual environment..."
    fi
fi

# Create virtual environment if it doesn't exist
if [ ! -d "venv" ]; then
    print_status "Creating virtual environment..."
    python3 -m venv venv
    print_success "Virtual environment created"
fi

# Activate virtual environment
print_status "Activating virtual environment..."
source venv/bin/activate

# Upgrade pip
print_status "Upgrading pip..."
python -m pip install --upgrade pip

# Install build dependencies
print_status "Installing build dependencies..."
pip install wheel setuptools

# Install dependencies
print_status "Installing project dependencies..."
pip install -r requirements-dev.txt

# Install the package in development mode
print_status "Installing package in development mode..."
pip install -e .

# Install pre-commit hooks
print_status "Setting up pre-commit hooks..."
pre-commit install
pre-commit install --hook-type commit-msg
print_success "Pre-commit hooks installed"

# Create .env file if it doesn't exist
if [ ! -f .env ]; then
    print_status "Creating .env file from template..."
    cp .env.example .env
    print_warning "Please update .env file with your configuration"
    print_status "Example: MONGODB_URI, REDIS_URL, etc."
fi

# Create necessary directories
print_status "Creating necessary directories..."
mkdir -p logs temp uploads

# Set permissions for scripts
print_status "Setting script permissions..."
chmod +x scripts/*.sh

# Check Docker and Docker Compose
print_status "Checking Docker availability..."
if command -v docker &> /dev/null; then
    if command -v docker-compose &> /dev/null || docker compose version &> /dev/null; then
        print_success "Docker and Docker Compose are available"
        print_status "You can start dependencies with: docker-compose -f docker/docker-compose.dev.yml up -d"
    else
        print_warning "Docker Compose not found. Install it to run dependencies locally"
    fi
else
    print_warning "Docker not found. Install it to run dependencies locally"
fi

# Verify installation
print_status "Verifying installation..."
if python -c "import fastapi, motor, redis, pydantic" 2>/dev/null; then
    print_success "All major dependencies are installed correctly"
else
    print_error "Some dependencies failed to install. Check the output above"
    exit 1
fi

# Run a quick syntax check
print_status "Running syntax check..."
if python -m py_compile src/main.py; then
    print_success "Main module syntax is correct"
else
    print_error "Syntax errors found in main module"
    exit 1
fi

# Generate initial migration (if using Alembic in the future)
# print_status "Generating database migrations..."
# alembic init migrations
# alembic revision --autogenerate -m "Initial migration"

print_success "Setup complete! 🎉"
echo
echo "Next steps:"
echo "1. Activate the environment: source venv/bin/activate"
echo "2. Update configuration: edit .env file"
echo "3. Start dependencies: docker-compose -f docker/docker-compose.dev.yml up -d"
echo "4. Start the service: ./scripts/start.sh"
echo "5. Run tests: pytest"
echo
echo "For more information, see:"
echo "- docs/README.md for detailed documentation"
echo "- docs/CONTRIBUTING.md for development guidelines"
echo "- docs/API.md for API documentation"