#!/bin/bash

# Ultimate MCP Server - Environment Activation and Server Startup Script (Linux/Mac)
# This script ensures the correct virtual environment is activated before running the server

set -e  # Exit on any error

echo
echo "=================================================================="
echo " Ultimate MCP Server - Unix Environment Activation Script"
echo "=================================================================="
echo

# Get the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Function to print colored output
print_status() {
    echo "✓ $1"
}

print_error() {
    echo "❌ $1"
}

print_warning() {
    echo "⚠️  $1"
}

# Check if virtual environment exists
if [ -f ".venv/bin/activate" ]; then
    print_status "Found virtual environment at .venv"
    
    # Activate the virtual environment
    echo
    echo "Activating virtual environment..."
    source .venv/bin/activate
    
    # Verify activation worked
    if ! command -v python >/dev/null 2>&1; then
        print_error "Failed to activate virtual environment"
        echo "Please check your virtual environment setup."
        exit 1
    fi
    
    print_status "Virtual environment activated"
    
elif [ -f "venv/bin/activate" ]; then
    print_status "Found virtual environment at venv"
    
    # Activate the virtual environment
    echo
    echo "Activating virtual environment..."
    source venv/bin/activate
    
    # Verify activation worked
    if ! command -v python >/dev/null 2>&1; then
        print_error "Failed to activate virtual environment"
        echo "Please check your virtual environment setup."
        exit 1
    fi
    
    print_status "Virtual environment activated"
    
else
    print_error "Virtual environment not found"
    echo
    echo "Please create a virtual environment first:"
    echo "  python -m venv .venv"
    echo "  source .venv/bin/activate"
    echo "  pip install -e ."
    echo
    exit 1
fi

# Check environment using our validation tool
echo
echo "Checking environment status..."
if ! python -m ultimate_mcp_server.cli env --check-only; then
    echo
    print_error "Environment validation failed!"
    echo
    echo "Run this command for detailed diagnostics:"
    echo "  python -m ultimate_mcp_server.cli env --verbose --suggest"
    echo
    exit 1
fi

print_status "Environment validation passed"

# Parse command line arguments - default to run if no arguments
COMMAND="run"
ARGS=""

if [ $# -eq 0 ]; then
    ARGS="--debug"
    echo
    echo "No arguments provided, starting server with default settings..."
else
    # Check if first argument is a known command
    case "$1" in
        run|env|providers|test|tools|examples)
            COMMAND="$1"
            shift
            ;;
    esac
    
    # Collect remaining arguments
    ARGS="$*"
fi

# Show what we're about to run
echo
echo "Starting Ultimate MCP Server with command: $COMMAND $ARGS"
echo
echo "=================================================================="

# Function to handle cleanup on script exit
cleanup() {
    echo
    echo "Script exiting..."
}

# Set up trap for cleanup
trap cleanup EXIT

# Run the Ultimate MCP Server
if python -m ultimate_mcp_server.cli $COMMAND $ARGS; then
    echo
    print_status "Server exited normally"
else
    EXIT_CODE=$?
    echo
    print_error "Server exited with error code $EXIT_CODE"
    exit $EXIT_CODE
fi 