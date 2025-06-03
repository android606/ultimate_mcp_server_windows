#!/bin/bash
echo "Setting up Virtual Environment for Ultimate MCP Server..."

# Check if Python is available
if ! command -v python3 &> /dev/null; then
    echo "Python 3 not found. Please install Python 3.13+ and try again."
    exit 1
fi

# Check for existing .venv directory
if [ -d ".venv" ]; then
    echo "Found existing .venv directory."
    read -p "Do you want to remove it and create a fresh environment? [y/N] " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        echo "Removing existing virtual environment..."
        rm -rf .venv
    fi
fi

# Create venv
echo "Creating virtual environment..."
python3 -m venv .venv

if [ $? -ne 0 ]; then
    echo "Virtual environment setup failed."
    exit 1
fi

# Activate the environment
echo "Activating environment..."
source .venv/bin/activate

# Install the package in development mode
echo "Installing Ultimate MCP Server in development mode..."
pip install -U pip
pip install -e ".[test]"

if [ $? -ne 0 ]; then
    echo "Package installation failed."
    exit 1
fi

echo ""
echo "✨ Setup complete! ✨"
echo ""
echo "To activate the environment:"
echo "    source .venv/bin/activate"
echo ""
echo "Then run the server with:"
echo "    python -m ultimate_mcp_server run --port 8014 --host 127.0.0.1 --debug"
echo ""
echo "To run tests:"
echo "    python -m pytest tests/test_server_startup.py -v"
echo "    python -m pytest tests/environment/ -v" 