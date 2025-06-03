#!/bin/bash
# Ultimate MCP Server - Environment Setup Assistant

set -e  # Exit on error

echo ""
echo "================================================================"
echo "       Ultimate MCP Server - Environment Setup Assistant"
echo "================================================================"
echo ""

# Check if Python 3.13+ is installed
if ! command -v python &> /dev/null; then
    echo "ERROR: Python not found in PATH."
    echo "Please install Python 3.13 or higher from https://www.python.org/downloads/"
    exit 1
fi

# Check Python version
PYTHON_VERSION=$(python -c "import sys; print(sys.version.split()[0])")
MAJOR=$(echo $PYTHON_VERSION | cut -d. -f1)
MINOR=$(echo $PYTHON_VERSION | cut -d. -f2)

if [ "$MAJOR" -lt 3 ] || ([ "$MAJOR" -eq 3 ] && [ "$MINOR" -lt 13 ]); then
    echo "ERROR: Python 3.13 or higher is required."
    echo "Current version: $PYTHON_VERSION"
    exit 1
fi

echo "Python $PYTHON_VERSION found."

# Check if virtual environment exists
if [ ! -d ".venv" ]; then
    echo "Creating virtual environment..."
    python -m venv .venv
    echo "Virtual environment created successfully."
else
    echo "Virtual environment already exists."
fi

# Activate the virtual environment and install dependencies
echo ""
echo "Activating virtual environment and installing dependencies..."
echo ""

# Use source to activate the environment
if [ -f ".venv/Scripts/activate" ]; then
    # Windows Git Bash
    source .venv/Scripts/activate
elif [ -f ".venv/bin/activate" ]; then
    # Linux/Mac
    source .venv/bin/activate
else
    echo "ERROR: Could not find activation script."
    exit 1
fi

# Check if activation was successful
if [ -z "$VIRTUAL_ENV" ]; then
    echo "ERROR: Failed to activate virtual environment."
    exit 1
fi

echo "Installing dependencies..."
pip install -e ".[test]"

echo ""
echo "Environment setup complete! Running verification..."
echo ""

# Run the verification script
python verify_environment.py
if [ $? -ne 0 ]; then
    echo "WARNING: Environment verification reported issues."
    echo "Please review the output above for recommended fixes."
else
    echo "Success! Your environment is ready for development."
fi

echo ""
echo "To activate this environment in a new terminal, run:"
if [ -f ".venv/Scripts/activate" ]; then
    echo "    source .venv/Scripts/activate"
else
    echo "    source .venv/bin/activate"
fi
echo ""

# Script ends with the environment activated 