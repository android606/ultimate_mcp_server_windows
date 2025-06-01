#!/bin/bash

# Get the directory where the script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

# Check if virtual environment exists
if [ ! -d ".venv" ]; then
    echo "Error: Virtual environment not found in .venv directory"
    read -p "Press Enter to exit..."
    exit 1
fi

# Activate the virtual environment
source .venv/Scripts/activate

# Check if activation was successful
if [ $? -ne 0 ]; then
    echo "Error: Failed to activate virtual environment"
    read -p "Press Enter to exit..."
    exit 1
fi

# Check if .env file exists
if [ ! -f ".env" ]; then
    echo "Error: .env file not found"
    read -p "Press Enter to exit..."
    exit 1
fi

echo "Starting MCP server in SSE mode for Cursor compatibility..."
python -m ultimate_mcp_server run --transport-mode sse

# Keep the window open if there's an error
if [ $? -ne 0 ]; then
    echo "Error: Server exited with an error"
    read -p "Press Enter to exit..."
    exit 1
fi 