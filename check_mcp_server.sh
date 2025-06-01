#!/bin/bash

# Try to get server info
response=$(curl -s http://localhost:8013/)

if [[ $response == *"mcp-server"* ]]; then
    echo "MCP server is running"
    exit 0
else
    echo "MCP server is not running"
    echo "Please check if the server started with your Windows login"
    echo "You can manually start it by running: ./start_mcp_server.bat"
    exit 1
fi 