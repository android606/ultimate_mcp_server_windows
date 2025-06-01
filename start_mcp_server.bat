@echo off
echo Starting Ultimate MCP Server in SSE mode for Cursor compatibility...
cd /d "%~dp0"

REM Check if Git Bash exists
if not exist "C:\Program Files\Git\bin\bash.exe" (
    echo Error: Git Bash not found at C:\Program Files\Git\bin\bash.exe
    echo Please install Git for Windows
    pause
    exit /b 1
)

REM Start the server using Git Bash with SSE mode
"C:\Program Files\Git\bin\bash.exe" -c "./start_mcp_server.sh"
if errorlevel 1 (
    echo Error: Failed to start MCP server
    pause
    exit /b 1
) 