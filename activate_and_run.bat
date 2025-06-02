@echo off
REM Ultimate MCP Server - Environment Activation and Server Startup Script (Windows)
REM This script ensures the correct virtual environment is activated before running the server

echo.
echo ===================================================================
echo  Ultimate MCP Server - Windows Environment Activation Script
echo ===================================================================
echo.

REM Save current directory
set "SCRIPT_DIR=%~dp0"
cd /d "%SCRIPT_DIR%"

REM Check if virtual environment exists
if exist ".venv\Scripts\activate.bat" (
    echo ✓ Found virtual environment at .venv
    
    REM Activate the virtual environment
    echo.
    echo Activating virtual environment...
    call .venv\Scripts\activate.bat
    
    REM Verify activation worked
    where python >nul 2>&1
    if %errorlevel% neq 0 (
        echo ❌ Failed to activate virtual environment
        echo Please check your virtual environment setup.
        pause
        exit /b 1
    )
    
    echo ✓ Virtual environment activated
    
) else (
    echo ❌ Virtual environment not found at .venv
    echo.
    echo Please create a virtual environment first:
    echo   python -m venv .venv
    echo   .venv\Scripts\activate.bat
    echo   pip install -e .
    echo.
    pause
    exit /b 1
)

REM Check environment using our validation tool
echo.
echo Checking environment status...
python -m ultimate_mcp_server.cli env --check-only
if %errorlevel% neq 0 (
    echo.
    echo ❌ Environment validation failed!
    echo.
    echo Run this command for detailed diagnostics:
    echo   python -m ultimate_mcp_server.cli env --verbose --suggest
    echo.
    pause
    exit /b 1
)

echo ✓ Environment validation passed

REM Parse command line arguments - default to run if no arguments
set "COMMAND=run"
set "ARGS="

if "%1"=="" (
    set "ARGS=--debug"
    echo.
    echo No arguments provided, starting server with default settings...
) else (
    REM Check if first argument is a known command
    if "%1"=="run" set "COMMAND=%1" & shift
    if "%1"=="env" set "COMMAND=%1" & shift
    if "%1"=="providers" set "COMMAND=%1" & shift
    if "%1"=="test" set "COMMAND=%1" & shift
    if "%1"=="tools" set "COMMAND=%1" & shift
    if "%1"=="examples" set "COMMAND=%1" & shift
    
    REM Collect remaining arguments
    :parse_args
    if "%1"=="" goto done_parsing
    set "ARGS=%ARGS% %1"
    shift
    goto parse_args
    :done_parsing
)

REM Show what we're about to run
echo.
echo Starting Ultimate MCP Server with command: %COMMAND% %ARGS%
echo.
echo ===================================================================

REM Run the Ultimate MCP Server
python -m ultimate_mcp_server.cli %COMMAND% %ARGS%

REM Check exit code
if %errorlevel% neq 0 (
    echo.
    echo ❌ Server exited with error code %errorlevel%
    pause
) else (
    echo.
    echo ✓ Server exited normally
)

echo.
echo Press any key to exit...
pause >nul 