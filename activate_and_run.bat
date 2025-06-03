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
    echo [OK] Found virtual environment at .venv
    
    REM Activate the virtual environment
    echo.
    echo Activating virtual environment...
    call .venv\Scripts\activate.bat
    
    REM Verify activation worked
    where python >nul 2>&1
    if %errorlevel% neq 0 (
        echo [ERROR] Failed to activate virtual environment
        echo Please check your virtual environment setup.
        exit /b 1
    )
    
    echo [OK] Virtual environment activated
    
) else (
    echo [ERROR] Virtual environment not found at .venv
    echo.
    echo Please create a virtual environment first:
    echo   python -m venv .venv
    echo   .venv\Scripts\activate.bat
    echo   pip install -e .
    echo.
    exit /b 1
)

REM Check environment using our validation tool
echo.
echo Checking environment status...
python -m ultimate_mcp_server env --check-only
if %errorlevel% neq 0 (
    echo.
    echo [ERROR] Environment validation failed!
    echo.
    echo Run this command for detailed diagnostics:
    echo   python -m ultimate_mcp_server env --verbose --suggest
    echo.
    exit /b 1
)

echo [OK] Environment validation passed

REM Parse command line arguments - default to production server settings if no arguments
set "ARGS="

if "%1"=="" (
    set "ARGS=run --load-all-tools --host 0.0.0.0 --port 8013"
    echo.
    echo No arguments provided, starting server with production settings...
    echo (load-all-tools, host 0.0.0.0, port 8013)
) else (
    REM Collect all arguments, starting with 'run' command
    set "ARGS=run"
    :parse_args
    if "%1"=="" goto done_parsing
    set "ARGS=%ARGS% %1"
    shift
    goto parse_args
    :done_parsing
)

REM Show what we're about to run
echo.
echo Starting Ultimate MCP Server with arguments: %ARGS%
echo.
echo ===================================================================

REM Run the Ultimate MCP Server directly (not through CLI)
python -m ultimate_mcp_server %ARGS%

REM Check exit code
if %errorlevel% neq 0 (
    echo.
    echo [ERROR] Server exited with error code %errorlevel%
    exit /b %errorlevel%
) else (
    echo.
    echo [OK] Server exited normally
    exit /b 0
) 