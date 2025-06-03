@echo off
REM Ultimate MCP Server - Task Scheduler Startup Script (Windows)
REM This script is designed to run from Windows Task Scheduler without user interaction

setlocal enabledelayedexpansion

REM Set up logging
set "LOG_DIR=%~dp0logs"
set "TIMESTAMP=%date:~-4,4%%date:~-10,2%%date:~-7,2%_%time:~0,2%%time:~3,2%%time:~6,2%"
set "TIMESTAMP=%TIMESTAMP: =0%"
set "LOG_FILE=%LOG_DIR%\server_startup_%TIMESTAMP%.log"
set "LATEST_LOG=%LOG_DIR%\latest.log"

REM Create logs directory if it doesn't exist
if not exist "%LOG_DIR%" mkdir "%LOG_DIR%"

REM Redirect all output to log file
(
echo.
echo ===================================================================
echo  Ultimate MCP Server - Task Scheduler Startup Script
echo  Started at: %date% %time%
echo ===================================================================
echo.

REM Save current directory and change to script directory
set "SCRIPT_DIR=%~dp0"
cd /d "%SCRIPT_DIR%"
echo Current directory: %CD%

REM Check if virtual environment exists
if exist ".venv\Scripts\activate.bat" (
    echo [OK] Found virtual environment at .venv
    
    REM Activate the virtual environment
    echo.
    echo Activating virtual environment...
    call .venv\Scripts\activate.bat
    
    REM Verify activation worked
    where python >nul 2>&1
    if !errorlevel! neq 0 (
        echo [ERROR] Failed to activate virtual environment
        echo Virtual environment activation failed - exiting
        exit /b 1
    )
    
    echo [OK] Virtual environment activated
    
) else (
    echo [ERROR] Virtual environment not found at .venv
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
if !errorlevel! neq 0 (
    echo.
    echo [ERROR] Environment validation failed!
    echo Run this command for detailed diagnostics:
    echo   python -m ultimate_mcp_server env --verbose --suggest
    echo.
    exit /b 1
)

echo [OK] Environment validation passed

REM Parse command line arguments or use production defaults
set "ARGS=run --load-all-tools --host 0.0.0.0 --port 8013"

REM Override if arguments provided
if not "%1"=="" (
    set "ARGS=run"
    :parse_args
    if not "%1"=="" (
        set "ARGS=!ARGS! %1"
        shift
        goto parse_args
    )
)

REM Show what we're about to run
echo.
echo Starting Ultimate MCP Server with arguments: %ARGS%
echo.
echo ===================================================================

REM Run the Ultimate MCP Server directly (not through CLI)
echo Server starting at: %date% %time%
python -m ultimate_mcp_server %ARGS%

REM Check exit code
set "EXIT_CODE=!errorlevel!"
if !EXIT_CODE! neq 0 (
    echo.
    echo [ERROR] Server exited with error code !EXIT_CODE! at %date% %time%
) else (
    echo.
    echo [OK] Server exited normally at %date% %time%
)

echo.
echo ===================================================================
echo Script completed at: %date% %time%
echo Exit code: !EXIT_CODE!
echo ===================================================================

exit /b !EXIT_CODE!

) >> "%LOG_FILE%" 2>&1

REM Also log to a latest.log for easy access
copy "%LOG_FILE%" "%LATEST_LOG%" >nul 2>&1 