@echo off
setlocal enabledelayedexpansion

echo.
echo ================================================================
echo        Ultimate MCP Server - Environment Setup Assistant
echo ================================================================
echo.

REM Check if Python 3.13+ is installed
where python >nul 2>nul
if %ERRORLEVEL% neq 0 (
    echo ERROR: Python not found in PATH.
    echo Please install Python 3.13 or higher from https://www.python.org/downloads/
    echo Ensure "Add Python to PATH" is checked during installation.
    exit /b 1
)

REM Check Python version
for /f "tokens=2" %%i in ('python -c "import sys; print(sys.version.split()[0])"') do set PYTHON_VERSION=%%i
for /f "tokens=1,2 delims=." %%a in ("%PYTHON_VERSION%") do (
    set MAJOR=%%a
    set MINOR=%%b
)

if %MAJOR% LSS 3 (
    echo ERROR: Python 3.13 or higher is required.
    echo Current version: %PYTHON_VERSION%
    exit /b 1
)

if %MAJOR% EQU 3 (
    if %MINOR% LSS 13 (
        echo ERROR: Python 3.13 or higher is required.
        echo Current version: %PYTHON_VERSION%
        exit /b 1
    )
)

echo Python %PYTHON_VERSION% found.

REM Check if virtual environment exists
if not exist .venv\ (
    echo Creating virtual environment...
    python -m venv .venv
    if %ERRORLEVEL% neq 0 (
        echo ERROR: Failed to create virtual environment.
        exit /b 1
    )
    echo Virtual environment created successfully.
) else (
    echo Virtual environment already exists.
)

REM Activate the virtual environment and install dependencies
echo.
echo Activating virtual environment and installing dependencies...
echo.

REM Use call to ensure the batch file continues after running activate
call .venv\Scripts\activate.bat

REM Check if activation was successful
if not defined VIRTUAL_ENV (
    echo ERROR: Failed to activate virtual environment.
    exit /b 1
)

echo Installing dependencies...
pip install -e ".[test]"
if %ERRORLEVEL% neq 0 (
    echo ERROR: Failed to install dependencies.
    exit /b 1
)

echo.
echo Environment setup complete! Running verification...
echo.

REM Run the verification script
python verify_environment.py
if %ERRORLEVEL% neq 0 (
    echo WARNING: Environment verification reported issues.
    echo Please review the output above for recommended fixes.
) else (
    echo Success! Your environment is ready for development.
)

echo.
echo To activate this environment in a new terminal, run:
echo    .venv\Scripts\activate.bat
echo.

REM Keep the environment activated
endlocal & (
    call .venv\Scripts\activate.bat
) 