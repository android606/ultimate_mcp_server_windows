@echo off
echo Setting up Virtual Environment for Ultimate MCP Server...

:: Check for admin rights
>nul 2>&1 "%SYSTEMROOT%\system32\cacls.exe" "%SYSTEMROOT%\system32\config\system"
if '%errorlevel%' NEQ '0' (
    echo Administrator rights required. Attempting to elevate...
    goto UACPrompt
) else (
    goto GotAdmin
)

:UACPrompt
    echo Set UAC = CreateObject^("Shell.Application"^) > "%temp%\getadmin.vbs"
    echo UAC.ShellExecute "%~s0", "", "", "runas", 1 >> "%temp%\getadmin.vbs"
    "%temp%\getadmin.vbs"
    exit /B

:GotAdmin
    if exist "%temp%\getadmin.vbs" ( del "%temp%\getadmin.vbs" )
    pushd "%CD%"
    CD /D "%~dp0"

:: Check if Python is available
where python >nul 2>nul
if %ERRORLEVEL% neq 0 (
    echo Python not found in PATH. Please install Python 3.13+ and try again.
    exit /b 1
)

:: Clean existing .venv directory if it exists and is causing problems
if exist .venv (
    echo Found existing .venv directory.
    choice /C YN /M "Do you want to remove it and create a fresh environment"
    if %ERRORLEVEL% equ 1 (
        echo Removing existing virtual environment...
        rmdir /S /Q .venv
    )
)

:: Run the setup script
python setup_venv.py

if %ERRORLEVEL% neq 0 (
    :: If normal approach fails, try the direct approach
    echo Standard setup failed, trying alternative approach...
    python -m venv .venv --clear
    
    if %ERRORLEVEL% neq 0 (
        echo Virtual environment setup failed.
        echo Please try running this script as administrator.
        exit /b 1
    )
    
    :: Install packages manually
    echo Installing packages...
    call .venv\Scripts\activate.bat
    pip install -e .[test]
)

echo.
echo To activate the environment, run:
echo     .venv\Scripts\activate.bat

echo.
echo Then run the server with:
echo     python -m ultimate_mcp_server run --port 8014 --host 127.0.0.1 --debug

echo.
echo To run tests:
echo     python -m pytest tests/test_server_startup.py -v
echo     python -m pytest tests/environment/ -v

echo.
echo Virtual Environment setup complete! 