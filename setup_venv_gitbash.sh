#!/bin/bash
echo "Setting up Virtual Environment for Ultimate MCP Server (Git Bash version)..."

# Check if Python is available
if ! command -v python &> /dev/null; then
    echo "Python not found. Please install Python 3.13+ and try again."
    exit 1
fi

# Check if we're running in Git Bash on Windows
if [[ "$(uname -s)" == MINGW* ]] || [[ "$(uname -s)" == MSYS* ]]; then
    echo "Detected Git Bash on Windows"
    WINDOWS=true
else
    WINDOWS=false
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

# Create venv directly - sometimes more reliable on Git Bash
echo "Creating virtual environment..."
python -m venv .venv --clear

if [ $? -ne 0 ]; then
    echo "Standard venv creation failed, trying alternative approach..."
    # Using system command to ensure proper permissions
    if [ "$WINDOWS" = true ]; then
        cmd.exe /c "python -m venv .venv --clear"
    else
        python3 -m venv .venv --clear
    fi
    
    if [ $? -ne 0 ]; then
        echo "Virtual environment setup failed."
        echo "Please try running Windows Command Prompt as administrator and run setup_venv_windows.bat"
        exit 1
    fi
fi

# Activate the environment
if [ "$WINDOWS" = true ]; then
    echo "Activating environment for Windows..."
    # Git Bash doesn't handle Windows batch files well, so use PowerShell
    VENV_PYTHON=".venv/Scripts/python.exe"
    VENV_PIP=".venv/Scripts/pip.exe"
else
    echo "Activating environment for Unix..."
    source .venv/bin/activate
    VENV_PYTHON=".venv/bin/python"
    VENV_PIP=".venv/bin/pip"
fi

# Install the package in development mode
echo "Installing Ultimate MCP Server in development mode..."
"$VENV_PYTHON" -m pip install -U pip
"$VENV_PYTHON" -m pip install -e ".[test]"

if [ $? -ne 0 ]; then
    echo "Package installation failed."
    exit 1
fi

# Customize the activation scripts
echo "Customizing activation scripts for better Windows compatibility..."
# Attempt to modify scripts directly if custom_venv didn't handle it
if [ "$WINDOWS" = true ] && [ -f ".venv/Scripts/activate.bat" ]; then
    # Add HOME setting to batch activation if not already present
    if ! grep -q "REM Ultimate MCP Server customization" ".venv/Scripts/activate.bat"; then
        echo '
REM Ultimate MCP Server customization
set "PATH=%VIRTUAL_ENV%\\Scripts;%PATH%"
REM Ensure HOME is set for Git
if not defined HOME (
    if defined USERPROFILE (
        set "HOME=%USERPROFILE%"
    ) else (
        if defined HOMEDRIVE (
            if defined HOMEPATH (
                set "HOME=%HOMEDRIVE%%HOMEPATH%"
            )
        )
    )
)' >> ".venv/Scripts/activate.bat"
    fi
fi

echo ""
echo "✨ Setup complete! ✨"
echo ""
echo "To activate the environment in Git Bash:"
if [ "$WINDOWS" = true ]; then
    echo "    source .venv/Scripts/activate"
    # Create a custom Git Bash activation script if it doesn't exist
    if [ ! -f ".venv/Scripts/activate.sh" ]; then
        echo "Creating Git Bash friendly activation script..."
        echo '#!/bin/bash
export VIRTUAL_ENV=$(cygpath "$(dirname "$(dirname "${BASH_SOURCE[0]}")")")
export PATH="$(cygpath -u "${VIRTUAL_ENV}/Scripts"):$PATH"
export UMCP_SKIP_ENV_CHECK=1
# Ensure HOME is set for Git
if [ -z "$HOME" ]; then
    if [ -n "$USERPROFILE" ]; then
        export HOME=$(cygpath "$USERPROFILE")
    elif [ -n "$HOMEDRIVE" ] && [ -n "$HOMEPATH" ]; then
        export HOME=$(cygpath "$HOMEDRIVE$HOMEPATH")
    fi
fi
' > ".venv/Scripts/activate.sh"
        chmod +x ".venv/Scripts/activate.sh"
        echo "    source .venv/Scripts/activate.sh  (alternative for Git Bash)"
    fi
else
    echo "    source .venv/bin/activate"
fi

echo ""
echo "Then run the server with:"
echo "    python -m ultimate_mcp_server run --port 8014 --host 127.0.0.1 --debug"
echo ""
echo "To run tests:"
echo "    python -m pytest tests/test_server_startup.py -v" 
echo "    python -m pytest tests/environment/ -v" 