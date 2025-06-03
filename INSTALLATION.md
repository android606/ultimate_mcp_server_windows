# 📦 Ultimate MCP Server Installation Guide

> **🙏 Massive Credit and Thanks to [Dicklesworthstone](https://github.com/Dicklesworthstone)** for creating the original [Ultimate MCP Server](https://github.com/Dicklesworthstone/ultimate_mcp_server)! This Windows-optimized fork builds upon their incredible work to provide enhanced compatibility and features for Windows environments. All the core innovation, architecture, and functionality comes from their original project. Please visit and star the original repository to show your appreciation for their amazing contributions to the MCP ecosystem.

## Prerequisites

- **Python**: 3.13+ recommended
- **Git**: For cloning the repository
- **Operating System**: Windows 10/11 (primary focus), macOS/Linux (supported)
- **RAM**: 4GB minimum, 8GB+ recommended
- **Storage**: 2GB for dependencies and caching

## Quick Installation Options

### Option 1: Using Setup Scripts (Recommended)

Choose the appropriate script for your shell:

```bash
# Windows Command Prompt
setup_venv_windows.bat

# PowerShell
.\setup_venv_powershell.ps1

# Git Bash on Windows
./setup_venv_gitbash.sh

# Unix/Linux/macOS
./setup_venv.sh
```

These scripts will:
- Create a Python virtual environment in `.venv`
- Install all dependencies
- Configure environment variables correctly for your platform
- Handle Windows-specific PATH and Git compatibility issues

### Option 2: Using Activation Scripts

```bash
# Clone the Windows-optimized repository
git clone https://github.com/android606/ultimate_mcp_server_windows.git
cd ultimate_mcp_server_windows

# Create virtual environment (first time only)
python -m venv .venv

# Windows: Use the activation script
activate_and_run.bat

# Linux/Mac: Use the shell script
chmod +x activate_and_run.sh
./activate_and_run.sh
```

### Option 3: Manual Installation

```bash
# Clone the repository
git clone https://github.com/android606/ultimate_mcp_server_windows.git
cd ultimate_mcp_server_windows

# Create and activate virtual environment
python -m venv .venv

# Activate virtual environment
# Windows:
.venv\Scripts\activate.bat
# PowerShell:
.venv\Scripts\Activate.ps1
# Linux/Mac or Git Bash:
source .venv/bin/activate

# Install in development mode
pip install -e .

# Verify environment
python -m ultimate_mcp_server.cli env --verbose

# Start the server
python -m ultimate_mcp_server run --debug
```

### Option 4: Using uv (Ultra-fast Package Manager)

```bash
# Install uv if needed
# Windows:
powershell -c "irm https://astral.sh/uv/install.ps1 | iex"
# Linux/Mac:
curl -LsSf https://astral.sh/uv/install.sh | sh

# Clone and install
git clone https://github.com/android606/ultimate_mcp_server_windows.git
cd ultimate_mcp_server_windows
uv sync

# Run the server
uv run python -m ultimate_mcp_server run --debug
```

## Configuration

### API Keys and Environment Setup

1. Copy the example configuration files:
   ```bash
   cp .env.example .env
   cp .env.secrets.example .env.secrets
   ```

2. Edit `.env.secrets` to add your API keys:
   ```
   OPENAI_API_KEY=your_openai_key_here
   ANTHROPIC_API_KEY=your_anthropic_key_here
   GOOGLE_API_KEY=your_google_key_here
   DEEPSEEK_API_KEY=your_deepseek_key_here
   OPENROUTER_API_KEY=your_openrouter_key_here
   TOGETHERAI_API_KEY=your_togetherai_key_here
   ```

3. (Optional) Adjust settings in `.env` for:
   - Server configuration (port, host, workers)
   - Logging and cache configuration
   - Model preferences
   - File system access paths

### Security Best Practices

- Never commit `.env.secrets` to version control
- Keep your API keys secure and rotate them regularly
- In production, use environment variables instead of files
- Adjust `FILESYSTEM__ALLOWED_DIRECTORIES` to limit directory access

## Running the Server

### Basic Server Startup

```bash
# Default startup
python -m ultimate_mcp_server run

# With all tools loaded
python -m ultimate_mcp_server run --load-all-tools

# Debug mode with custom port
python -m ultimate_mcp_server run --port 8014 --host 127.0.0.1 --debug
```

### Advanced Options

```bash
# Load specific tools
python -m ultimate_mcp_server run --include-tools browser,audio

# Exclude specific tools
python -m ultimate_mcp_server run --load-all-tools --exclude-tools filesystem

# Server accessible from network
python -m ultimate_mcp_server run --host 0.0.0.0 --port 8013

# Multiple workers
python -m ultimate_mcp_server run --workers 4
```

### Running as a Windows Scheduled Task

A template for Windows Task Scheduler is provided (`ultimate_mcp_scheduler_task.xml`).
Edit the placeholders, then import into Task Scheduler.

## Environment Validation

The server performs environment validation on startup:

```bash
# Check environment status
python -m ultimate_mcp_server env --verbose

# Get setup suggestions
python -m ultimate_mcp_server env --suggest

# Skip validation (not recommended)
python -m ultimate_mcp_server run --skip-env-check
```

## Running Tests

```bash
# Run server integration tests
python -m pytest tests/test_server_startup.py -v

# Run environment verification tests
python -m pytest tests/environment/ -v

# Run all tests
python -m pytest
```

## Troubleshooting

### Permission Issues
- Run setup scripts with administrator privileges
- On Windows, ensure write access to installation directory

### Python Not Found
- Verify Python is in your PATH
- Windows: Check that "Add to PATH" was selected during installation

### Git Issues
- Ensure `HOME` environment variable is properly set
- Windows CMD: `set HOME=%USERPROFILE%`
- PowerShell: `$env:HOME = $env:USERPROFILE`
- Git Bash: `export HOME=$(cygpath "$USERPROFILE")`

### Missing Packages
   ```bash
pip install pytest-asyncio pytest-cov pytest-mock anyio
   ```

### Environment Problems
   ```bash
   # Remove old environment
   rmdir /s .venv  # Windows
   rm -rf .venv    # Linux/Mac
   
   # Create new environment
   python -m venv .venv
   .venv\Scripts\activate.bat  # Windows
   source .venv/bin/activate   # Linux/Mac
   
   # Reinstall
   pip install -e .
   ```

## Next Steps

After successful installation:

1. **🔑 Configure API Keys**: Add your LLM provider keys to `.env` file
2. **🧪 Test Basic Functionality**: Run `python -m ultimate_mcp_server.cli test openai`
3. **🔍 Explore Tools**: Check `python -m ultimate_mcp_server.cli tools`
4. **📚 Run Examples**: Explore the `examples/` directory
5. **📖 Read Documentation**: Review the main README.md and original project documentation
6. **🚀 Start Building**: Begin integrating the server with your AI applications

## Contributing

This Windows-optimized fork welcomes contributions, especially for:
- Windows compatibility improvements
- Environment validation enhancements
- Installation automation
- Testing framework improvements

Please see CONTRIBUTING.md for guidelines, and don't forget to contribute back to [Dicklesworthstone's original project](https://github.com/Dicklesworthstone/ultimate_mcp_server) when possible!

---

> **Again, huge thanks to [Dicklesworthstone](https://github.com/Dicklesworthstone)** for creating this incredible MCP server platform. This fork simply adds Windows-specific optimizations and environment validation to make the installation and setup process smoother for Windows users. All the core functionality, innovation, and hard work comes from the original project! 