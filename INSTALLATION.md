# 📦 Ultimate MCP Server Installation Guide

> **🙏 Massive Credit and Thanks to [Dicklesworthstone](https://github.com/Dicklesworthstone)** for creating the original [Ultimate MCP Server](https://github.com/Dicklesworthstone/ultimate_mcp_server)! This Windows-optimized fork builds upon their incredible work to provide enhanced compatibility and features for Windows environments. All the core innovation, architecture, and functionality comes from their original project. Please visit and star the original repository to show your appreciation for their amazing contributions to the MCP ecosystem.

## Quick Installation from GitHub

**Important**: This installation method uses the Windows-optimized fork that includes additional environment validation, enhanced compatibility, and Windows-specific improvements.

### Prerequisites

- **Python**: 3.8 or higher (3.13+ recommended)
- **Git**: For cloning the repository
- **Operating System**: Windows 10/11 (primary focus), macOS, Linux (community supported)
- **RAM**: 4GB minimum, 8GB+ recommended for full functionality
- **Storage**: 2GB for dependencies and caching

### Installation Methods

#### Method 1: Quick Setup with Activation Scripts (Recommended)

This method uses the included activation scripts that automatically handle virtual environment setup and validation.

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

The activation scripts will:
- ✅ Verify virtual environment is activated
- ✅ Validate all required dependencies are installed
- ✅ Check Python version compatibility
- ✅ Start the server with appropriate settings
- ✅ Provide helpful error messages and suggestions

#### Method 2: Manual Installation

```bash
# Clone the repository
git clone https://github.com/android606/ultimate_mcp_server_windows.git
cd ultimate_mcp_server_windows

# Create and activate virtual environment
python -m venv .venv

# Activate virtual environment
# Windows:
.venv\Scripts\activate.bat
# Linux/Mac:
source .venv/bin/activate

# Install the package in development mode
pip install -e .

# Verify environment is properly configured
python -m ultimate_mcp_server.cli env --verbose

# Start the server
python -m ultimate_mcp_server.cli run --debug
```

#### Method 3: Using uv (Ultra-fast Package Manager)

```bash
# Install uv if you don't have it
# Windows:
powershell -c "irm https://astral.sh/uv/install.ps1 | iex"
# Linux/Mac:
curl -LsSf https://astral.sh/uv/install.sh | sh

# Clone and install
git clone https://github.com/android606/ultimate_mcp_server_windows.git
cd ultimate_mcp_server_windows

# Create virtual environment and install dependencies
uv sync

# Run the server
uv run python -m ultimate_mcp_server.cli run --debug
```

## Environment Validation and Setup

This Windows-optimized fork includes comprehensive environment validation to ensure your installation works correctly.

### Built-in Environment Checking

The installation includes several tools to help you verify and troubleshoot your setup:

```bash
# Check environment status
python -m ultimate_mcp_server.cli env

# Detailed environment information
python -m ultimate_mcp_server.cli env --verbose

# Get setup suggestions if issues are found
python -m ultimate_mcp_server.cli env --suggest

# Quick validation (exit code 0 = success, 1 = issues)
python -m ultimate_mcp_server.cli env --check-only

# Require virtual environment (strict mode)
python -m ultimate_mcp_server.cli env --strict
```

### Environment Validation Features

The validation system checks:

- ✅ **Virtual Environment**: Detects if running in venv, virtualenv, or conda
- ✅ **Python Version**: Ensures Python 3.8+ compatibility
- ✅ **Required Packages**: Validates all necessary dependencies are installed
- ✅ **Package Versions**: Checks for compatible package versions
- ✅ **Environment Path**: Locates project virtual environment if not activated

### Automatic Environment Validation

By default, the server performs environment validation on startup. You can control this behavior:

```bash
# Normal startup with environment checking (default)
python -m ultimate_mcp_server.cli run

# Skip environment validation (not recommended)
python -m ultimate_mcp_server.cli run --skip-env-check

# Strict mode - require virtual environment
python -m ultimate_mcp_server.cli env --strict && python -m ultimate_mcp_server.cli run
```

### Troubleshooting Environment Issues

If you encounter environment problems:

1. **Check Environment Status**:
   ```bash
   python -m ultimate_mcp_server.cli env --verbose
   ```

2. **Get Setup Suggestions**:
   ```bash
   python -m ultimate_mcp_server.cli env --suggest
   ```

3. **Recreate Virtual Environment**:
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

4. **Verify Installation**:
   ```bash
   python -c "import ultimate_mcp_server; print('✅ Import successful')"
   ```

## Configuration

### Environment Variables

Create a `.env` file in your project directory for API keys and configuration:

```bash
# LLM Provider API Keys (add the ones you plan to use)
OPENAI_API_KEY=your_openai_key_here
ANTHROPIC_API_KEY=your_anthropic_key_here
GOOGLE_API_KEY=your_google_key_here
DEEPSEEK_API_KEY=your_deepseek_key_here
XAI_API_KEY=your_xai_key_here

# Server Configuration (optional)
MCP_HOST=127.0.0.1
MCP_PORT=8013
MCP_LOG_LEVEL=INFO

# Tool-specific Configuration (optional)
ALLOWED_DIRS=/path/to/safe/directories  # For filesystem tools
```

### Basic Configuration

```bash
# Check available providers
python -m ultimate_mcp_server.cli providers

# Test a provider connection
python -m ultimate_mcp_server.cli test openai --prompt "Hello, world!"

# Check available tools
python -m ultimate_mcp_server.cli tools

# Run with specific settings
python -m ultimate_mcp_server.cli run --host 0.0.0.0 --port 8013 --debug
```

## Running the Server

### Basic Server Startup

```bash
# Default startup (Base Toolset only)
python -m ultimate_mcp_server.cli run

# With all tools loaded
python -m ultimate_mcp_server.cli run --load-all-tools

# Debug mode for development
python -m ultimate_mcp_server.cli run --debug

# Custom port to avoid conflicts
python -m ultimate_mcp_server.cli run --port 8014
```

### Advanced Server Options

```bash
# Load specific additional tools
python -m ultimate_mcp_server.cli run --include-tools browser,audio

# Load all tools except specific ones
python -m ultimate_mcp_server.cli run --load-all-tools --exclude-tools filesystem

# Server accessible from network
python -m ultimate_mcp_server.cli run --host 0.0.0.0 --port 8013

# Multiple workers for high concurrency
python -m ultimate_mcp_server.cli run --workers 4
```

### Using the Activation Scripts

The included activation scripts make server management easier:

```bash
# Windows - Simple server startup
activate_and_run.bat

# Windows - Pass specific arguments
activate_and_run.bat run --load-all-tools --debug

# Linux/Mac - Simple server startup
./activate_and_run.sh

# Linux/Mac - Pass specific arguments
./activate_and_run.sh run --port 8014 --debug
```

## Verification and Testing

### Basic Functionality Tests

```bash
# Check installation and version
python -m ultimate_mcp_server.cli --version

# Test environment
python -m ultimate_mcp_server.cli env --check-only

# List available tools
python -m ultimate_mcp_server.cli tools

# Test LLM provider connectivity
python -m ultimate_mcp_server.cli test openai --prompt "What is 2+2?"

# Generate a completion directly
python -m ultimate_mcp_server.cli complete --provider openai --prompt "Hello, world!"
```

### Server Connectivity Tests

```bash
# Start server in background
python -m ultimate_mcp_server.cli run --port 8013 &

# Test SSE endpoint (correct endpoint)
curl http://localhost:8013/sse

# Test with alternative port
curl http://localhost:8014/sse
```

### Advanced Testing

```bash
# Run the included test suite
python -m pytest tests/ -v

# Test specific tool status
python test_get_all_tools_status.py

# Run environment validation tests
python test_environment_validation.py

# Test MCP connectivity
python smart_mcp_test.py
```

## Server Endpoints & MCP Integration

### SSE (Server-Sent Events) Endpoint

The Ultimate MCP Server uses **Server-Sent Events (SSE)** for MCP protocol communication:

- **Correct endpoint**: `http://host:port/sse`
- **Incorrect**: `http://host:port` (will return 404)

### MCP Client Integration

For AI agents and external applications:

```python
# Example MCP client connection
import asyncio
from mcp import ClientSession
from mcp.client.sse import sse_client

async def connect_to_server():
    async with sse_client("http://localhost:8013/sse") as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            
            # List available tools
            tools = await session.list_tools()
            print(f"Available tools: {[tool.name for tool in tools.tools]}")
            
            # Check tool status
            result = await session.call_tool("get_all_tools_status", {})
            print(f"Tool status: {result}")

# Run the client
asyncio.run(connect_to_server())
```

### Configuration for External Clients

When connecting external MCP clients (like Cursor AI):

1. **Use the correct SSE endpoint**: Always append `/sse` to your server URL
2. **Configure environment variables** for service discovery:
   ```bash
   MCP_SERVER_HOST=127.0.0.1
   MCP_SERVER_PORT=8013
   MCP_SERVER_URL=http://127.0.0.1:8013/sse
   ```

## Port Management and Conflicts

### Avoiding Port Conflicts

The installation includes port management to prevent conflicts with existing services:

```bash
# Production server typically runs on port 8013
# Use different ports for testing and development

# Development server
python -m ultimate_mcp_server.cli run --port 8014

# Testing
python -m ultimate_mcp_server.cli run --port 8015

# Alternative production port
python -m ultimate_mcp_server.cli run --port 8016
```

### Port Allocation Reference

| Service Type | Port | Purpose |
|--------------|------|---------|
| Production | 8013 | Default production server |
| Development | 8014 | Development and testing |
| Testing | 8015-8030 | Automated tests and CI |
| User Testing | 8031+ | Manual testing by users |

## Troubleshooting

### Common Installation Issues

1. **Python Version Issues**:
   ```bash
   python --version  # Should be 3.8+
   python -m ultimate_mcp_server.cli env --verbose
   ```

2. **Virtual Environment Issues**:
   ```bash
   # Check if in virtual environment
   python -m ultimate_mcp_server.cli env
   
   # Create new virtual environment
   python -m venv .venv
   .venv\Scripts\activate.bat  # Windows
   source .venv/bin/activate   # Linux/Mac
   ```

3. **Missing Dependencies**:
   ```bash
   # Reinstall all dependencies
   pip install -e .
   
   # Check what's missing
   python -m ultimate_mcp_server.cli env --suggest
   ```

4. **Import Errors**:
   ```bash
   # Test basic import
   python -c "import ultimate_mcp_server; print('Success')"
   
   # Check Python path
   python -c "import sys; print('\n'.join(sys.path))"
   ```

5. **Port Conflicts**:
   ```bash
   # Check what's using port 8013
   netstat -an | find "8013"  # Windows
   lsof -i :8013              # Linux/Mac
   
   # Use different port
   python -m ultimate_mcp_server.cli run --port 8014
   ```

### Windows-Specific Issues

1. **Long Path Support**: Enable long path support in Windows if you encounter path-related errors
2. **Antivirus Software**: Add exclusions for the virtual environment directory
3. **PowerShell Execution Policy**: You may need to enable script execution:
   ```powershell
   Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
   ```

### Getting Help

- **Environment Diagnostics**: Run `python -m ultimate_mcp_server.cli env --verbose --suggest`
- **Issues**: Report problems at [GitHub Issues](https://github.com/android606/ultimate_mcp_server_windows/issues)
- **Original Project**: Visit [Dicklesworthstone's Ultimate MCP Server](https://github.com/Dicklesworthstone/ultimate_mcp_server) for core documentation
- **CLI Help**: Run `python -m ultimate_mcp_server.cli --help` for command details

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