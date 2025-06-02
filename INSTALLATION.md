# 📦 Ultimate MCP Server Installation Guide

## Quick Installation

### Option 1: Install from PyPI (Recommended - when published)
```bash
# Install the package
pip install ultimate-mcp-server

# Run the server
umcp run --port 8013
```

### Option 2: Install from Source (Current)
```bash
# Clone the repository
git clone https://github.com/Dicklesworthstone/ultimate_mcp_server.git
cd ultimate_mcp_server

# Install with uv (recommended)
uv sync
uv run umcp run --port 8013

# Or with pip
pip install -e .
umcp run --port 8013
```

### Option 3: Docker (Containerized)
```bash
# Build and run with Docker
docker build -t ultimate-mcp-server .
docker run -p 8013:8013 ultimate-mcp-server

# Or use docker-compose
docker-compose up
```

## System Requirements

- **Python**: 3.13 or higher
- **Operating System**: Windows, macOS, Linux
- **RAM**: 4GB minimum, 8GB+ recommended
- **Storage**: 2GB for dependencies and caching

## Installation Methods Detailed

### Method 1: uv (Ultra-fast Python Package Manager)

First install `uv` if you don't have it:
```bash
# On macOS/Linux
curl -LsSf https://astral.sh/uv/install.sh | sh

# On Windows
powershell -c "irm https://astral.sh/uv/install.ps1 | iex"
```

Then install and run:
```bash
git clone https://github.com/Dicklesworthstone/ultimate_mcp_server.git
cd ultimate_mcp_server
uv sync  # Install all dependencies
uv run umcp run --port 8013
```

### Method 2: Traditional pip

```bash
git clone https://github.com/Dicklesworthstone/ultimate_mcp_server.git
cd ultimate_mcp_server
pip install -e ".[all]"  # Install with all optional dependencies
umcp run --port 8013
```

### Method 3: Development Setup

```bash
git clone https://github.com/Dicklesworthstone/ultimate_mcp_server.git
cd ultimate_mcp_server

# Install development dependencies
uv sync --dev

# Install pre-commit hooks
uv run pre-commit install

# Run tests
uv run pytest

# Run the server
uv run umcp run --port 8013
```

## Optional Dependencies

The package includes several optional dependency groups:

```bash
# Install with advanced ML features (torch, transformers)
pip install "ultimate-mcp-server[advanced]"

# Install with development tools
pip install "ultimate-mcp-server[dev]"

# Install with documentation tools  
pip install "ultimate-mcp-server[docs]"

# Install everything
pip install "ultimate-mcp-server[all]"
```

## Configuration

### Environment Variables
Create a `.env` file in your project directory:
```bash
# LLM Provider API Keys
OPENAI_API_KEY=your_openai_key_here
ANTHROPIC_API_KEY=your_anthropic_key_here
GOOGLE_API_KEY=your_google_key_here
DEEPSEEK_API_KEY=your_deepseek_key_here

# Optional configurations
MCP_HOST=127.0.0.1
MCP_PORT=8013
MCP_LOG_LEVEL=INFO
```

### Basic Configuration
```bash
# Check available providers
umcp providers

# Test a provider
umcp test openai --prompt "Hello, world!"

# Run with specific settings
umcp run --host 0.0.0.0 --port 8013 --debug
```

## Verification

After installation, verify everything works:

```bash
# Check version
umcp --version

# Check available tools
umcp tools

# Test basic functionality
umcp complete --provider openai --prompt "What is 2+2?"

# Start the server and test with curl
umcp run --port 8013 &
curl http://localhost:8013/sse
```

## Server Endpoints & Monitoring

### SSE (Server-Sent Events) Endpoint

The Ultimate MCP Server uses **Server-Sent Events (SSE)** for MCP protocol communication. When connecting to the server:

- **Correct endpoint**: `http://host:port/sse`
- **Incorrect**: `http://host:port` (will return 404)

Examples:
```bash
# Default local server
curl http://localhost:8013/sse

# Custom port
curl http://localhost:8014/sse

# Remote server
curl http://your-server.com:8013/sse
```

### Tool Status Monitoring

The server includes a `get_all_tools_status` tool that provides comprehensive status information about all tools and their dependencies:

```python
# Using the MCP client programmatically
import asyncio
from mcp import ClientSession
from mcp.client.sse import sse_client

async def check_tool_status():
    async with sse_client("http://localhost:8013/sse") as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            result = await session.call_tool("get_all_tools_status", {})
            print(result)

# Run the status check
asyncio.run(check_tool_status())
```

This tool reports:
- **Tool availability** (AVAILABLE, UNAVAILABLE, LOADING, DISABLED_BY_CONFIG, ERROR)
- **LLM provider status** for tools that depend on external APIs
- **Missing dependencies** and configuration issues
- **Detailed error messages** for troubleshooting

### Configuration for External Clients

When connecting external MCP clients (like Cursor AI, the tool context estimator, or custom applications), ensure you:

1. **Use the correct SSE endpoint**: Always append `/sse` to your server URL
2. **Configure environment variables** for server discovery:
   ```bash
   # In your .env file
   MCP_SERVER_HOST=127.0.0.1
   MCP_SERVER_PORT=8013
   MCP_SERVER_URL=http://127.0.0.1:8013/sse  # Full URL with /sse
   ```

### Testing Connectivity

Use the included tool context estimator to verify your server is accessible:

```bash
# Test connection to default server
python mcp_tool_context_estimator.py --quiet

# Test specific server
python mcp_tool_context_estimator.py --url http://localhost:8014 --quiet

# The script will automatically append /sse if missing
```

## Upgrading

### From PyPI (when available)
```bash
pip install --upgrade ultimate-mcp-server
```

### From Source
```bash
cd ultimate_mcp_server
git pull origin main
uv sync  # Update dependencies
```

### Using Release Script (for maintainers)
```bash
# Bump version and create release
python setup_release.py patch  # or minor, major
```

## Troubleshooting

### Common Issues

1. **Import Errors**: Make sure you have Python 3.13+
   ```bash
   python --version  # Should be 3.13+
   ```

2. **Missing Dependencies**: Reinstall with all dependencies
   ```bash
   pip install -e ".[all]"
   ```

3. **Port Conflicts**: Change the port
   ```bash
   umcp run --port 8014
   ```

4. **Permission Issues**: On some systems, use user installs
   ```bash
   pip install --user -e .
   ```

### Getting Help

- **Documentation**: Check the main README.md
- **Issues**: Report bugs on GitHub
- **Examples**: See the `examples/` directory
- **CLI Help**: Run `umcp --help` for command details

### Platform-Specific Notes

#### Windows
- Use PowerShell or Command Prompt
- Some OCR features require Visual C++ redistributables
- Windows Defender may need exclusions for large ML models

#### macOS
- Install Xcode command line tools: `xcode-select --install`
- Some dependencies may require Homebrew packages

#### Linux
- Install system dependencies for OCR:
  ```bash
  sudo apt-get install tesseract-ocr poppler-utils
  ```

## Development Installation

For contributors and developers:

```bash
# Clone and setup development environment
git clone https://github.com/Dicklesworthstone/ultimate_mcp_server.git
cd ultimate_mcp_server

# Install in development mode with all dependencies
uv sync --dev

# Install pre-commit hooks for code quality
uv run pre-commit install

# Run the test suite
uv run pytest

# Run linting and type checks
uv run ruff check
uv run mypy ultimate_mcp_server

# Start the server for development
uv run umcp run --debug --port 8013
```

## Next Steps

After installation:

1. **Configure API Keys**: Add your LLM provider keys to `.env`
2. **Test Basic Functionality**: Run `umcp test openai` 
3. **Explore Examples**: Check out `examples/` directory
4. **Read Documentation**: Review the main README.md
5. **Join Development**: See CONTRIBUTING.md for contribution guidelines 