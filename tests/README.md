# Ultimate MCP Server Test Suite

## Overview

This directory contains comprehensive tests for the Ultimate MCP Server using pytest with async support.

## Test Framework Features

- **Async Support**: Full pytest-asyncio integration for testing async MCP operations
- **Server Lifecycle Management**: Automated server startup/shutdown for integration tests
- **Port Conflict Avoidance**: Each test uses unique ports to avoid conflicts with production instances
- **Comprehensive Tool Testing**: Tests for core MCP functionality including tool execution and status checking

## Port Allocation

To avoid conflicts between tests and production servers, the following port allocation is used:

| Service/Test | Port | Description |
|--------------|------|-------------|
| Production (default) | 8013 | Default production server port |
| Production (alternate) | 8014-8015 | Alternate production ports per workspace rules |
| Manual testing (`smart_mcp_test.py`) | 8030 | Interactive testing script |
| `MCPServerFixture` default | 8024 | Base test fixture |
| `mcp_server` fixture | 8025 | Async pytest fixture |
| Individual tests | 8026-8029 | Dedicated ports for specific test cases |

## Key Tests

### `test_get_all_tools_status`
- **Purpose**: Validates the `get_all_tools_status` meta-API tool
- **Features**: 
  - Starts server with `--load-all-tools` to include all meta tools
  - Connects via MCP SSE client
  - Executes the tool and validates response structure
  - Checks for proper error handling
- **Marker**: `@pytest.mark.tools_status`

### `test_echo_tool_execution`
- **Purpose**: Basic MCP connectivity and tool execution test
- **Features**: Tests basic MCP protocol functionality

### Other Integration Tests
- Server startup validation
- Health endpoint testing
- Configuration validation
- Tool listing functionality

## Running Tests

### Run All Tests
```bash
python -m pytest tests/test_server_startup.py
```

### Run Specific Test Categories
```bash
# Run only tools_status tests
python -m pytest tests/test_server_startup.py -m tools_status

# Run integration tests
python -m pytest tests/test_server_startup.py -m integration

# Run excluding slow tests
python -m pytest tests/test_server_startup.py -m "not slow"
```

### Run Individual Tests
```bash
# Test the get_all_tools_status functionality
python -m pytest tests/test_server_startup.py::TestMCPFunctionality::test_get_all_tools_status -v

# Quick connectivity test
python -m pytest tests/test_server_startup.py::TestMCPFunctionality::test_echo_tool_execution -v
```

## Configuration

The test configuration is defined in `pytest.ini` with:
- Async mode enabled
- Custom markers for test categorization
- Increased timeouts for server startup operations
- Warning filters for cleaner output

## Requirements

- `pytest`
- `pytest-asyncio`
- `mcp` library
- Ultimate MCP Server installed in development mode

## Notes

- Tests use the `--load-all-tools` flag to ensure all meta-API tools are available
- Each test class manages its own server instance with unique ports
- Server logs are monitored for "ready for requests" messages before proceeding with tests
- Encoding issues are handled gracefully with UTF-8 and error replacement 