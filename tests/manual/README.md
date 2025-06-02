# 🧪 Manual Test Scripts

This directory contains manual test scripts for the Ultimate MCP Server (Windows-optimized fork).

## Test Scripts

### **Environment and Setup Tests**
- **`test_environment_validation.py`** - Tests the environment validation system, checking virtual environment detection, Python version compatibility, and dependency verification
- **`debug_mcp_connection.py`** - Debug script for troubleshooting MCP protocol connections and SSE endpoint issues

### **Server Functionality Tests**  
- **`test_get_all_tools_status.py`** - Tests the `get_all_tools_status` tool functionality for monitoring tool availability
- **`test_get_all_tools_status_detailed.py`** - Extended version with detailed tool status information
- **`smart_mcp_test.py`** - Comprehensive MCP server functionality test with intelligent error handling
- **`simple_mcp_test.py`** - Basic MCP connectivity and tool listing test

## Running Manual Tests

### Prerequisites
```bash
# Ensure you're in the project root with activated virtual environment
cd ultimate_mcp_server_windows
.venv\Scripts\activate.bat  # Windows
# source .venv/bin/activate  # Linux/Mac

# Install test dependencies
pip install -e ".[test]"
```

### Environment Validation
```bash
# Test environment setup
python tests/manual/test_environment_validation.py

# Debug connection issues
python tests/manual/debug_mcp_connection.py
```

### Server Functionality
```bash
# Start server first (use different port for testing)
python -m ultimate_mcp_server.cli run --port 8024 --debug &

# Run tests
python tests/manual/simple_mcp_test.py
python tests/manual/smart_mcp_test.py
python tests/manual/test_get_all_tools_status.py
```

### Port Usage
These manual tests use ports 8024-8030 to avoid conflicts with production servers:
- **8024**: Basic functionality tests
- **8025**: Environment validation tests  
- **8026**: Integration tests
- **8027-8030**: Available for additional manual testing

## Test Categories

### ✅ **Environment Tests**
Validate setup, dependencies, and Windows-specific compatibility

### 🔌 **Connection Tests**  
Test MCP protocol connectivity, SSE endpoints, and communication

### 🛠️ **Tool Tests**
Verify individual tool functionality and status reporting

### 🏗️ **Integration Tests**
Test complete workflows and tool interactions

## Troubleshooting

### Common Issues
1. **Port conflicts**: Use different ports with `--port` flag
2. **Permission errors**: Run as Administrator if needed
3. **Virtual environment**: Ensure proper activation before testing
4. **Dependencies**: Install test dependencies with `pip install -e ".[test]"`

### Windows-Specific Issues
- **Path separators**: Tests handle both forward and backslash paths
- **Process termination**: Some tests may require manual process cleanup
- **Antivirus interference**: Add exclusions for test directories

## Contributing Test Scripts

When adding new manual tests:
1. **Use unique ports** (8024-8030 range)
2. **Add Windows compatibility** checks
3. **Include error handling** and cleanup
4. **Document purpose** in this README
5. **Follow naming convention**: `test_*.py` for test scripts

See `CONTRIBUTING.md` for detailed guidelines on test development. 