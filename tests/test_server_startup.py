#!/usr/bin/env python3
"""
Comprehensive pytest tests for Ultimate MCP Server startup and functionality.

Port Allocation for Tests (to avoid conflicts):
- Production server default: 8013
- Production server alternate: 8014, 8015
- Manual testing (smart_mcp_test.py): 8030
- MCPServerFixture default: 8024
- mcp_server fixture: 8025
- test_server_starts_and_reports_ready: 8026
- test_health_endpoint_responds: 8027
- test_no_initialization_race_condition: 8028
- test_server_config_fields_exist: 8029
"""

import asyncio
import json
import os
import platform
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Optional

import pytest
import pytest_asyncio

try:
    from mcp import ClientSession
    from mcp.client.sse import sse_client
except ImportError:
    pytest.skip("MCP library not available", allow_module_level=True)


class MCPServerFixture:
    """Test fixture for managing MCP server lifecycle"""
    
    def __init__(self, host="127.0.0.1", port=8024):  # Changed default port to avoid conflicts
        self.host = host
        self.port = port
        self.server_process: Optional[subprocess.Popen] = None
        self.ready_event = threading.Event()
        self.log_thread: Optional[threading.Thread] = None
        self.startup_logs = []
        
    async def start_and_wait(self, timeout=120):
        """Start server and wait for ready signal"""
        print(f"🚀 Starting test server on {self.host}:{self.port}")
        
        # Use the current Python interpreter if we're already in a virtual environment
        venv_python = self._get_venv_python_path()
        print(f"Using Python interpreter: {venv_python}")
        
        # Start server process
        cmd = [
            venv_python, "-m", "ultimate_mcp_server", "run",
            "--port", str(self.port),
            "--host", self.host,
            "--debug",
            "--load-all-tools",  # Load all tools to include get_all_tools_status
            "--skip-env-check",  # Skip environment validation
            "--force"            # Force server to start despite issues
        ]
        
        self.server_process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding='utf-8',
            errors='replace',  # Replace problematic characters
            bufsize=1
        )
        
        # Monitor logs in background
        self.log_thread = threading.Thread(
            target=self._monitor_logs,
            daemon=True
        )
        self.log_thread.start()
        
        # Wait for ready signal
        if self.ready_event.wait(timeout):
            print("✅ Server ready")
            return True
        else:
            await self.stop()
            raise TimeoutError(f"Server startup timed out after {timeout}s")
    
    def _get_venv_python_path(self):
        """Get the path to the virtual environment Python executable based on platform"""
        # If we're already in a virtual environment, use the current Python
        if self._is_in_virtualenv():
            return sys.executable
        
        # Find the project root (where this file is located)
        current_file = Path(__file__)
        project_root = current_file.parent.parent  # Go up from tests/
        
        # Common virtual environment directory names
        venv_dirs = ['.venv', 'venv', 'env', '.env']
        
        # Try to find the venv directory in project root
        venv_path = None
        for venv_dir in venv_dirs:
            path = project_root / venv_dir
            if path.exists() and path.is_dir():
                # Check for common signs of a virtual environment
                if (path / 'pyvenv.cfg').exists():
                    venv_path = path
                    break
        
        if not venv_path:
            # Fallback to the current Python interpreter if venv not found
            print("⚠️ Virtual environment not found, using current Python interpreter")
            return sys.executable
        
        # Build the path to Python executable based on platform
        if platform.system() == 'Windows':
            python_path = venv_path / 'Scripts' / 'python.exe'
        else:  # Linux, macOS, etc.
            python_path = venv_path / 'bin' / 'python'
        
        # Verify the executable exists
        if not python_path.exists():
            print(f"⚠️ Python executable not found at {python_path}, falling back to current interpreter")
            return sys.executable
            
        return str(python_path.absolute())
    
    def _is_in_virtualenv(self):
        """Check if we're already in a virtual environment"""
        # Multiple checks for different ways to detect virtual environments
        return (
            hasattr(sys, 'real_prefix') or 
            (hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix) or
            os.path.exists(os.path.join(sys.prefix, 'pyvenv.cfg'))
        )
    
    def _monitor_logs(self):
        """Monitor logs for server ready signal"""
        if not self.server_process:
            return
            
        try:
            for line in iter(self.server_process.stderr.readline, ''):
                line_str = line.strip()
                self.startup_logs.append(line_str)
                print(f"[SERVER] {line_str}")
                
                # Check for Application startup complete message (modern way)
                if "Application startup complete" in line_str:
                    print("💡 Application startup complete detected!")
                    self.ready_event.set()
                    break
                    
                # Check for old-style ready message as a fallback
                if "SERVER READY FOR REQUESTS" in line_str:
                    print("💡 SERVER READY FOR REQUESTS detected!")
                    self.ready_event.set()
                    break
                    
                if self.server_process.poll() is not None:
                    break
                    
        except Exception as e:
            print(f"Error monitoring logs: {e}")
    
    async def stop(self):
        """Stop the server"""
        if self.server_process:
            print("🛑 Stopping test server")
            self.server_process.terminate()
            try:
                self.server_process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self.server_process.kill()
                self.server_process.wait()
            self.server_process = None


@pytest_asyncio.fixture
async def mcp_server():
    """Pytest fixture for MCP server with unique port"""
    server = MCPServerFixture(port=8025)  # Unique port for this fixture
    await server.start_and_wait()
    yield server
    await server.stop()


class TestServerStartup:
    """Test server startup and basic functionality"""
    
    @pytest.mark.asyncio
    async def test_server_starts_and_reports_ready(self):
        """Test that server starts and reports ready status"""
        server = MCPServerFixture(port=8026)  # Unique port for this test
        try:
            await server.start_and_wait()
            
            # If we got here, the server started successfully
            assert True, "Server started successfully"
            
        finally:
            await server.stop()
    
    @pytest.mark.asyncio
    async def test_health_endpoint_responds(self):
        """Test that health endpoint responds correctly"""
        server = MCPServerFixture(port=8027)  # Unique port for this test
        try:
            await server.start_and_wait()
            
            # Test health endpoint with aiohttp
            import aiohttp
            async with aiohttp.ClientSession() as session:
                url = f"http://{server.host}:{server.port}/health"
                async with session.get(url) as response:
                    assert response.status == 200
                    data = await response.json()
                    assert data["status"] == "healthy"
                    
        finally:
            await server.stop()


class TestMCPFunctionality:
    """Test MCP protocol functionality"""
    
    @pytest.mark.asyncio
    async def test_sse_connection_works(self, mcp_server):
        """Test that SSE connection can be established"""
        session_url = f"http://{mcp_server.host}:{mcp_server.port}/sse"
        
        async with sse_client(session_url) as (read, write):
            # Connection successful if we get here
            assert True
    
    @pytest.mark.asyncio
    async def test_mcp_session_initialization(self, mcp_server):
        """Test MCP session can be initialized"""
        session_url = f"http://{mcp_server.host}:{mcp_server.port}/sse"
        
        async with sse_client(session_url) as (read, write):
            async with ClientSession(read, write) as session:
                init_result = await session.initialize()
                assert init_result is not None
    
    @pytest.mark.asyncio
    async def test_list_tools_works(self, mcp_server):
        """Test that list_tools returns available tools"""
        session_url = f"http://{mcp_server.host}:{mcp_server.port}/sse"
        
        async with sse_client(session_url) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()
                
                tools = await session.list_tools()
                assert len(tools.tools) > 0, "No tools found"
                
                # Check for expected tools
                tool_names = [tool.name for tool in tools.tools]
                assert "echo" in tool_names, "Echo tool not found"
    
    @pytest.mark.asyncio
    async def test_echo_tool_execution(self, mcp_server):
        """Test that echo tool can be executed"""
        session_url = f"http://{mcp_server.host}:{mcp_server.port}/sse"
        
        async with sse_client(session_url) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()
                
                test_message = "Hello from pytest!"
                result = await session.call_tool("echo", {"message": test_message})
                
                assert not result.isError, f"Echo tool failed: {result.error}"
                assert len(result.content) > 0, "No content returned"
                
                # Parse the response
                content = result.content[0].text
                data = json.loads(content)
                assert data["message"] == test_message, "Echo message mismatch"
    
    @pytest.mark.asyncio
    async def test_provider_status_tool(self, mcp_server):
        """Test provider status tool functionality"""
        session_url = f"http://{mcp_server.host}:{mcp_server.port}/sse"
        
        async with sse_client(session_url) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()
                
                result = await session.call_tool("get_provider_status", {"random_string": "test"})
                
                # Provider status may fail if no providers configured, but tool should respond
                if not result.isError:
                    content = result.content[0].text
                    data = json.loads(content)
                    assert "providers" in data, "Provider status missing providers key"
    
    @pytest.mark.asyncio
    async def test_filesystem_tool_works(self, mcp_server):
        """Test filesystem tool functionality"""
        session_url = f"http://{mcp_server.host}:{mcp_server.port}/sse"
        
        async with sse_client(session_url) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()
                
                # Test list_directory tool
                result = await session.call_tool("list_directory", {"path": "."})
                
                assert not result.isError, f"List directory failed: {result.error}"
                assert len(result.content) > 0, "No content returned"
                
                # Parse the response
                content = result.content[0].text
                data = json.loads(content)
                assert "files" in data, "List directory missing files key"
                
    @pytest.mark.asyncio
    @pytest.mark.tools_status
    async def test_get_all_tools_status(self, mcp_server):
        """Test get_all_tools_status functionality"""
        session_url = f"http://{mcp_server.host}:{mcp_server.port}/sse"
        
        async with sse_client(session_url) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()
                
                # Call the get_all_tools_status tool
                result = await session.call_tool("get_all_tools_status", {"random_string": "test"})
                
                assert not result.isError, f"get_all_tools_status failed: {result.error}"
                assert len(result.content) > 0, "No content returned"
                
                # Parse the response
                content = result.content[0].text
                data = json.loads(content)
                
                # Check structure
                assert "tools" in data, "Missing tools key in response"
                assert isinstance(data["tools"], dict), "Tools should be a dictionary"
                
                # Check at least a few common tools are present and marked as available
                important_tools = ["generate_completion", "echo", "list_directory"]
                for tool in important_tools:
                    assert tool in data["tools"], f"Tool {tool} missing from status"
                    assert "available" in data["tools"][tool], f"Tool {tool} missing available status"


class TestRegressionPrevention:
    """Tests designed specifically to prevent regressions"""
    
    @pytest.mark.asyncio
    async def test_no_initialization_race_condition(self):
        """Test that multiple connections don't cause race conditions during initialization"""
        server = MCPServerFixture(port=8028)  # Unique port for this test
        try:
            await server.start_and_wait()
            
            # Create multiple connections in quick succession
            session_url = f"http://{server.host}:{server.port}/sse"
            
            async def connect_and_initialize():
                async with sse_client(session_url) as (read, write):
                    async with ClientSession(read, write) as session:
                        # Just initialize and exit
                        await session.initialize()
                        return True
            
            # Start multiple connections concurrently
            tasks = [connect_and_initialize() for _ in range(5)]
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Check all connections succeeded
            for i, result in enumerate(results):
                assert result is True, f"Connection {i} failed: {result}"
                
        finally:
            await server.stop()
    
    @pytest.mark.asyncio
    async def test_server_config_fields_exist(self):
        """Test that server configuration fields are properly populated"""
        server = MCPServerFixture(port=8029)  # Unique port for this test
        try:
            await server.start_and_wait()
            
            # Connect to the server
            session_url = f"http://{server.host}:{server.port}/sse"
            
            async with sse_client(session_url) as (read, write):
                async with ClientSession(read, write) as session:
                    init_result = await session.initialize()
                    
                    # Access the server resource to check configuration
                    result = await session.get_resource("info://server")
                    assert not result.isError, f"Failed to get server info: {result.error}"
                    
                    content = result.content[0].text
                    data = json.loads(content)
                    
                    # Check critical fields exist
                    assert "version" in data, "Server info missing version"
                    assert "providers" in data, "Server info missing providers"
                    assert "tools_count" in data, "Server info missing tools_count"
                    
        finally:
            await server.stop()


if __name__ == "__main__":
    # Run tests if executed directly
    pytest.main([__file__, "-v"]) 