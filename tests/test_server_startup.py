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
        
        # Start server process
        cmd = [
            sys.executable, "-m", "ultimate_mcp_server", "run",
            "--port", str(self.port),
            "--host", self.host,
            "--debug",
            "--load-all-tools"  # Load all tools to include get_all_tools_status
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
    
    def _monitor_logs(self):
        """Monitor server logs for ready signal"""
        if not self.server_process:
            return
            
        try:
            for line in iter(self.server_process.stderr.readline, ''):
                self.startup_logs.append(line.strip())
                print(f"[SERVER] {line.strip()}")
                
                if "🚀 SERVER READY FOR REQUESTS" in line:
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
            
            # Check that ready message was logged
            ready_logs = [log for log in server.startup_logs if "SERVER READY FOR REQUESTS" in log]
            assert len(ready_logs) > 0, "Server did not report ready status"
            
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
                
                assert not result.isError, f"list_directory failed: {result.error}"
                assert len(result.content) > 0, "No directory content returned"

    @pytest.mark.asyncio
    @pytest.mark.tools_status
    async def test_get_all_tools_status(self, mcp_server):
        """Test get_all_tools_status tool returns comprehensive status without errors"""
        session_url = f"http://{mcp_server.host}:{mcp_server.port}/sse"
        
        async with sse_client(session_url) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()
                
                # First check if get_all_tools_status tool is available
                tools = await session.list_tools()
                tool_names = [tool.name for tool in tools.tools]
                
                if "get_all_tools_status" not in tool_names:
                    pytest.skip("get_all_tools_status tool not available in this configuration")
                
                # Call the get_all_tools_status tool
                result = await session.call_tool("get_all_tools_status", {})
                
                # Verify the call completed successfully
                assert not result.isError, f"get_all_tools_status failed: {result.error}"
                assert len(result.content) > 0, "No content returned from get_all_tools_status"
                
                # Parse and validate the response structure
                content = result.content[0].text
                data = json.loads(content)
                
                # Verify required fields are present
                assert "tools_status" in data, "tools_status field missing"
                assert "summary" in data, "summary field missing"
                
                # Verify tools_status is a list
                assert isinstance(data["tools_status"], list), "tools_status should be a list"
                
                # Verify summary structure
                summary = data["summary"]
                expected_summary_keys = ["total_tools", "available", "unavailable", "loading", "error", "disabled_by_config"]
                for key in expected_summary_keys:
                    assert key in summary, f"summary missing expected key: {key}"
                    assert isinstance(summary[key], int), f"summary[{key}] should be an integer"
                
                # Verify total_tools count makes sense
                assert summary["total_tools"] >= 0, "total_tools should be non-negative"
                assert len(data["tools_status"]) <= summary["total_tools"], "tools_status list length inconsistent with total_tools"
                
                # Verify each tool status entry has required fields
                for i, tool_status in enumerate(data["tools_status"]):
                    assert "tool_name" in tool_status, f"tools_status[{i}] missing tool_name"
                    assert "status" in tool_status, f"tools_status[{i}] missing status"
                    
                    # Verify status is a valid value
                    valid_statuses = ["AVAILABLE", "UNAVAILABLE", "LOADING", "DISABLED_BY_CONFIG", "ERROR", "UNKNOWN"]
                    assert tool_status["status"] in valid_statuses, f"tools_status[{i}] has invalid status: {tool_status['status']}"
                
                # Log some useful information for debugging
                print(f"✅ get_all_tools_status returned {summary['total_tools']} tools:")
                print(f"   Available: {summary['available']}")
                print(f"   Unavailable: {summary['unavailable']}")
                print(f"   Loading: {summary['loading']}")
                print(f"   Error: {summary['error']}")
                print(f"   Disabled by config: {summary['disabled_by_config']}")
                
                # Verify no errors in the response itself
                assert "error" not in data or data["error"] is None, f"get_all_tools_status returned an error: {data.get('error')}"


class TestRegressionPrevention:
    """Tests to prevent regression of known issues"""
    
    @pytest.mark.asyncio
    async def test_no_initialization_race_condition(self):
        """
        Regression test: Ensure 'Received request before initialization was complete' 
        error does not occur
        """
        server = MCPServerFixture(port=8028)  # Unique port for this test
        try:
            await server.start_and_wait()
            
            # Immediately try to connect and make requests
            # This should NOT fail with initialization error
            session_url = f"http://{server.host}:{server.port}/sse"
            
            async with sse_client(session_url) as (read, write):
                async with ClientSession(read, write) as session:
                    await session.initialize()
                    
                    # These rapid-fire requests should all succeed
                    for i in range(5):
                        tools = await session.list_tools()
                        assert len(tools.tools) > 0
                        
                        result = await session.call_tool("echo", {"message": f"test {i}"})
                        assert not result.isError
                        
        finally:
            await server.stop()
    
    @pytest.mark.asyncio
    async def test_server_config_fields_exist(self):
        """
        Regression test: Ensure ServerConfig has all required fields
        """
        server = MCPServerFixture(port=8029)  # Unique port for this test
        try:
            await server.start_and_wait()
            
            # Server should start without ValueError about missing config fields
            error_logs = [log for log in server.startup_logs if "ValueError" in log and "object has no field" in log]
            assert len(error_logs) == 0, f"Server config errors found: {error_logs}"
            
        finally:
            await server.stop()


if __name__ == "__main__":
    # Run tests if executed directly
    pytest.main([__file__, "-v"]) 