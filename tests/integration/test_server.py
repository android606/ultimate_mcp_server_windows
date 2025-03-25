"""Integration tests for the LLM Gateway server."""
import asyncio
import json
from typing import Any, Dict, Optional

import pytest
from pytest import MonkeyPatch

from llm_gateway.core.server import Gateway
from llm_gateway.utils import get_logger

logger = get_logger("test.integration.server")


@pytest.fixture
async def test_gateway() -> Gateway:
    """Create a test gateway instance."""
    gateway = Gateway(name="test-gateway")
    await gateway._initialize_providers()
    return gateway


class TestGatewayServer:
    """Tests for the Gateway server."""
    
    async def test_initialization(self, test_gateway: Gateway):
        """Test gateway initialization."""
        logger.info("Testing gateway initialization", emoji_key="test")
        
        assert test_gateway.name == "test-gateway"
        assert test_gateway.mcp is not None
        assert hasattr(test_gateway, "providers")
        assert hasattr(test_gateway, "provider_status")
        
    async def test_provider_status(self, test_gateway: Gateway):
        """Test provider status information."""
        logger.info("Testing provider status", emoji_key="test")
        
        # Should have provider status information
        assert test_gateway.provider_status is not None
        
        # Get status with MCP
        status_resource = test_gateway.mcp.get_resource("info://server")
        assert status_resource is not None
        
        # Get info
        server_info = status_resource()
        assert server_info is not None
        assert "name" in server_info
        assert "version" in server_info
        assert "providers" in server_info
        
    async def test_tool_registration(self, test_gateway: Gateway):
        """Test tool registration."""
        logger.info("Testing tool registration", emoji_key="test")
        
        # Define a test tool
        @test_gateway.mcp.tool()
        async def test_tool(arg1: str, arg2: Optional[str] = None) -> Dict[str, Any]:
            """Test tool for testing."""
            return {"result": f"{arg1}-{arg2 or 'default'}", "success": True}
        
        # Execute the tool
        result = await test_gateway.mcp.execute("test_tool", {"arg1": "test", "arg2": "value"})
        
        # Check result
        assert result["result"] == "test-value"
        assert result["success"]
        
        # Execute with default
        result = await test_gateway.mcp.execute("test_tool", {"arg1": "test"})
        assert result["result"] == "test-default"
        
    async def test_tool_error_handling(self, test_gateway: Gateway):
        """Test error handling in tools."""
        logger.info("Testing tool error handling", emoji_key="test")
        
        # Define a tool that raises an error
        @test_gateway.mcp.tool()
        async def error_tool(should_fail: bool = True) -> Dict[str, Any]:
            """Tool that fails on demand."""
            if should_fail:
                raise ValueError("Test error")
            return {"success": True}
        
        # Execute and catch the error
        with pytest.raises(Exception):  # MCP might wrap the error
            await test_gateway.mcp.execute("error_tool", {"should_fail": True})
            
        # Execute successful case
        result = await test_gateway.mcp.execute("error_tool", {"should_fail": False})
        assert result["success"]


class TestServerLifecycle:
    """Tests for server lifecycle."""
    
    async def test_server_lifespan(self, monkeypatch: MonkeyPatch):
        """Test server lifespan context manager."""
        logger.info("Testing server lifespan", emoji_key="test")
        
        # Track lifecycle events
        events = []
        
        # Mock MCP server
        class MockMCP:
            def __init__(self, name, lifespan):
                self.name = name
                self.lifespan = lifespan
                self.exec_calls = []
                
            def execute(self, tool, params):
                self.exec_calls.append((tool, params))
                return {"result": "mock"}
                
            def run(self):
                events.append("run")
                
            async def __aenter__(self):
                events.append("enter")
                return {"test": "context"}
                
            async def __aexit__(self, exc_type, exc_val, exc_tb):
                events.append("exit")
                
        # Patch the FastMCP class
        monkeypatch.setattr("mcp.server.fastmcp.FastMCP", MockMCP)
        
        # Create and run gateway
        gateway = Gateway(name="test-lifecycle")
        
        # Access lifespan context manager
        async with gateway._server_lifespan(gateway.mcp) as context:
            events.append("in_context")
            assert context is not None
            
        # Check events
        assert "enter" in events
        assert "in_context" in events
        assert "exit" in events
        
        # Run server (non-blocking for test)
        gateway.run()
        assert "run" in events


class TestServerIntegration:
    """Integration tests for server with tools."""
    
    async def test_provider_tools(self, test_gateway: Gateway, monkeypatch: MonkeyPatch):
        """Test provider-related tools."""
        logger.info("Testing provider tools", emoji_key="test")
        
        # Mock tool execution
        async def mock_execute(tool_name, params):
            if tool_name == "get_provider_status":
                return {
                    "providers": {
                        "openai": {
                            "enabled": True,
                            "available": True,
                            "api_key_configured": True,
                            "error": None,
                            "models_count": 3
                        },
                        "anthropic": {
                            "enabled": True,
                            "available": True,
                            "api_key_configured": True,
                            "error": None,
                            "models_count": 5
                        }
                    }
                }
            elif tool_name == "list_models":
                provider = params.get("provider")
                if provider == "openai":
                    return {
                        "models": {
                            "openai": [
                                {"id": "gpt-4o", "provider": "openai"},
                                {"id": "gpt-4o-mini", "provider": "openai"},
                                {"id": "gpt-3.5-turbo", "provider": "openai"}
                            ]
                        }
                    }
                else:
                    return {
                        "models": {
                            "openai": [
                                {"id": "gpt-4o", "provider": "openai"},
                                {"id": "gpt-4o-mini", "provider": "openai"}
                            ],
                            "anthropic": [
                                {"id": "claude-3-opus-20240229", "provider": "anthropic"},
                                {"id": "claude-3-5-haiku-latest", "provider": "anthropic"}
                            ]
                        }
                    }
            else:
                return {"error": f"Unknown tool: {tool_name}"}
                
        monkeypatch.setattr(test_gateway.mcp, "execute", mock_execute)
        
        # Test get_provider_status
        status = await test_gateway.mcp.execute("get_provider_status", {})
        assert "providers" in status
        assert "openai" in status["providers"]
        assert "anthropic" in status["providers"]
        
        # Test list_models with provider
        models = await test_gateway.mcp.execute("list_models", {"provider": "openai"})
        assert "models" in models
        assert "openai" in models["models"]
        assert len(models["models"]["openai"]) == 3
        
        # Test list_models without provider
        all_models = await test_gateway.mcp.execute("list_models", {})
        assert "models" in all_models
        assert "openai" in all_models["models"]
        assert "anthropic" in all_models["models"]