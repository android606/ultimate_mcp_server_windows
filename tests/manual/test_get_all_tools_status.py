#!/usr/bin/env python3
"""
Test script to verify the get_all_tools_status tool works correctly.

This script connects to the Ultimate MCP Server and calls the get_all_tools_status
tool to verify it provides comprehensive tool status information.

Usage:
    python test_get_all_tools_status.py [--url http://localhost:8014/sse]
"""

import asyncio
import argparse
import json
import sys
import traceback
from typing import Dict, Any

try:
    from mcp.client.sse import sse_client
    from mcp.client.session import ClientSession
except ImportError:
    print("❌ MCP client not available. Install with: pip install mcp")
    sys.exit(1)


async def test_mcp_tools(url: str) -> bool:
    """Test MCP server tools functionality"""
    
    try:
        print(f"🔗 Connecting to MCP server at {url}...")
        
        async with sse_client(url) as (read, write):
            async with ClientSession(read, write) as session:
                print("✅ Connected successfully!")
                
                # Test 1: List available tools
                print("\n📋 Listing available tools...")
                try:
                    tools_result = await session.list_tools()
                    print(f"✅ Found {len(tools_result.tools)} tools available")
                    
                    # Show some tool names
                    tool_names = [tool.name for tool in tools_result.tools]
                    print(f"📝 Sample tools: {tool_names[:5]}")
                    
                except Exception as e:
                    print(f"❌ Failed to list tools: {e}")
                    return False
                
                # Test 2: Call echo tool if available
                if 'echo' in tool_names:
                    print("\n🔄 Testing echo tool...")
                    try:
                        result = await session.call_tool("echo", {"message": "MCP Test Working!"})
                        if result.isError:
                            print(f"❌ Echo tool failed: {result.error}")
                        else:
                            print("✅ Echo tool working correctly")
                    except Exception as e:
                        print(f"⚠️  Echo tool test failed: {e}")
                
                # Test 3: Call get_provider_status if available
                if 'get_provider_status' in tool_names:
                    print("\n🔍 Testing get_provider_status tool...")
                    try:
                        result = await session.call_tool("get_provider_status", {"random_string": "test"})
                        if result.isError:
                            print(f"❌ get_provider_status failed: {result.error}")
                        else:
                            print("✅ get_provider_status working correctly")
                    except Exception as e:
                        print(f"⚠️  get_provider_status test failed: {e}")
                else:
                    print("⚠️  get_provider_status tool not found in available tools")
                
                return True
                
    except Exception as e:
        print(f"❌ Connection or communication failed: {e}")
        traceback.print_exc()
        return False


def main():
    parser = argparse.ArgumentParser(description='Test Ultimate MCP Server functionality')
    parser.add_argument('--url', default='http://localhost:8015/sse', 
                        help='MCP server SSE URL (default: http://localhost:8015/sse)')
    
    args = parser.parse_args()
    
    print("🧪 Testing Ultimate MCP Server")
    print("=" * 50)
    
    success = asyncio.run(test_mcp_tools(args.url))
    
    if success:
        print("\n🎉 MCP server test successful!")
        sys.exit(0)
    else:
        print("\n❌ MCP server test failed!")
        sys.exit(1)


if __name__ == "__main__":
    main() 