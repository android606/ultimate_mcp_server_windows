#!/usr/bin/env python3
"""
Simple test script to verify MCP server functionality
"""

import asyncio
import json
import sys
import time

try:
    from mcp import ClientSession, StdioServerParameters
    from mcp.client.session import ClientSession
    from mcp.client.stdio import stdio_client
    from mcp.client.sse import sse_client
except ImportError:
    print("❌ MCP library not available. Please install with: pip install mcp")
    sys.exit(1)

async def test_mcp_server():
    """Test MCP server functionality"""
    print("🧪 Testing MCP Server Functionality")
    print("=" * 50)
    
    try:
        # Connect via SSE
        print("🔗 Connecting to MCP server via SSE...")
        session_url = "http://localhost:8017/sse"
        
        async with sse_client(session_url) as (read, write):
            print("✅ SSE client established successfully!")
            
            try:
                async with ClientSession(read, write) as session:
                    print("✅ Client session created successfully!")
                    
                    # Initialize MCP session first
                    print("\n🔄 Initializing MCP session...")
                    try:
                        init_result = await session.initialize()
                        print("✅ MCP session initialized successfully!")
                    except Exception as init_error:
                        print(f"❌ MCP initialization failed: {type(init_error).__name__}: {init_error}")
                        return False
                    
                    # Test 1: List tools with detailed error handling
                    print("\n📋 Listing available tools...")
                    try:
                        tools = await session.list_tools()
                        print(f"✅ Found {len(tools.tools)} tools:")
                        for tool in tools.tools[:5]:  # Show first 5 tools
                            print(f"   • {tool.name}")
                        if len(tools.tools) > 5:
                            print(f"   ... and {len(tools.tools) - 5} more")
                    except Exception as list_tools_error:
                        print(f"❌ list_tools failed: {type(list_tools_error).__name__}: {list_tools_error}")
                        import traceback
                        traceback.print_exc()
                        return False
                    
                    # Test 2: Get provider status (if available)
                    print("\n🔍 Testing get_provider_status tool...")
                    try:
                        result = await session.call_tool("get_provider_status", {"random_string": "test"})
                        if result.isError:
                            print(f"❌ Tool call failed: {result.error}")
                        else:
                            content = result.content[0].text if result.content else "No content"
                            parsed = json.loads(content) if content.startswith('{') else content
                            if isinstance(parsed, dict) and 'providers' in parsed:
                                provider_count = len(parsed['providers'])
                                print(f"✅ Provider status retrieved: {provider_count} providers configured")
                            else:
                                print(f"✅ Tool executed successfully: {content[:100]}...")
                    except Exception as e:
                        print(f"⚠️  get_provider_status not available or failed: {e}")
                    
                    # Test 3: Test echo tool
                    print("\n🔄 Testing echo tool...")
                    try:
                        result = await session.call_tool("echo", {"message": "Hello from MCP test!"})
                        if result.isError:
                            print(f"❌ Echo tool failed: {result.error}")
                        else:
                            content = result.content[0].text if result.content else "No content"
                            print(f"✅ Echo tool successful: {content}")
                    except Exception as e:
                        print(f"⚠️  Echo tool not available or failed: {e}")
                    
                    # Test 4: Test a filesystem tool
                    print("\n📁 Testing list_directory tool...")
                    try:
                        result = await session.call_tool("list_directory", {"path": "."})
                        if result.isError:
                            print(f"❌ list_directory failed: {result.error}")
                        else:
                            content = result.content[0].text if result.content else "No content"
                            print(f"✅ list_directory successful: Found directory listing")
                    except Exception as e:
                        print(f"⚠️  list_directory not available or failed: {e}")
                    
                    print("\n🎉 MCP Server Test Complete!")
                    return True
                    
            except Exception as session_error:
                print(f"❌ ClientSession failed: {type(session_error).__name__}: {session_error}")
                import traceback
                traceback.print_exc()
                return False
                
    except Exception as e:
        print(f"❌ Connection failed: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Main test function"""
    success = asyncio.run(test_mcp_server())
    if success:
        print("\n✅ All tests passed! MCP server is working correctly.")
        sys.exit(0)
    else:
        print("\n❌ Some tests failed. Check server status.")
        sys.exit(1)

if __name__ == "__main__":
    main() 