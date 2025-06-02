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
import traceback
from typing import Dict, Any

from mcp.client.sse import sse_client
from mcp import ClientSession


async def test_get_all_tools_status(server_url: str) -> Dict[str, Any]:
    """
    Test the get_all_tools_status tool by connecting to the server and calling it.
    
    Args:
        server_url: The SSE endpoint URL of the MCP server
        
    Returns:
        Dictionary containing the test results
    """
    print("🧪 Testing get_all_tools_status tool")
    print("=" * 50)
    
    result = {
        "success": False,
        "error": None,
        "error_details": None,
        "tool_found": False,
        "total_tools": 0,
        "server_info": {},
        "tool_result": None
    }
    
    try:
        print(f"🔗 Connecting to MCP server at {server_url}...")
        
        async with sse_client(server_url) as (read, write):
            print("🤝 Initializing MCP session...")
            
            async with ClientSession(read, write) as session:
                # Initialize the session
                init_result = await session.initialize()
                
                result["server_info"] = {
                    "name": init_result.serverInfo.name,
                    "version": init_result.serverInfo.version
                }
                
                print(f"✅ Connected to server: {result['server_info']['name']} v{result['server_info']['version']}")
                
                # List available tools
                print("📋 Listing available tools...")
                tools_result = await session.list_tools()
                tools = tools_result.tools if hasattr(tools_result, 'tools') else []
                
                result["total_tools"] = len(tools)
                print(f"📊 Found {result['total_tools']} total tools")
                
                # Check if get_all_tools_status is available
                tool_names = [tool.name for tool in tools]
                result["tool_found"] = "get_all_tools_status" in tool_names
                print(f"🔍 get_all_tools_status tool present: {result['tool_found']}")
                
                if not result["tool_found"]:
                    result["error"] = "get_all_tools_status tool not found in server tools"
                    return result
                
                # Call the get_all_tools_status tool
                print("🔧 Calling get_all_tools_status tool...")
                
                try:
                    # Try the tool call with detailed error capture
                    tool_result = await session.call_tool("get_all_tools_status", {})
                    
                    print("✅ Tool call succeeded!")
                    result["success"] = True
                    result["tool_result"] = tool_result.content if hasattr(tool_result, 'content') else str(tool_result)
                    
                    # Print a summary of the results
                    if hasattr(tool_result, 'content') and isinstance(tool_result.content, list):
                        for content_item in tool_result.content:
                            if hasattr(content_item, 'text'):
                                try:
                                    parsed_result = json.loads(content_item.text)
                                    if "tools_status" in parsed_result:
                                        tools_status = parsed_result["tools_status"]
                                        print(f"📊 Tool status report contains {len(tools_status)} tools")
                                        
                                        # Count by status
                                        status_counts = {}
                                        for tool_status in tools_status:
                                            status = tool_status.get("status", "UNKNOWN")
                                            status_counts[status] = status_counts.get(status, 0) + 1
                                        
                                        print("📈 Status breakdown:")
                                        for status, count in status_counts.items():
                                            print(f"   {status}: {count}")
                                            
                                except json.JSONDecodeError:
                                    print("⚠️  Tool result is not valid JSON")
                                except Exception as parse_e:
                                    print(f"⚠️  Error parsing tool result: {str(parse_e)}")
                                    
                except Exception as tool_error:
                    print(f"❌ Error calling tool: {str(tool_error)}")
                    print(f"🔍 Error type: {type(tool_error).__name__}")
                    
                    result["error"] = str(tool_error)
                    result["error_details"] = {
                        "type": type(tool_error).__name__,
                        "traceback": traceback.format_exc()
                    }
                    
                    # Try to get more details about the error
                    if hasattr(tool_error, '__cause__') and tool_error.__cause__:
                        print(f"🔍 Underlying cause: {str(tool_error.__cause__)}")
                        result["error_details"]["cause"] = str(tool_error.__cause__)
                    
                    if hasattr(tool_error, 'exceptions'):
                        print(f"🔍 Sub-exceptions: {len(tool_error.exceptions)}")
                        sub_exceptions = []
                        for i, sub_exc in enumerate(tool_error.exceptions):
                            print(f"   {i}: {type(sub_exc).__name__}: {str(sub_exc)}")
                            sub_exceptions.append({
                                "type": type(sub_exc).__name__,
                                "message": str(sub_exc),
                                "traceback": "".join(traceback.format_exception(type(sub_exc), sub_exc, sub_exc.__traceback__))
                            })
                        result["error_details"]["sub_exceptions"] = sub_exceptions
                    
                    return result
                    
    except Exception as e:
        print(f"💥 Connection error: {str(e)}")
        print(f"🔍 Error type: {type(e).__name__}")
        
        result["error"] = str(e)
        result["error_details"] = {
            "type": type(e).__name__,
            "traceback": traceback.format_exc()
        }
        
    return result


def ensure_sse_endpoint(url: str) -> str:
    """Ensure URL ends with /sse for SSE transport"""
    if not url.endswith('/sse'):
        url = url.rstrip('/')
        url += '/sse'
    return url


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Test get_all_tools_status tool")
    parser.add_argument("--url", default="http://localhost:8014", 
                        help="Base URL of the MCP server (default: http://localhost:8014)")
    return parser.parse_args()


async def main():
    """Main function"""
    args = parse_args()
    server_url = ensure_sse_endpoint(args.url)
    
    # Run the test
    result = await test_get_all_tools_status(server_url)
    
    # Print final results
    print("\n📊 Test Results:")
    if result["success"]:
        print("✅ Test PASSED")
        print(f"📋 Server: {result['server_info']['name']} v{result['server_info']['version']}")
        print(f"🔧 Tool result type: {type(result['tool_result'])}")
    else:
        print("❌ Test FAILED")
        print(f"💥 Error: {result['error']}")
        print(f"🔍 Error type: {result['error_details']['type'] if result['error_details'] else 'Unknown'}")
        
        # Print detailed error information
        if result["error_details"] and "sub_exceptions" in result["error_details"]:
            print("\n🔍 Detailed Error Information:")
            for i, sub_exc in enumerate(result["error_details"]["sub_exceptions"]):
                print(f"\nSub-exception {i}:")
                print(f"  Type: {sub_exc['type']}")
                print(f"  Message: {sub_exc['message']}")
                print(f"  Traceback:")
                print("  " + "\n  ".join(sub_exc['traceback'].split('\n')))


if __name__ == "__main__":
    asyncio.run(main()) 