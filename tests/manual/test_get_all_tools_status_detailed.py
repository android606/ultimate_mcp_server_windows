#!/usr/bin/env python3
"""
Simple test script to see the raw output of get_all_tools_status tool.
"""

import asyncio
import json
from mcp.client.sse import sse_client
from mcp import ClientSession


async def test_simple():
    """Simple test to see raw tool output"""
    
    server_url = "http://localhost:8015/sse"
    
    print("🔗 Connecting to server...")
    async with sse_client(server_url) as (read, write):
        async with ClientSession(read, write) as session:
            # Initialize
            await session.initialize()
            
            # Call the tool
            print("🔧 Calling get_all_tools_status...")
            result = await session.call_tool("get_all_tools_status", {})
            
            print(f"✅ Tool call completed!")
            print(f"📊 Result type: {type(result)}")
            print(f"📊 Result attributes: {dir(result)}")
            
            if hasattr(result, 'content'):
                print(f"📊 Content type: {type(result.content)}")
                print(f"📊 Content: {result.content}")
                
                if isinstance(result.content, list):
                    print(f"📊 Content length: {len(result.content)}")
                    for i, item in enumerate(result.content):
                        print(f"📊 Item {i} type: {type(item)}")
                        print(f"📊 Item {i} attributes: {dir(item)}")
                        if hasattr(item, 'text'):
                            print(f"📊 Item {i} text: {item.text[:500]}...")  # First 500 chars
                            
                            # Try to parse as JSON
                            try:
                                parsed = json.loads(item.text)
                                print(f"📊 Item {i} parsed JSON keys: {list(parsed.keys())}")
                                if "tools_status" in parsed:
                                    print(f"📊 tools_status length: {len(parsed['tools_status'])}")
                                if "summary" in parsed:
                                    print(f"📊 summary: {parsed['summary']}")
                            except json.JSONDecodeError:
                                print(f"📊 Item {i} text is not valid JSON")
            else:
                print(f"📊 Raw result: {result}")


if __name__ == "__main__":
    asyncio.run(test_simple()) 