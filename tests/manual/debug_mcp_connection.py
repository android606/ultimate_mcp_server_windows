#!/usr/bin/env python3
"""
Detailed MCP connection debugging script
"""

import asyncio
import json
import sys
import time
import aiohttp

try:
    from mcp import ClientSession
    from mcp.client.sse import sse_client
except ImportError:
    print("❌ MCP library not available. Please install with: pip install mcp")
    sys.exit(1)

async def debug_sse_connection():
    """Debug SSE connection step by step"""
    print("🔍 Debugging SSE Connection")
    print("=" * 50)
    
    # Step 1: Test basic HTTP connection
    print("Step 1: Testing basic HTTP connection...")
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get("http://localhost:8017/health") as response:
                print(f"✅ Health endpoint: {response.status}")
                text = await response.text()
                print(f"   Response: {text}")
    except Exception as e:
        print(f"❌ Health endpoint failed: {e}")
        return False
    
    # Step 2: Test SSE endpoint accessibility
    print("\nStep 2: Testing SSE endpoint accessibility...")
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get("http://localhost:8017/sse") as response:
                print(f"✅ SSE endpoint status: {response.status}")
                print(f"   Content-Type: {response.headers.get('content-type', 'None')}")
    except Exception as e:
        print(f"❌ SSE endpoint failed: {e}")
        return False
    
    # Step 3: Test MCP SSE client connection
    print("\nStep 3: Testing MCP SSE client connection...")
    try:
        session_url = "http://localhost:8017/sse"
        print(f"   Connecting to: {session_url}")
        
        async with sse_client(session_url) as (read, write):
            print("✅ SSE client connected successfully!")
            
            # Step 4: Test MCP session creation
            print("\nStep 4: Testing MCP session creation...")
            try:
                async with ClientSession(read, write) as mcp_session:
                    print("✅ MCP session created successfully!")
                    
                    # Step 5: Test initialize
                    print("\nStep 5: Testing MCP initialization...")
                    try:
                        init_result = await mcp_session.initialize()
                        print(f"✅ MCP initialized: {init_result}")
                        
                        # Step 6: Test list_tools with timeout
                        print("\nStep 6: Testing list_tools with timeout...")
                        try:
                            # Add timeout to prevent hanging
                            tools = await asyncio.wait_for(mcp_session.list_tools(), timeout=10.0)
                            print(f"✅ Found {len(tools.tools)} tools:")
                            for tool in tools.tools[:3]:
                                print(f"   • {tool.name}")
                            return True
                            
                        except asyncio.TimeoutError:
                            print("❌ list_tools timed out after 10 seconds")
                            return False
                        except Exception as e:
                            print(f"❌ list_tools failed: {type(e).__name__}: {e}")
                            return False
                            
                    except Exception as e:
                        print(f"❌ MCP initialization failed: {type(e).__name__}: {e}")
                        return False
                        
            except Exception as e:
                print(f"❌ MCP session creation failed: {type(e).__name__}: {e}")
                return False
                
    except Exception as e:
        print(f"❌ SSE client failed: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Main debug function"""
    success = asyncio.run(debug_sse_connection())
    if success:
        print("\n✅ All connection tests passed!")
        sys.exit(0)
    else:
        print("\n❌ Connection debugging failed.")
        sys.exit(1)

if __name__ == "__main__":
    main() 