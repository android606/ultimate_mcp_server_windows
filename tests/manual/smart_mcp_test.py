#!/usr/bin/env python3
"""
Smart MCP test script that waits for server ready signal
"""

import asyncio
import json
import sys
import time
import subprocess
import threading
import signal
from typing import Optional

try:
    from mcp import ClientSession
    from mcp.client.sse import sse_client
except ImportError:
    print("❌ MCP library not available. Please install with: pip install mcp")
    sys.exit(1)

class SmartMCPTester:
    def __init__(self, host="127.0.0.1", port=8030):  # Changed from 8017 to 8030 to avoid conflicts with production servers
        self.host = host
        self.port = port
        self.server_process: Optional[subprocess.Popen] = None
        self.ready_event = threading.Event()
        self.log_thread: Optional[threading.Thread] = None
        
    def start_server_and_wait(self, timeout=120):
        """Start server and wait for ready signal"""
        print(f"🚀 Starting Ultimate MCP Server on {self.host}:{self.port}...")
        
        # Start the server process
        cmd = [
            sys.executable, "-m", "ultimate_mcp_server", "run",
            "--port", str(self.port),
            "--host", self.host,
            "--debug"
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
        
        # Start log monitoring thread
        self.log_thread = threading.Thread(
            target=self._monitor_server_logs,
            daemon=True
        )
        self.log_thread.start()
        
        # Wait for ready signal
        print(f"⏳ Waiting for server ready signal (timeout: {timeout}s)...")
        if self.ready_event.wait(timeout):
            print("✅ Server is ready for requests!")
            return True
        else:
            print("❌ Server startup timed out")
            self.stop_server()
            return False
    
    def _monitor_server_logs(self):
        """Monitor server logs for ready signal"""
        if not self.server_process:
            return
            
        try:
            # Monitor stderr for the ready message
            for line in iter(self.server_process.stderr.readline, ''):
                print(f"[SERVER] {line.strip()}")
                
                # Look for the ready signal
                if "🚀 SERVER READY FOR REQUESTS" in line:
                    print("🎯 Ready signal detected!")
                    self.ready_event.set()
                    break
                    
                # Check if process died
                if self.server_process.poll() is not None:
                    print("❌ Server process exited unexpectedly")
                    break
                    
        except Exception as e:
            print(f"❌ Error monitoring logs: {e}")
    
    def stop_server(self):
        """Stop the server process"""
        if self.server_process:
            print("🛑 Stopping server...")
            self.server_process.terminate()
            try:
                self.server_process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                print("⚠️  Server didn't stop gracefully, killing...")
                self.server_process.kill()
                self.server_process.wait()
            self.server_process = None
    
    async def test_mcp_functionality(self):
        """Test MCP server functionality"""
        print("\n🧪 Testing MCP Server Functionality")
        print("=" * 50)
        
        session_url = f"http://{self.host}:{self.port}/sse"
        
        try:
            async with sse_client(session_url) as (read, write):
                print("✅ SSE connection established")
                
                async with ClientSession(read, write) as session:
                    print("✅ MCP session created")
                    
                    # Initialize session
                    await session.initialize()
                    print("✅ MCP session initialized")
                    
                    # Test 1: List tools
                    print("\n📋 Testing list_tools...")
                    tools = await session.list_tools()
                    print(f"✅ Found {len(tools.tools)} tools")
                    
                    # Test 2: Call echo tool
                    print("\n🔄 Testing echo tool...")
                    result = await session.call_tool("echo", {"message": "Test message"})
                    if result.isError:
                        print(f"❌ Echo failed: {result.error}")
                        return False
                    else:
                        print("✅ Echo tool successful")
                    
                    # Test 3: Test provider status
                    print("\n🔍 Testing get_provider_status...")
                    try:
                        result = await session.call_tool("get_provider_status", {"random_string": "test"})
                        if not result.isError:
                            print("✅ Provider status retrieved")
                        else:
                            print(f"⚠️  Provider status failed: {result.error}")
                    except Exception as e:
                        print(f"⚠️  Provider status not available: {e}")
                    
                    print("\n🎉 All MCP tests completed successfully!")
                    return True
                    
        except Exception as e:
            print(f"❌ MCP test failed: {e}")
            return False
    
    def run_full_test(self):
        """Run complete test suite"""
        try:
            # Start server and wait for ready
            if not self.start_server_and_wait():
                return False
            
            # Run MCP tests
            success = asyncio.run(self.test_mcp_functionality())
            
            return success
            
        except KeyboardInterrupt:
            print("\n⚠️  Test interrupted by user")
            return False
        except Exception as e:
            print(f"❌ Test failed: {e}")
            return False
        finally:
            self.stop_server()

def main():
    """Main test function"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Smart MCP Server Tester")
    parser.add_argument("--host", default="127.0.0.1", help="Server host")
    parser.add_argument("--port", type=int, default=8030, help="Server port")
    
    args = parser.parse_args()
    
    tester = SmartMCPTester(host=args.host, port=args.port)
    
    # Handle Ctrl+C gracefully
    def signal_handler(sig, frame):
        print("\n⚠️  Received interrupt signal, stopping...")
        tester.stop_server()
        sys.exit(1)
    
    signal.signal(signal.SIGINT, signal_handler)
    
    success = tester.run_full_test()
    
    if success:
        print("\n✅ All tests passed! MCP server is working correctly.")
        sys.exit(0)
    else:
        print("\n❌ Some tests failed.")
        sys.exit(1)

if __name__ == "__main__":
    main() 