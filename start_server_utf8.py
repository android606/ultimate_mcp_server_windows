#!/usr/bin/env python3
"""
Start Ultimate MCP Server with proper UTF-8 encoding on Windows.
This avoids the Unicode decode errors that can occur with subprocess.
"""

import os
import sys
import subprocess

def main():
    """Start the server with proper encoding settings."""
    
    # Set environment variables for UTF-8 handling
    env = os.environ.copy()
    env['PYTHONIOENCODING'] = 'utf-8'
    env['PYTHONUTF8'] = '1'
    env['PYTHONLEGACYWINDOWSFSENCODING'] = 'utf-8'
    
    # Command to run the server
    cmd = [
        sys.executable, '-m', 'ultimate_mcp_server', 'run',
        '--transport-mode', 'sse',
        '--host', '127.0.0.1',
        '--port', '8092'
    ]
    
    print("🚀 Starting Ultimate MCP Server with UTF-8 encoding...")
    print(f"📋 Command: {' '.join(cmd)}")
    print("🔧 Environment variables set:")
    print(f"   PYTHONIOENCODING: {env.get('PYTHONIOENCODING')}")
    print(f"   PYTHONUTF8: {env.get('PYTHONUTF8')}")
    print(f"   PYTHONLEGACYWINDOWSFSENCODING: {env.get('PYTHONLEGACYWINDOWSFSENCODING')}")
    print()
    
    # Start the server
    try:
        subprocess.run(cmd, env=env, check=True)
    except KeyboardInterrupt:
        print("\n🛑 Server stopped by user")
    except subprocess.CalledProcessError as e:
        print(f"❌ Server failed with exit code {e.returncode}")
        sys.exit(e.returncode)

if __name__ == "__main__":
    main() 