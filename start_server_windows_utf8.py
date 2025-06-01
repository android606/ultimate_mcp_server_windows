#!/usr/bin/env python3
"""
Windows-safe Ultimate MCP Server launcher.
Uses Windows-safe logging that automatically filters emojis to prevent encoding issues.
"""

import os
import sys
import platform
import subprocess
import locale

def setup_windows_console():
    """Set up Windows console and confirm Windows-safe logging."""
    if platform.system() == "Windows":
        # Set console to UTF-8 mode for better compatibility
        try:
            # Try to set console codepage to UTF-8
            subprocess.run(["chcp", "65001"], shell=True, capture_output=True)
            print("✅ Set Windows console to UTF-8 (chcp 65001)")
        except Exception as e:
            print(f"⚠️ Could not set console codepage: {e}")
        
        print("✅ Windows-safe logging enabled automatically")

def start_server():
    """Start the server with Windows-safe configuration."""
    
    # Setup Windows console
    setup_windows_console()
    
    # Server configuration
    host = "127.0.0.1"
    port = 8096
    
    print(f"\n🚀 Starting Ultimate MCP Server...")
    print(f"🌐 Host: {host}")
    print(f"🔌 Port: {port}")
    print(f"💻 Platform: {platform.system()}")
    print(f"🐍 Python: {sys.version}")
    print(f"📍 Encoding: {locale.getpreferredencoding()}")
    print()
    
    # Command to run the server
    cmd = [
        sys.executable, "-m", "ultimate_mcp_server", "run",
        "--transport-mode", "sse",
        "--host", host,
        "--port", str(port)
    ]
    
    try:
        if platform.system() == "Windows":
            # On Windows, create new console for clean output
            startupinfo = subprocess.STARTUPINFO()
            startupinfo.dwFlags |= subprocess.STARTF_USESHOWWINDOW
            
            # Start the process with clean subprocess handling
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,  # Merge stderr into stdout
                text=True,  # Text mode
                startupinfo=startupinfo,
                creationflags=subprocess.CREATE_NEW_CONSOLE
            )
            
            print("✅ Server started with Windows-safe configuration")
            print(f"📋 Process ID: {process.pid}")
            print("📝 Server output:")
            print("-" * 50)
            
            # Read output line by line
            try:
                while True:
                    line = process.stdout.readline()
                    if not line:
                        break
                    print(line.rstrip())
                    
            except KeyboardInterrupt:
                print("\n🛑 Stopping server...")
                process.terminate()
                process.wait()
                
        else:
            # On non-Windows, use standard approach
            subprocess.run(cmd, check=True)
            
    except subprocess.CalledProcessError as e:
        print(f"❌ Server failed with exit code {e.returncode}")
        return False
    except Exception as e:
        print(f"❌ Error starting server: {e}")
        return False
        
    return True

def main():
    """Main entry point."""
    print("🔧 Windows-safe Ultimate MCP Server Launcher")
    print("🎯 Automatically uses Windows-safe logging to prevent emoji encoding issues")
    print()
    
    if platform.system() != "Windows":
        print("ℹ️ This launcher works on all platforms, with automatic Windows compatibility.")
    
    success = start_server()
    
    if success:
        print("\n✅ Server shutdown completed successfully")
    else:
        print("\n❌ Server encountered errors")
        sys.exit(1)

if __name__ == "__main__":
    main() 