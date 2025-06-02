import asyncio
import os
import platform
import subprocess
import sys
import threading
import time
from pathlib import Path

class MCPServerFixture:
    """Test fixture for managing MCP server lifecycle"""
    
    def __init__(self, host="127.0.0.1", port=8026):
        self.host = host
        self.port = port
        self.server_process = None
        self.ready_event = threading.Event()
        self.log_thread = None
        self.startup_logs = []
        
    async def start_and_wait(self, timeout=120):
        """Start server and wait for ready signal"""
        print(f"🚀 Starting test server on {self.host}:{self.port}")
        
        # Get the virtual environment Python executable
        venv_python = self._get_venv_python_path()
        print(f"Using Python interpreter: {venv_python}")
        
        # Start server process with the correct Python and environment flags
        cmd = [
            venv_python, "-m", "ultimate_mcp_server", "run",
            "--port", str(self.port),
            "--host", self.host,
            "--debug",
            "--load-all-tools",
            "--skip-env-check",  # Skip environment validation
            "--force"            # Force server to start despite issues
        ]
        
        self.server_process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding='utf-8',
            errors='replace',
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
    
    def _get_venv_python_path(self):
        """Get the path to the virtual environment Python executable based on platform"""
        # If we're already in a virtual environment, use the current Python
        if self._is_in_virtualenv():
            return sys.executable
            
        # Find the project root (current directory)
        project_root = Path.cwd()
        
        # Common virtual environment directory names
        venv_dirs = ['.venv', 'venv', 'env', '.env']
        
        # Try to find the venv directory in project root
        venv_path = None
        for venv_dir in venv_dirs:
            path = project_root / venv_dir
            if path.exists() and path.is_dir():
                # Check for common signs of a virtual environment
                if (path / 'pyvenv.cfg').exists():
                    venv_path = path
                    break
        
        if not venv_path:
            # Fallback to the current Python interpreter if venv not found
            print("⚠️ Virtual environment not found, using current Python interpreter")
            return sys.executable
        
        # Build the path to Python executable based on platform
        if platform.system() == 'Windows':
            python_path = venv_path / 'Scripts' / 'python.exe'
        else:  # Linux, macOS, etc.
            python_path = venv_path / 'bin' / 'python'
        
        # Verify the executable exists
        if not python_path.exists():
            print(f"⚠️ Python executable not found at {python_path}, falling back to current interpreter")
            return sys.executable
            
        return str(python_path.absolute())
    
    def _is_in_virtualenv(self):
        """Check if we're already in a virtual environment"""
        # Multiple checks for different ways to detect virtual environments
        return (
            hasattr(sys, 'real_prefix') or 
            (hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix) or
            os.path.exists(os.path.join(sys.prefix, 'pyvenv.cfg'))
        )
    
    def _monitor_logs(self):
        """Monitor server logs for ready signal"""
        if not self.server_process:
            return
            
        try:
            for line in iter(self.server_process.stderr.readline, ''):
                line_str = line.strip()
                self.startup_logs.append(line_str)
                print(f"[SERVER] {line_str}")
                
                # Check for Application startup complete message
                if "Application startup complete" in line_str:
                    print("💡 Application startup complete detected!")
                    self.ready_event.set()
                    break
                    
                # Check for old-style ready message as a fallback
                if "SERVER READY FOR REQUESTS" in line_str:
                    print("💡 SERVER READY FOR REQUESTS detected!")
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

async def test_server_startup():
    """Test that server starts and reports ready status"""
    server = MCPServerFixture(port=8026)
    try:
        await server.start_and_wait()
        
        # Server is ready if we got here
        print("✅ Test passed: Server started successfully!")
        
    finally:
        await server.stop()

if __name__ == "__main__":
    # Run the test
    asyncio.run(test_server_startup()) 