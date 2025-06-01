#!/usr/bin/env python3
"""
Simple Ultimate MCP Server launcher for debugging Unicode issues.
"""

import subprocess
import sys
import os

print("DEBUG: Starting minimal server launcher...")

# Try to find Python in the virtual environment
venv_python = ".venv/Scripts/python.exe"
system_python = "python"

if os.path.exists(venv_python):
    python_exe = venv_python
    print(f"DEBUG: Using venv Python: {python_exe}")
else:
    python_exe = system_python
    print(f"DEBUG: Using system Python: {python_exe}")

# Basic command
cmd = [python_exe, "-m", "ultimate_mcp_server", "run", "--transport-mode", "sse"]

print(f"DEBUG: Command: {' '.join(cmd)}")
print("DEBUG: Starting subprocess...")

try:
    result = subprocess.run(cmd, capture_output=False, text=True, encoding="utf-8", errors="replace")
    print(f"DEBUG: Process exited with code: {result.returncode}")
except Exception as e:
    print(f"DEBUG: Exception: {e}")
    print(f"DEBUG: Exception type: {type(e)}")

print("DEBUG: Script finished.") 