#!/usr/bin/env python3
"""
Setup script for Ultimate MCP Server.

This script creates a virtual environment with the custom EnvBuilder
and installs the required packages.
"""

import os
import sys
import platform
import subprocess
from pathlib import Path

# Ensure we can import from the tests directory
sys.path.insert(0, str(Path(__file__).parent))

# Import our custom EnvBuilder
from tests.custom_venv import create_ultimate_mcp_venv


def main():
    """Create a virtual environment and install required packages."""
    print("Setting up Ultimate MCP Server development environment...")
    
    # Check if user should use a shell-specific script instead
    if platform.system() == 'Windows':
        print("\nNOTE: For better Windows compatibility, consider using one of these scripts instead:")
        print("  - setup_venv_windows.bat      (for Command Prompt)")
        print("  - setup_venv_powershell.ps1   (for PowerShell)")
        print("  - setup_venv_gitbash.sh       (for Git Bash)\n")
    
    # Determine the virtual environment directory
    venv_dir = Path('.venv')
    
    # Create the virtual environment with our custom builder
    print(f"Creating virtual environment in {venv_dir}...")
    env_path = create_ultimate_mcp_venv(
        venv_dir,
        system_site_packages=False,
        clear=False,  # Don't clear if it exists
        with_pip=True
    )
    
    # Get the pip executable path
    if platform.system() == 'Windows':
        python_exe = env_path / 'Scripts' / 'python.exe'
        pip_exe = env_path / 'Scripts' / 'pip.exe'
    else:
        python_exe = env_path / 'bin' / 'python'
        pip_exe = env_path / 'bin' / 'pip'
    
    # Install the package in development mode
    print("Installing Ultimate MCP Server in development mode...")
    try:
        subprocess.check_call([
            str(pip_exe), 'install', '-e', '.[test]'
        ])
        print("✅ Installation successful!")
    except subprocess.CalledProcessError as e:
        print(f"❌ Installation failed: {e}")
        return 1
    
    # Print activation instructions
    print("\n✨ Setup complete! ✨")
    print("\nTo activate the environment:")
    if platform.system() == 'Windows':
        print(f"    {env_path}\\Scripts\\activate.bat  (for cmd.exe)")
        print(f"    {env_path}\\Scripts\\Activate.ps1  (for PowerShell)")
    else:
        print(f"    source {env_path}/bin/activate  (for bash/zsh)")
    
    print("\nTo run the server:")
    print("    python -m ultimate_mcp_server run --port 8014 --host 127.0.0.1 --debug")
    
    print("\nTo run tests:")
    print("    python -m pytest tests/test_server_startup.py -v")
    print("    python -m pytest tests/environment/ -v")
    
    return 0


if __name__ == '__main__':
    sys.exit(main()) 