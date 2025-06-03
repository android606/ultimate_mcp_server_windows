#!/usr/bin/env python
"""
Ultimate MCP Server - Environment Verification Script

This script checks that your development environment is correctly set up
for working on the Ultimate MCP Server project.

It verifies:
1. Python version (3.13+)
2. Virtual environment activation
3. Package installation in development mode
4. Required packages for testing
5. PATH configuration

Run this script from the project root with:
python verify_environment.py
"""

import os
import sys
import platform
import subprocess
import shutil
import importlib.util
from pathlib import Path

# ASCII art project logo - optional eye candy
LOGO = r"""
 _   _ _ _   _                 _         __  __ ___ ___   ___ 
| | | | | |_(_)_ __ ___   __ _| |_ ___  |  \/  / __| _ \ / __| ___ _ ___ _____ _ _ 
| |_| | |  _| | '_ ` _ \ / _` | __/ _ \ | |\/| \__ \  _/ \__ \/ -_) '_\ V / -_) '_|
 \___/|_|\__|_| | | | | | (_| | ||  __/ |_|  |_|___/_|   |___/\___|_|  \_/\___|_|  
                |_| |_| |_|\__,_|\__\___|                                          
"""

# Constants
REQUIRED_PYTHON_VERSION = (3, 13)
PROJECT_NAME = "ultimate_mcp_server"
REQUIRED_PACKAGES = [
    "pytest",
    "pytest-asyncio",  
    "pytest-cov",      
    "pytest-mock",     
    "anyio",           
]

def print_section(title):
    """Print a section header."""
    print("\n" + "=" * 80)
    print(f" {title} ".center(80, "="))
    print("=" * 80)

def print_result(test, result, message=""):
    """Print a test result."""
    status = "✅ PASS" if result else "❌ FAIL"
    print(f"{status}: {test}")
    if message and not result:
        print(f"       {message}")
    return result

def get_python_paths():
    """Get all Python executables in PATH."""
    paths = []
    path_var = os.environ.get("PATH", "")
    
    for directory in path_var.split(os.pathsep):
        python_exe = os.path.join(directory, "python.exe" if sys.platform == "win32" else "python")
        if os.path.isfile(python_exe) and os.access(python_exe, os.X_OK):
            paths.append(python_exe)
            
    return paths

def check_python_version():
    """Check if the Python version is correct."""
    current_version = sys.version_info[:2]
    result = current_version >= REQUIRED_PYTHON_VERSION
    message = f"Required: Python {REQUIRED_PYTHON_VERSION[0]}.{REQUIRED_PYTHON_VERSION[1]}+, Found: {platform.python_version()}"
    return print_result("Python version", result, message)

def check_virtualenv():
    """Check if running in a virtual environment."""
    in_virtualenv = hasattr(sys, 'real_prefix') or (hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix)
    message = "Not running in a virtual environment"
    return print_result("Virtual environment", in_virtualenv, message)

def check_project_installed():
    """Check if the project is installed in development mode."""
    project_installed = False
    correct_path = False
    
    try:
        import ultimate_mcp_server
        project_installed = True
        
        # Check if it's an editable installation by checking the path
        project_path = os.path.dirname(os.path.abspath(__file__))
        module_path = os.path.dirname(ultimate_mcp_server.__file__)
        
        # Check if the installation is in the same directory structure
        correct_path = os.path.dirname(module_path) == project_path
        
        if not correct_path:
            message = f"Project is installed, but not in development mode.\nProject path: {project_path}\nModule path: {os.path.dirname(module_path)}"
        else:
            message = ""
    except ImportError:
        message = "Project is not installed"
    
    return print_result("Project installation", project_installed and correct_path, message)

def check_required_packages():
    """Check if required packages are installed."""
    missing = []
    for package in REQUIRED_PACKAGES:
        if importlib.util.find_spec(package.replace('-', '_')) is None:
            missing.append(package)
    
    result = len(missing) == 0
    message = f"Missing packages: {', '.join(missing)}" if missing else ""
    return print_result("Required packages", result, message)

def check_path_config():
    """Check PATH configuration."""
    # Get Python executable that would be used when typing 'python' in the shell
    current_python = sys.executable
    
    # Check if it's in the virtual environment
    venv_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '.venv'))
    is_venv_python = os.path.normpath(current_python).startswith(os.path.normpath(venv_path))
    
    # Get all Python executables in PATH
    all_pythons = get_python_paths()
    
    # Check if the virtual environment Python is first in PATH
    venv_python_first = False
    if all_pythons:
        first_python = os.path.normpath(all_pythons[0])
        venv_python_first = first_python.startswith(os.path.normpath(venv_path))
    
    result = is_venv_python and venv_python_first
    
    message = ""
    if not is_venv_python:
        message += f"Current Python is not from the virtual environment: {current_python}\n"
    if not venv_python_first and all_pythons:
        message += f"Virtual environment Python is not first in PATH. First Python: {all_pythons[0]}\n"
    
    if all_pythons:
        print(f"Python interpreters in PATH order:")
        for i, path in enumerate(all_pythons[:3], 1):
            in_venv = os.path.normpath(path).startswith(os.path.normpath(venv_path))
            status = "[VENV]" if in_venv else "     "
            print(f"  {i}. {status} {path}")
        if len(all_pythons) > 3:
            print(f"  ... and {len(all_pythons) - 3} more")
    
    return print_result("PATH configuration", result, message.strip())

def print_instructions():
    """Print instructions for fixing environment issues."""
    print_section("Instructions for Setup")
    
    print("To set up your environment correctly:")
    
    print("\n1. Ensure Python 3.13+ is installed")
    print("   https://www.python.org/downloads/")
    
    print("\n2. Create a virtual environment:")
    print("   python -m venv .venv")
    
    print("\n3. Activate the virtual environment:")
    if sys.platform == "win32":
        print("   • Command Prompt: .venv\\Scripts\\activate.bat")
        print("   • PowerShell: .venv\\Scripts\\Activate.ps1")
        print("   • Git Bash: source .venv/Scripts/activate")
    else:
        print("   source .venv/bin/activate")
    
    print("\n4. Install the package in development mode with test dependencies:")
    print("   pip install -e \".[test]\"")
    
    print("\n5. Verify environment setup:")
    print("   python verify_environment.py")

def main():
    """Run all checks and print results."""
    print(LOGO)
    print("\nUltimate MCP Server - Environment Verification")
    
    print_section("Environment Checks")
    
    # Run all checks
    python_ok = check_python_version()
    venv_ok = check_virtualenv()
    project_ok = check_project_installed()
    packages_ok = check_required_packages()
    path_ok = check_path_config()
    
    # Summarize results
    print_section("Results Summary")
    
    all_ok = python_ok and venv_ok and project_ok and packages_ok and path_ok
    
    if all_ok:
        print("✅ All checks passed! Your environment is correctly set up.")
    else:
        print("❌ Some checks failed. Please fix the issues above.")
        print_instructions()
    
    return 0 if all_ok else 1

if __name__ == "__main__":
    sys.exit(main()) 