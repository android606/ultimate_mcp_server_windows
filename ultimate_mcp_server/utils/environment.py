#!/usr/bin/env python3
"""
Environment validation utilities for Ultimate MCP Server.

This module provides functions to validate that the correct virtual environment
is activated and all required dependencies are available.
"""

import os
import sys
import subprocess
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import importlib.util


def is_virtual_environment() -> bool:
    """
    Check if we're running in a virtual environment.
    
    Returns:
        bool: True if in virtual environment, False otherwise
    """
    # Method 1: Check if sys.prefix != sys.base_prefix (works for venv, virtualenv)
    if sys.prefix != sys.base_prefix:
        return True
    
    # Method 2: Check for VIRTUAL_ENV environment variable
    if os.environ.get('VIRTUAL_ENV'):
        return True
    
    # Method 3: Check for conda environment
    if os.environ.get('CONDA_DEFAULT_ENV'):
        return True
    
    return False


def get_virtual_env_info() -> Dict[str, Optional[str]]:
    """
    Get information about the current virtual environment.
    
    Returns:
        Dict containing environment information
    """
    info = {
        'is_virtual_env': is_virtual_environment(),
        'virtual_env_path': os.environ.get('VIRTUAL_ENV'),
        'conda_env': os.environ.get('CONDA_DEFAULT_ENV'),
        'python_prefix': sys.prefix,
        'python_base_prefix': sys.base_prefix,
        'python_executable': sys.executable,
    }
    
    # Try to determine environment name
    env_name = None
    if info['virtual_env_path']:
        env_name = os.path.basename(info['virtual_env_path'])
    elif info['conda_env']:
        env_name = info['conda_env']
    
    info['env_name'] = env_name
    return info


def find_project_virtual_env() -> Optional[Path]:
    """
    Try to find the project's virtual environment directory.
    
    Returns:
        Path to virtual environment if found, None otherwise
    """
    # Get the project root (where this file is located)
    current_file = Path(__file__)
    project_root = current_file.parent.parent.parent  # Go up from utils/environment.py
    
    # Common virtual environment directory names
    venv_names = ['.venv', 'venv', 'env', '.env']
    
    for venv_name in venv_names:
        venv_path = project_root / venv_name
        if venv_path.exists() and venv_path.is_dir():
            # Check if it looks like a virtual environment
            if (venv_path / 'pyvenv.cfg').exists() or \
               (venv_path / 'Scripts' / 'activate').exists() or \
               (venv_path / 'bin' / 'activate').exists():
                return venv_path
    
    return None


def check_required_packages(required_packages: List[str]) -> Dict[str, bool]:
    """
    Check if required packages are installed and importable.
    
    Args:
        required_packages: List of package names to check
    
    Returns:
        Dict mapping package names to availability status
    """
    results = {}
    
    for package in required_packages:
        try:
            spec = importlib.util.find_spec(package)
            results[package] = spec is not None
        except (ImportError, ModuleNotFoundError, ValueError):
            results[package] = False
    
    return results


def get_ultimate_mcp_requirements() -> List[str]:
    """
    Get the list of required packages for Ultimate MCP Server.
    
    Returns:
        List of required package names
    """
    return [
        'mcp',
        'fastapi',
        'uvicorn',
        'pydantic',
        'aiofiles',
        'httpx',
        'anthropic',
        'openai',
        'pytest',  # For testing
        'pytest_asyncio',  # For async testing
    ]


def validate_environment(strict: bool = False) -> Tuple[bool, List[str], List[str]]:
    """
    Validate the current environment for running Ultimate MCP Server.
    
    Args:
        strict: If True, require virtual environment activation
    
    Returns:
        Tuple of (is_valid, list_of_errors, list_of_warnings)
    """
    errors = []
    warnings = []
    
    # Check if in virtual environment
    env_info = get_virtual_env_info()
    if strict and not env_info['is_virtual_env']:
        errors.append("Not running in a virtual environment")
    
    # Check required packages
    required_packages = get_ultimate_mcp_requirements()
    package_status = check_required_packages(required_packages)
    
    missing_packages = [pkg for pkg, available in package_status.items() if not available]
    if missing_packages:
        errors.append(f"Missing required packages: {', '.join(missing_packages)}")
    
    # Check Python version - Requires Python 3.13+
    # Now we treat this as a warning instead of an error
    required_version = (3, 13)
    current_version = sys.version_info[:2]
    if current_version < required_version:
        warnings.append(f"Python {required_version[0]}.{required_version[1]}+ recommended, but running {current_version[0]}.{current_version[1]} (may cause compatibility issues)")
    
    return len(errors) == 0, errors, warnings


def print_environment_status(verbose: bool = False) -> None:
    """
    Print current environment status information.
    
    Args:
        verbose: If True, print detailed information
    """
    env_info = get_virtual_env_info()
    is_valid, errors, warnings = validate_environment()
    
    print("🔍 Ultimate MCP Server Environment Status")
    print("=" * 50)
    
    # Virtual environment status
    if env_info['is_virtual_env']:
        print(f"✅ Virtual Environment: Active ({env_info.get('env_name', 'Unknown')})")
        if verbose and env_info['virtual_env_path']:
            print(f"   Path: {env_info['virtual_env_path']}")
    else:
        print("❓ Virtual Environment: Not Active")
        
        # Try to find project venv
        project_venv = find_project_virtual_env()
        if project_venv:
            print(f"💡 Found project virtual env at: {project_venv}")
            print("   To activate:")
            if os.name == 'nt':  # Windows
                print(f"   {project_venv / 'Scripts' / 'activate.bat'}")
            else:  # Unix/Linux/Mac
                print(f"   source {project_venv / 'bin' / 'activate'}")
    
    # Python version
    required_version = (3, 13)
    current_version = sys.version_info[:2]
    if current_version >= required_version:
        print(f"✅ Python Version: {sys.version.split()[0]}")
    else:
        print(f"⚠️ Python Version: {sys.version.split()[0]} (Recommended: {required_version[0]}.{required_version[1]}+)")
    
    if verbose:
        print(f"   Executable: {sys.executable}")
    
    # Package availability
    required_packages = get_ultimate_mcp_requirements()
    package_status = check_required_packages(required_packages)
    
    available_packages = [pkg for pkg, available in package_status.items() if available]
    missing_packages = [pkg for pkg, available in package_status.items() if not available]
    
    if available_packages:
        print(f"✅ Available Packages ({len(available_packages)}): {', '.join(available_packages)}")
    
    if missing_packages:
        print(f"❌ Missing Packages ({len(missing_packages)}): {', '.join(missing_packages)}")
        print("   Install with: pip install " + " ".join(missing_packages))
    
    # Overall status
    print("\n" + "=" * 50)
    if is_valid and not warnings:
        print("✅ Environment Status: Ready for Ultimate MCP Server")
    elif is_valid:
        print("⚠️ Environment Status: Ready with warnings")
        for warning in warnings:
            print(f"   • {warning}")
    else:
        print("❌ Environment Status: Issues detected")
        for error in errors:
            print(f"   • {error}")
        if warnings:
            print("\n   Warnings:")
            for warning in warnings:
                print(f"   • {warning}")


def suggest_environment_setup() -> None:
    """Print suggestions for setting up the environment correctly."""
    project_venv = find_project_virtual_env()
    
    print("\n🚀 Environment Setup Suggestions")
    print("=" * 50)
    
    if not is_virtual_environment():
        if project_venv:
            print("1. Activate existing virtual environment:")
            if os.name == 'nt':  # Windows
                print(f"   {project_venv / 'Scripts' / 'activate.bat'}")
                print("   # or in PowerShell:")
                print(f"   {project_venv / 'Scripts' / 'Activate.ps1'}")
            else:  # Unix/Linux/Mac
                print(f"   source {project_venv / 'bin' / 'activate'}")
        else:
            print("1. Create and activate a virtual environment:")
            print("   python -m venv .venv")
            print(r"   .venv\Scripts\activate.bat   # Windows")
            print("   source .venv/bin/activate   # Linux/Mac")
    
    # Check Python version
    required_version = (3, 13)
    current_version = sys.version_info[:2]
    if current_version < required_version:
        print("\n2. Python Version Recommendation:")
        print(f"   Current Python: {current_version[0]}.{current_version[1]}")
        print(f"   Recommended: {required_version[0]}.{required_version[1]}+")
        print("   While the application may work with your current Python version,")
        print("   for best results we recommend using Python 3.13+ in a virtual environment.")
        print("   Download from: https://www.python.org/downloads/")
        print("")
        print("   After installing Python 3.13+, create a new virtual environment with it:")
        print("   python3.13 -m venv .venv-py313")
        print(r"   .venv-py313\Scripts\activate.bat   # Windows")
        print("   source .venv-py313/bin/activate   # Linux/Mac")
    
    print("\n3. Install Ultimate MCP Server in development mode:")
    print("   pip install -e .")
    
    print("\n4. Install additional development dependencies:")
    print("   pip install pytest pytest-asyncio")
    
    print("\n5. Verify installation:")
    print("   python -c \"import ultimate_mcp_server; print('✅ Import successful')\"")


if __name__ == "__main__":
    """Allow running this module directly for environment checking."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Check Ultimate MCP Server environment")
    parser.add_argument("--verbose", "-v", action="store_true", help="Show detailed information")
    parser.add_argument("--strict", "-s", action="store_true", help="Require virtual environment")
    parser.add_argument("--suggest", action="store_true", help="Show setup suggestions")
    
    args = parser.parse_args()
    
    print_environment_status(verbose=args.verbose)
    
    if args.suggest or not validate_environment(strict=args.strict)[0]:
        suggest_environment_setup() 