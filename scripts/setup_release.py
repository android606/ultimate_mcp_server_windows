#!/usr/bin/env python3
"""Release automation script for Ultimate MCP Server."""

import re
import subprocess
import sys
from pathlib import Path
from typing import Optional

def get_current_version() -> str:
    """Get current version from __init__.py."""
    init_file = Path("ultimate_mcp_server/__init__.py")
    content = init_file.read_text()
    match = re.search(r'__version__ = ["\']([^"\']+)["\']', content)
    if not match:
        raise ValueError("Could not find version in __init__.py")
    return match.group(1)

def bump_version(current_version: str, bump_type: str) -> str:
    """Bump version based on type (major, minor, patch)."""
    major, minor, patch = map(int, current_version.split('.'))
    
    if bump_type == "major":
        return f"{major + 1}.0.0"
    elif bump_type == "minor":
        return f"{major}.{minor + 1}.0"
    elif bump_type == "patch":
        return f"{major}.{minor}.{patch + 1}"
    else:
        raise ValueError("bump_type must be 'major', 'minor', or 'patch'")

def update_version_files(new_version: str):
    """Update version in all relevant files."""
    # Update __init__.py
    init_file = Path("ultimate_mcp_server/__init__.py")
    content = init_file.read_text()
    content = re.sub(
        r'__version__ = ["\'][^"\']+["\']',
        f'__version__ = "{new_version}"',
        content
    )
    init_file.write_text(content)
    
    # Update pyproject.toml
    pyproject_file = Path("pyproject.toml")
    content = pyproject_file.read_text()
    content = re.sub(
        r'version = ["\'][^"\']+["\']',
        f'version = "{new_version}"',
        content
    )
    pyproject_file.write_text(content)
    
    # Update CLI version
    cli_file = Path("ultimate_mcp_server/cli/typer_cli.py")
    content = cli_file.read_text()
    content = re.sub(
        r'__version__ = ["\'][^"\']+["\']',
        f'__version__ = "{new_version}"',
        content
    )
    cli_file.write_text(content)

def run_command(cmd: list[str]) -> bool:
    """Run a command and return success status."""
    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        print(f"✅ {' '.join(cmd)}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ {' '.join(cmd)} failed: {e.stderr}")
        return False

def main():
    """Main release script."""
    if len(sys.argv) != 2 or sys.argv[1] not in ["major", "minor", "patch"]:
        print("Usage: python setup_release.py [major|minor|patch]")
        sys.exit(1)
    
    bump_type = sys.argv[1]
    
    # Get current version
    current_version = get_current_version()
    new_version = bump_version(current_version, bump_type)
    
    print(f"🚀 Preparing release: {current_version} → {new_version}")
    
    # Update version files
    print("📝 Updating version files...")
    update_version_files(new_version)
    
    # Run tests
    print("🧪 Running tests...")
    if not run_command(["python", "-m", "pytest"]):
        print("❌ Tests failed, aborting release")
        sys.exit(1)
    
    # Build package
    print("📦 Building package...")
    if not run_command(["python", "-m", "build"]):
        print("❌ Build failed, aborting release")
        sys.exit(1)
    
    # Git operations
    print("📋 Creating git commit and tag...")
    if not run_command(["git", "add", "."]):
        sys.exit(1)
    if not run_command(["git", "commit", "-m", f"Release v{new_version}"]):
        sys.exit(1)
    if not run_command(["git", "tag", f"v{new_version}"]):
        sys.exit(1)
    
    print(f"✅ Release v{new_version} prepared!")
    print(f"📤 To publish, run:")
    print(f"   git push origin main --tags")
    print(f"   twine upload dist/*")

if __name__ == "__main__":
    main() 