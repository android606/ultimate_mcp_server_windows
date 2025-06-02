#!/usr/bin/env python3
"""
Test script for demonstrating Ultimate MCP Server environment validation.

This script tests various scenarios of environment validation.
"""
import subprocess
import sys
import os
from pathlib import Path


def run_command(cmd, expect_failure=False):
    """Run a command and return the result."""
    print(f"\n🔍 Running: {' '.join(cmd)}")
    print("=" * 50)
    
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=30  # 30 second timeout
        )
        
        if result.stdout:
            print(result.stdout)
        if result.stderr:
            print("STDERR:")
            print(result.stderr)
        
        print(f"Exit code: {result.returncode}")
        
        if expect_failure and result.returncode == 0:
            print("⚠️  Expected failure but command succeeded")
        elif not expect_failure and result.returncode != 0:
            print("⚠️  Expected success but command failed")
        else:
            print("✅ Command behaved as expected")
            
        return result
        
    except subprocess.TimeoutExpired:
        print("❌ Command timed out")
        return None
    except Exception as e:
        print(f"❌ Command failed with exception: {e}")
        return None


def main():
    """Test environment validation scenarios."""
    print("🧪 Ultimate MCP Server Environment Validation Tests")
    print("=" * 60)
    
    # Test 1: Basic environment check
    print("\n📋 Test 1: Basic Environment Status")
    run_command([sys.executable, "-m", "ultimate_mcp_server.cli", "env"])
    
    # Test 2: Verbose environment check
    print("\n📋 Test 2: Verbose Environment Status")
    run_command([sys.executable, "-m", "ultimate_mcp_server.cli", "env", "--verbose"])
    
    # Test 3: Check-only mode (should succeed)
    print("\n📋 Test 3: Check-Only Mode")
    run_command([sys.executable, "-m", "ultimate_mcp_server.cli", "env", "--check-only"])
    
    # Test 4: Strict mode (should succeed if in venv)
    print("\n📋 Test 4: Strict Mode (requires virtual environment)")
    run_command([sys.executable, "-m", "ultimate_mcp_server.cli", "env", "--strict"])
    
    # Test 5: Show suggestions
    print("\n📋 Test 5: Environment Setup Suggestions")
    run_command([sys.executable, "-m", "ultimate_mcp_server.cli", "env", "--suggest"])
    
    # Test 6: Test the environment utility directly
    print("\n📋 Test 6: Direct Environment Utility")
    run_command([sys.executable, "-m", "ultimate_mcp_server.utils.environment"])
    
    # Test 7: Test server startup environment validation (dry run)
    print("\n📋 Test 7: Server Startup with Environment Check")
    print("Note: This will test environment validation during server startup")
    print("We'll skip actual server startup for this test")
    
    # We can't easily test the interactive part of server startup, but we can
    # verify the CLI accepts the new parameter
    print("\n📋 Test 8: Server CLI Help (showing new --skip-env-check option)")
    run_command([sys.executable, "-m", "ultimate_mcp_server.cli", "run", "--help"])


if __name__ == "__main__":
    main() 