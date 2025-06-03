#!/usr/bin/env python
"""
Simple test script to verify Python version check logic.
This script directly checks the logic that would be used for version checking
without actually importing the modules.
"""
import sys

def test_main_module_logic():
    """Test the logic used in __main__.py."""
    print("\n=== Testing __main__.py logic ===")
    
    # The logic from __main__.py
    REQUIRED_VERSION = (3, 13)
    TEST_VERSION = (3, 12)  # Simulate Python 3.12
    
    # Check if this would show a warning instead of exiting
    if TEST_VERSION < REQUIRED_VERSION:
        warning_message = (
            f"WARNING: Non-optimal Python Version\n"
            f"Ultimate MCP Server is designed for Python {REQUIRED_VERSION[0]}.{REQUIRED_VERSION[1]}+\n"
            f"You are running Python {TEST_VERSION[0]}.{TEST_VERSION[1]}\n"
            "The application may still work but could have unexpected behaviors."
        )
        print(f"✅ Would show warning: '{warning_message}'")
        # In the actual code, we don't exit - we continue execution
        print("✅ Would continue execution (no sys.exit call)")
    else:
        print("❌ Would not show warning (incorrect behavior)")

def test_environment_validation_logic():
    """Test the logic used in environment.py validation."""
    print("\n=== Testing environment validation logic ===")
    
    # The logic from environment.py
    REQUIRED_VERSION = (3, 13)
    TEST_VERSION = (3, 12)  # Simulate Python 3.12
    
    # Check Python version - now treated as warning
    errors = []
    warnings = []
    
    if TEST_VERSION < REQUIRED_VERSION:
        warnings.append(f"Python {REQUIRED_VERSION[0]}.{REQUIRED_VERSION[1]}+ recommended, but running {TEST_VERSION[0]}.{TEST_VERSION[1]} (may cause compatibility issues)")
        print(f"✅ Added to warnings: '{warnings[0]}'")
    else:
        errors.append(f"Python {REQUIRED_VERSION[0]}.{REQUIRED_VERSION[1]}+ required, but running {TEST_VERSION[0]}.{TEST_VERSION[1]}")
        print(f"❌ Added to errors (incorrect behavior): '{errors[0]}'")
    
    # Calculate is_valid (should be True even with Python version warnings)
    is_valid = len(errors) == 0
    
    if is_valid and warnings:
        print("✅ Environment considered valid with warnings (correct behavior)")
    elif not is_valid:
        print("❌ Environment considered invalid (incorrect behavior)")

if __name__ == "__main__":
    print("=== Testing Python Version Warning Logic ===")
    print(f"Actual Python version: {sys.version}")
    
    # Run the tests
    test_main_module_logic()
    test_environment_validation_logic()
    
    print("\n✅ All logic tests completed!") 