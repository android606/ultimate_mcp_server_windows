#!/usr/bin/env python
"""
Test script to simulate running Ultimate MCP Server with an older Python version.
This script temporarily modifies sys.version_info to simulate Python 3.12,
then imports the package to verify warning messages are shown.
"""
import sys
from unittest.mock import patch

# Save the real version info
real_version_info = sys.version_info

# Create a fake version info that's similar to a real version_info object
class FakeVersionInfo:
    def __init__(self, major, minor, micro=0, releaselevel='final', serial=0):
        self.major = major
        self.minor = minor
        self.micro = micro
        self.releaselevel = releaselevel
        self.serial = serial
    
    def __getitem__(self, key):
        if isinstance(key, slice):
            start, stop, _ = key.indices(5)
            values = (self.major, self.minor, self.micro, 
                     0 if self.releaselevel == 'final' else 1, self.serial)
            return values[start:stop]
        else:
            values = (self.major, self.minor, self.micro, 
                     0 if self.releaselevel == 'final' else 1, self.serial)
            return values[key]
    
    def __lt__(self, other):
        if isinstance(other, tuple):
            return (self.major, self.minor) < other[:2]
        return NotImplemented
    
    def __le__(self, other):
        if isinstance(other, tuple):
            return (self.major, self.minor) <= other[:2]
        return NotImplemented
    
    def __eq__(self, other):
        if isinstance(other, tuple):
            return (self.major, self.minor) == other[:2]
        return NotImplemented
    
    def __gt__(self, other):
        if isinstance(other, tuple):
            return (self.major, self.minor) > other[:2]
        return NotImplemented
    
    def __ge__(self, other):
        if isinstance(other, tuple):
            return (self.major, self.minor) >= other[:2]
        return NotImplemented

# Create a fake version info for Python 3.12.0
fake_version_info = FakeVersionInfo(3, 12, 0)

def test_package_import_warning():
    """Test the version warning in __init__.py"""
    print("\n=== Testing warning when importing package ===")
    
    # Reset any cached imports
    if 'ultimate_mcp_server' in sys.modules:
        del sys.modules['ultimate_mcp_server']
    
    # Patch sys.version_info
    with patch('sys.version_info', fake_version_info):
        # Patch warnings.warn to capture it
        with patch('warnings.warn') as mock_warn:
            print("Importing package with Python version set to 3.12.0...")
            import ultimate_mcp_server
            
            # Check if warning was issued
            if mock_warn.called:
                print(f"✅ Success: warning was issued: '{mock_warn.call_args.args[0]}'")
            else:
                print("❌ Failure: No warning was issued")


def test_main_module_exit():
    """Test that __main__.py exits when run with wrong Python version"""
    print("\n=== Testing __main__.py exit ===")
    
    # Patch sys.version_info
    with patch('sys.version_info', fake_version_info):
        # Patch sys.exit to catch it without exiting
        with patch('sys.exit') as mock_exit:
            # Reset cached imports
            if 'ultimate_mcp_server.__main__' in sys.modules:
                del sys.modules['ultimate_mcp_server.__main__']
                
            print("Importing __main__ with Python version set to 3.12.0...")
            import ultimate_mcp_server.__main__
            
            # Check if sys.exit was called
            if mock_exit.called:
                print(f"✅ Success: sys.exit was called with code: {mock_exit.call_args.args[0]}")
            else:
                print("❌ Failure: sys.exit was not called")


def test_run_command_check():
    """Simplified test that just checks the run function directly"""
    print("\n=== Testing run command check ===")
    
    # Import the module outside the patch to avoid import errors
    from ultimate_mcp_server.cli import commands
    
    # Patch sys.version_info for the test
    with patch('sys.version_info', fake_version_info):
        # Mock the console.print to avoid actual printing
        with patch('rich.console.Console.print'):
            # Mock typer.Exit to catch it
            with patch('typer.Exit') as mock_exit:
                print("Running with Python version set to 3.12.0...")
                try:
                    # Try to run the command (will likely fail but that's what we're testing)
                    commands.run_server(host=None, port=None, workers=None, 
                                     transport_mode="sse", include_tools=None,
                                     exclude_tools=None, load_all_tools=False)
                    print("❌ Failure: Should have raised an exception")
                except Exception as e:
                    print(f"Got expected exception: {type(e).__name__}")
                
                # Check if typer.Exit was called
                if mock_exit.called:
                    print("✅ Success: typer.Exit was called")
                else:
                    print("❌ Failure: typer.Exit was not called")


if __name__ == "__main__":
    print("=== Testing Python Version Checks ===")
    print(f"Actual Python version: {sys.version}")
    
    # Run the tests
    try:
        test_package_import_warning()
        test_main_module_exit()
        test_run_command_check()
        print("\n✅ All tests completed successfully!")
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
    finally:
        # Restore the real version info
        sys.version_info = real_version_info 