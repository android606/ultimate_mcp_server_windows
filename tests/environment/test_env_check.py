#!/usr/bin/env python
"""
Test script for environment verification in conftest.py
This script will try to import the conftest module directly,
which will trigger the environment checks.
"""
import sys
import os

# Add the tests directory to the Python path
tests_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, tests_dir)

print("Attempting to import conftest module...")
try:
    import conftest
    print("Successfully imported conftest module (this should not happen without proper environment)")
except SystemExit as e:
    print(f"SystemExit: {e}")
    print("Environment check worked as expected.")
except Exception as e:
    print(f"Unexpected error: {e}") 