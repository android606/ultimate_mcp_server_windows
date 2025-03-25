#!/usr/bin/env python
"""Debug script to test .env loading."""
import os
import sys
from pathlib import Path

# Base directory
BASE_DIR = Path(__file__).parent

print(f"Current working directory: {os.getcwd()}")
print(f"Script location: {BASE_DIR}")
print(f".env path: {BASE_DIR / '.env'}")
print(f"Does .env exist? {(BASE_DIR / '.env').exists()}")

# Import directly from decouple
try:
    from decouple import Config, RepositoryEnv
    
    print("\nTesting python-decouple direct loading:")
    config_source = RepositoryEnv(BASE_DIR / '.env')
    env_config = Config(config_source)
    
    # Try to read some values
    anthropic_key = env_config('ANTHROPIC_API_KEY', default="NOT_FOUND")
    openai_key = env_config('OPENAI_API_KEY', default="NOT_FOUND")
    
    print(f"ANTHROPIC_API_KEY from direct decouple: {'Found' if anthropic_key != 'NOT_FOUND' else 'Not found'}")
    if anthropic_key != "NOT_FOUND":
        print(f"  Key starts with: {anthropic_key[:10]}...")
        
    print(f"OPENAI_API_KEY from direct decouple: {'Found' if openai_key != 'NOT_FOUND' else 'Not found'}")
    if openai_key != "NOT_FOUND":
        print(f"  Key starts with: {openai_key[:10]}...")
except Exception as e:
    print(f"Error in direct decouple loading: {e}")

# Add the project root to path for imports
sys.path.insert(0, str(BASE_DIR))

# Import our own get_env with better error handling
try:
    from llm_gateway.config import BASE_DIR as CONFIG_BASE_DIR
    print(f"\nllm_gateway.config.BASE_DIR: {CONFIG_BASE_DIR}")
    print(f"Does config see .env? {(CONFIG_BASE_DIR / '.env').exists()}")
    
    from llm_gateway.config import get_env
    print("\nTesting get_env function:")
    
    anthropic_key = get_env('ANTHROPIC_API_KEY', default="NOT_FOUND")
    openai_key = get_env('OPENAI_API_KEY', default="NOT_FOUND")
    
    print(f"ANTHROPIC_API_KEY from get_env: {'Found' if anthropic_key != 'NOT_FOUND' else 'Not found'}")
    if anthropic_key != "NOT_FOUND":
        print(f"  Key starts with: {anthropic_key[:10]}...")
    
    print(f"OPENAI_API_KEY from get_env: {'Found' if openai_key != 'NOT_FOUND' else 'Not found'}")
    if openai_key != "NOT_FOUND":
        print(f"  Key starts with: {openai_key[:10]}...")
except Exception as e:
    print(f"Error in get_env: {e}")

# Check direct environment variables
print("\nChecking OS environment variables:")
anthropic_env = os.environ.get('ANTHROPIC_API_KEY')
openai_env = os.environ.get('OPENAI_API_KEY')

print(f"ANTHROPIC_API_KEY in os.environ: {'Found' if anthropic_env else 'Not found'}")
if anthropic_env:
    print(f"  Key starts with: {anthropic_env[:10]}...")
    
print(f"OPENAI_API_KEY in os.environ: {'Found' if openai_env else 'Not found'}")
if openai_env:
    print(f"  Key starts with: {openai_env[:10]}...") 