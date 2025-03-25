#!/usr/bin/env python
"""Script to fix environment variable loading for example scripts."""
import os
import sys
from pathlib import Path

# Base directory for the project
BASE_DIR = Path(__file__).parent

# Import decouple directly for a reliable .env reader
from decouple import config as decouple_config

# Get the absolute path to .env file
env_file = BASE_DIR / '.env'
print(f"Loading environment variables from: {env_file}")
print(f"Does .env exist? {env_file.exists()}")

# Apply a monkey patch to the providers
def patch_providers():
    """Monkey patch the provider initialization to directly set API keys."""
    from llm_gateway.core.providers import base
    
    # Store the original provider initialization method
    original_initialize = base.Provider.initialize
    
    async def patched_initialize(self):
        """Patched initialize method that ensures API keys are set."""
        # Make sure API keys are set before initialization
        if self.provider_name == "openai":
            api_key = decouple_config("OPENAI_API_KEY", default=None)
            if api_key:
                os.environ["OPENAI_API_KEY"] = api_key
                print(f"✅ Set OPENAI_API_KEY from .env")
                
        elif self.provider_name == "anthropic":
            api_key = decouple_config("ANTHROPIC_API_KEY", default=None)
            if api_key:
                os.environ["ANTHROPIC_API_KEY"] = api_key
                print(f"✅ Set ANTHROPIC_API_KEY from .env")
                
        elif self.provider_name == "gemini":
            api_key = decouple_config("GEMINI_API_KEY", default=None)
            if api_key:
                os.environ["GEMINI_API_KEY"] = api_key
                print(f"✅ Set GEMINI_API_KEY from .env")
                
        elif self.provider_name == "deepseek":
            api_key = decouple_config("DEEPSEEK_API_KEY", default=None)
            if api_key:
                os.environ["DEEPSEEK_API_KEY"] = api_key
                print(f"✅ Set DEEPSEEK_API_KEY from .env")
        
        # Call the original initialization method
        return await original_initialize(self)
    
    # Apply the patch
    base.Provider.initialize = patched_initialize
    print("✅ Patched provider initialization to use .env directly")

# Add a function to run an example script with patched environment
def run_example(example_script):
    """Run an example script with patched environment.
    
    Args:
        example_script: Path to example script
    """
    # Make sure the script exists
    script_path = Path(example_script)
    if not script_path.exists():
        print(f"❌ Script not found: {script_path}")
        return 1
    
    print(f"🚀 Running: {script_path}")
    
    # Apply the provider patch
    patch_providers()
    
    # Execute the script
    with open(script_path) as f:
        script_code = f.read()
    
    # Run the script in the current process
    sys.path.insert(0, str(BASE_DIR))
    exec(script_code)
    
    return 0

# Main function
if __name__ == "__main__":
    # Check if an example script was provided
    if len(sys.argv) < 2:
        print("Usage: python fix_env_problems.py <example_script>")
        print("Example: python fix_env_problems.py examples/claude_integration.py")
        sys.exit(1)
    
    # Run the example
    example_script = sys.argv[1]
    exit_code = run_example(example_script)
    sys.exit(exit_code) 