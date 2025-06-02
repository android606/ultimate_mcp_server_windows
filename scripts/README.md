# 🔧 Utility Scripts

This directory contains utility and maintenance scripts for the Ultimate MCP Server (Windows-optimized fork).

## Setup and Configuration Scripts

### **Environment and API Setup**
- **`check_api_keys.py`** - Validates API key configuration for all supported LLM providers (OpenAI, Anthropic, Google, etc.)
- **`list_models.py`** - Lists available models from all configured providers with cost and capability information

### **Development Tools**
- **`model_preferences.py`** - Configures default model preferences and provider priorities for cost optimization
- **`setup_release.py`** - Automates release preparation, version bumping, and changelog generation

### **Testing and Quality Assurance**
- **`run_all_demo_scripts_and_check_for_errors.py`** - Comprehensive test runner that executes all demo scripts and validates outputs

## Usage Examples

### API Key Validation
```bash
# Check all configured API keys
python scripts/check_api_keys.py

# Check specific provider
python scripts/check_api_keys.py --provider openai
```

### Model Information
```bash
# List all available models
python scripts/list_models.py

# List models for specific provider
python scripts/list_models.py --provider anthropic

# Show cost information
python scripts/list_models.py --show-costs
```

### Development Workflow
```bash
# Configure model preferences for development
python scripts/model_preferences.py --set-default-provider openai

# Run all demos to check for regressions  
python scripts/run_all_demo_scripts_and_check_for_errors.py

# Prepare release
python scripts/setup_release.py --version 1.2.3
```

## Script Categories

### 🔑 **API Management**
Scripts for managing API keys, provider configuration, and authentication

### 📋 **Model Management**  
Tools for discovering, configuring, and optimizing model usage

### 🏗️ **Development Tools**
Utilities for development workflow, testing, and release management

### 📊 **Quality Assurance**
Scripts for comprehensive testing and validation

## Windows-Specific Considerations

### Path Handling
All scripts properly handle Windows path separators and long paths.

### Environment Variables
Scripts read from both `.env` files and Windows environment variables.

### Process Management
Scripts include proper cleanup for Windows process termination.

## Integration with Main Project

### CLI Integration
Many scripts can be called via the main CLI:
```bash
# Instead of: python scripts/check_api_keys.py
python -m ultimate_mcp_server.cli env --check-api-keys

# Instead of: python scripts/list_models.py  
python -m ultimate_mcp_server.cli models --list
```

### Configuration Files
Scripts respect configuration in:
- `.env` files
- `mcp_config_for_cursor.json`
- Environment variables

## Contributing New Scripts

When adding utility scripts:

1. **Follow naming convention**: Use descriptive names with underscores
2. **Add error handling**: Include proper exception handling and cleanup
3. **Support Windows**: Test on Windows environments specifically
4. **Add documentation**: Update this README with script description
5. **CLI integration**: Consider adding to main CLI interface

### Script Template
```python
#!/usr/bin/env python3
\"\"\"
Script Description: Brief description of what this script does

Usage:
    python scripts/script_name.py [options]
\"\"\"

import argparse
import sys
from pathlib import Path

def main():
    parser = argparse.ArgumentParser(description="Script description")
    # Add arguments...
    
    args = parser.parse_args()
    
    try:
        # Script logic...
        pass
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()
```

## Maintenance

### Regular Tasks
- **Weekly**: Run `run_all_demo_scripts_and_check_for_errors.py`
- **Before releases**: Execute `setup_release.py`
- **After config changes**: Validate with `check_api_keys.py`

### Monitoring
Scripts generate logs in the `logs/` directory for troubleshooting and monitoring.

---

> **💡 Tip**: Use `python scripts/script_name.py --help` for detailed usage information on any script. 