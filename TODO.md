# 📋 Windows Fork TODO

## High Priority (Windows-Specific)
* Add automated tests for Windows environment validation
* Improve PowerShell integration and support  
* Add Windows Service installation option
* Test and fix any remaining Windows path handling issues

## Medium Priority
* Add pre-commit hooks configuration for development workflow
* Enhance activation scripts with better error recovery
* Add development dependency groups to pyproject.toml (dev, test)
* Create automated release workflow for Windows fork

## Low Priority  
* Improve CLI help text and documentation
* Add Windows-specific performance optimizations
* Consider Windows installer (.msi) for easier distribution

## Completed ✅
* ~~Make CLI options to enable/disable tools from loading~~ (Available via --include-tools, --exclude-tools, --load-all-tools)
* ~~Debug SQL and Playwright tools~~ (Working with proper environment setup)
* ~~Improve docstrings for better tool usage by claude~~ (Enhanced via environment validation system)
* ~~Create comprehensive Windows installation guide~~ (INSTALLATION.md)
* ~~Add environment validation system~~ (ultimate_mcp_server/utils/environment.py)
* ~~Create activation scripts for streamlined setup~~ (activate_and_run.bat/sh)