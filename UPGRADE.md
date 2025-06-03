# 🔄 Ultimate MCP Server Upgrade Guide

> **🙏 Credit to [Dicklesworthstone](https://github.com/Dicklesworthstone)** for the original [Ultimate MCP Server](https://github.com/Dicklesworthstone/ultimate_mcp_server) that this Windows-optimized fork is based on.

This guide helps you upgrade existing installations of the Ultimate MCP Server to newer versions, including migrating from the original repository to this Windows-optimized fork.

## Quick Upgrade Commands

### If you're already using this Windows fork:

```bash
# Navigate to your existing installation
cd ultimate_mcp_server_windows

# Pull latest changes
git pull origin main

# Activate virtual environment
# Windows:
.venv\Scripts\activate.bat
# Linux/Mac:
source .venv/bin/activate

# Update dependencies
pip install -e . --upgrade

# Verify the upgrade
python -m ultimate_mcp_server.cli env --verbose
python -m ultimate_mcp_server.cli --version
```

### Using the activation scripts (recommended):

```bash
# Windows
activate_and_run.bat upgrade

# Linux/Mac
./activate_and_run.sh upgrade
```

## Migration Guide: From Original Repository to Windows Fork

If you're currently using Dicklesworthstone's original repository and want to migrate to this Windows-optimized fork:

### Step 1: Backup Your Configuration

```bash
# Navigate to your current installation
cd ultimate_mcp_server  # or wherever you have it installed

# Backup your configuration files
mkdir ../ultimate_mcp_backup
cp .env ../ultimate_mcp_backup/
cp -r storage/ ../ultimate_mcp_backup/ 2>/dev/null || true
cp -r logs/ ../ultimate_mcp_backup/ 2>/dev/null || true
cp -r examples/data/ ../ultimate_mcp_backup/ 2>/dev/null || true

# Note any custom configurations you've made
echo "Current server status before migration:" > ../ultimate_mcp_backup/pre_migration_notes.txt
python -m ultimate_mcp_server.cli tools >> ../ultimate_mcp_backup/pre_migration_notes.txt
python -m ultimate_mcp_server.cli providers >> ../ultimate_mcp_backup/pre_migration_notes.txt
```

### Step 2: Install the Windows Fork

```bash
# Navigate to parent directory
cd ..

# Clone the Windows-optimized fork
git clone https://github.com/android606/ultimate_mcp_server_windows.git
cd ultimate_mcp_server_windows

# Create new virtual environment
python -m venv .venv

# Activate and install
# Windows:
.venv\Scripts\activate.bat
# Linux/Mac:
source .venv/bin/activate

# Install the package
pip install -e .
```

### Step 3: Restore Your Configuration

```bash
# Copy back your configuration
cp ../ultimate_mcp_backup/.env .
cp -r ../ultimate_mcp_backup/storage/ . 2>/dev/null || true
cp -r ../ultimate_mcp_backup/logs/ . 2>/dev/null || true
cp -r ../ultimate_mcp_backup/examples/data/ ./examples/ 2>/dev/null || true

# Verify environment and configuration
python -m ultimate_mcp_server.cli env --verbose
```

### Step 4: Test the Migration

```bash
# Test basic functionality
python -m ultimate_mcp_server.cli env --check-only
python -m ultimate_mcp_server.cli tools
python -m ultimate_mcp_server.cli providers

# Test server startup
python -m ultimate_mcp_server.cli run --port 8014 --debug &
# Test connection (in another terminal)
curl http://localhost:8014/sse

# Stop test server
# Find and kill the process or use Ctrl+C
```

## Detailed Upgrade Procedures

### 1. Standard Upgrade (Same Repository)

For routine updates within the same repository:

```bash
# Navigate to installation directory
cd ultimate_mcp_server_windows

# Check current status before upgrade
echo "=== PRE-UPGRADE STATUS ==="
python -m ultimate_mcp_server.cli --version
python -m ultimate_mcp_server.cli env --check-only
git log --oneline -5

# Pull latest changes
git pull origin main

# Check if there are conflicts
git status

# Activate virtual environment
# Windows:
.venv\Scripts\activate.bat
# Linux/Mac:
source .venv/bin/activate

# Update dependencies (handles new requirements)
pip install -e . --upgrade

# Clear any cached bytecode
find . -name "*.pyc" -delete 2>/dev/null || true
find . -name "__pycache__" -type d -exec rm -rf {} + 2>/dev/null || true

# Verify upgrade
echo "=== POST-UPGRADE STATUS ==="
python -m ultimate_mcp_server.cli --version
python -m ultimate_mcp_server.cli env --verbose
```

### 2. Major Version Upgrades

For significant version changes that might have breaking changes:

```bash
# Create backup before major upgrade
mkdir backups/upgrade_$(date +%Y%m%d_%H%M%S)
cp .env backups/upgrade_$(date +%Y%m%d_%H%M%S)/
cp -r storage/ backups/upgrade_$(date +%Y%m%d_%H%M%S)/ 2>/dev/null || true

# Check for upgrade notes
git log --grep="BREAKING" --grep="breaking" --oneline
curl -s https://api.github.com/repos/android606/ultimate_mcp_server_windows/releases/latest | grep "body"

# Perform upgrade
git pull origin main

# Recreate virtual environment for major changes
deactivate 2>/dev/null || true
rm -rf .venv
python -m venv .venv

# Windows:
.venv\Scripts\activate.bat
# Linux/Mac:
source .venv/bin/activate

# Install fresh
pip install -e .

# Test compatibility
python -m ultimate_mcp_server.cli env --suggest
```

### 3. Dependency-Only Upgrades

To update just the Python dependencies without changing the codebase:

```bash
# Activate environment
# Windows:
.venv\Scripts\activate.bat
# Linux/Mac:
source .venv/bin/activate

# Update all dependencies to latest compatible versions
pip install --upgrade pip
pip install -e . --upgrade

# Or if using uv:
uv sync --upgrade

# Verify no compatibility issues
python -m ultimate_mcp_server.cli env --verbose
python -c "import ultimate_mcp_server; print('✅ Import test passed')"
```

## Handling Upgrade Issues

### Common Upgrade Problems

#### 1. Dependency Conflicts

```bash
# Check for conflicts
pip check

# View outdated packages
pip list --outdated

# Force reinstall if needed
pip install -e . --force-reinstall --no-deps
pip install -e .  # Install dependencies
```

#### 2. Python Version Incompatibility

```bash
# Check current Python version
python --version
python -m ultimate_mcp_server.cli env --verbose

# If you need to upgrade Python:
# 1. Install newer Python version from python.org
# 2. Create new virtual environment with new Python
python3.13 -m venv .venv_new  # Use newer Python
# Move to new environment and reinstall
```

#### 3. Configuration Changes

```bash
# Check if your .env file has all required new variables
python -m ultimate_mcp_server.cli env --suggest

# Compare with example configuration
diff .env .env.example || true

# Update your .env file with any new required variables
```

#### 4. Tool Compatibility Issues

```bash
# Check tool status after upgrade
python -m ultimate_mcp_server.cli run --load-all-tools --port 8015 &
# In another terminal:
python test_get_all_tools_status.py

# Check for any tool errors
python -m ultimate_mcp_server.cli tools | grep -i error || true
```

### Recovery Procedures

#### Rollback to Previous Version

```bash
# Check git log to find previous working version
git log --oneline -10

# Rollback to specific commit
git checkout [previous_commit_hash]

# Or rollback one commit
git reset --hard HEAD~1

# Reinstall the rolled-back version
pip install -e .

# Test the rollback
python -m ultimate_mcp_server.cli env --check-only
```

#### Clean Installation Recovery

```bash
# Nuclear option: completely fresh installation
cd ..
mv ultimate_mcp_server_windows ultimate_mcp_server_windows_backup

# Fresh clone
git clone https://github.com/android606/ultimate_mcp_server_windows.git
cd ultimate_mcp_server_windows

# Setup fresh environment
python -m venv .venv
# Windows:
.venv\Scripts\activate.bat
# Linux/Mac:
source .venv/bin/activate

# Install
pip install -e .

# Restore your config
cp ../ultimate_mcp_server_windows_backup/.env .
cp -r ../ultimate_mcp_server_windows_backup/storage/ . 2>/dev/null || true
```

## Environment Migration

### Virtual Environment Updates

If you need to recreate your virtual environment:

```bash
# Backup current package list
pip freeze > requirements_backup.txt

# Remove old environment
# Windows:
rmdir /s .venv
# Linux/Mac:
rm -rf .venv

# Create new environment
python -m venv .venv

# Activate new environment
# Windows:
.venv\Scripts\activate.bat
# Linux/Mac:
source .venv/bin/activate

# Install from scratch (recommended)
pip install -e .

# Or restore from backup (if needed)
# pip install -r requirements_backup.txt
```

### Configuration Migration

For changes to configuration format:

```bash
# Check if your configuration needs updates
python -m ultimate_mcp_server.cli env --suggest

# View configuration schema changes
git log --grep="config" --grep="environment" --oneline

# Update configuration format if needed
# (Follow any specific instructions in release notes)
```

## Testing Your Upgrade

### Comprehensive Upgrade Testing

```bash
# Environment validation
echo "=== ENVIRONMENT TEST ==="
python -m ultimate_mcp_server.cli env --strict

# Basic functionality
echo "=== BASIC FUNCTIONALITY TEST ==="
python -m ultimate_mcp_server.cli --version
python -m ultimate_mcp_server.cli tools
python -m ultimate_mcp_server.cli providers

# Server startup test
echo "=== SERVER STARTUP TEST ==="
python -m ultimate_mcp_server.cli run --port 8015 --debug &
SERVER_PID=$!
sleep 10

# Test server endpoint
curl -s http://localhost:8015/sse | head -5
kill $SERVER_PID 2>/dev/null || true

# Tool status test
echo "=== TOOL STATUS TEST ==="
python test_get_all_tools_status.py || true

# Provider connectivity test
echo "=== PROVIDER TEST ==="
python -m ultimate_mcp_server.cli test openai --prompt "test" || echo "OpenAI test failed - check API key"

echo "=== UPGRADE TESTING COMPLETE ==="
```

## Maintenance and Monitoring

### Regular Maintenance

Set up a routine for keeping your installation current:

```bash
# Weekly maintenance script
echo "#!/bin/bash" > weekly_maintenance.sh
echo "cd ultimate_mcp_server_windows" >> weekly_maintenance.sh
echo "git pull origin main" >> weekly_maintenance.sh
echo ".venv\Scripts\activate.bat || source .venv/bin/activate" >> weekly_maintenance.sh
echo "pip install -e . --upgrade" >> weekly_maintenance.sh
echo "python -m ultimate_mcp_server.cli env --check-only" >> weekly_maintenance.sh
chmod +x weekly_maintenance.sh
```

### Monitoring for Updates

```bash
# Check for new releases
curl -s https://api.github.com/repos/android606/ultimate_mcp_server_windows/releases/latest | jq -r '.tag_name'

# Set up a simple update checker
echo "python -c \"import subprocess; import json; r=subprocess.run(['curl', '-s', 'https://api.github.com/repos/android606/ultimate_mcp_server_windows/releases/latest'], capture_output=True, text=True); print('Latest release:', json.loads(r.stdout)['tag_name'])\"" > check_updates.py
```

## Getting Help with Upgrades

### If You Encounter Issues:

1. **Check the Environment**: `python -m ultimate_mcp_server.cli env --verbose --suggest`
2. **Review Release Notes**: Check GitHub releases for breaking changes
3. **Search Issues**: Look for similar problems in [GitHub Issues](https://github.com/android606/ultimate_mcp_server_windows/issues)
4. **Create Issue**: Report upgrade problems with detailed environment info
5. **Consult Original Project**: Check [Dicklesworthstone's repository](https://github.com/Dicklesworthstone/ultimate_mcp_server) for core documentation

### Useful Diagnostic Commands:

```bash
# Complete environment report
python -m ultimate_mcp_server.cli env --verbose > upgrade_diagnostic.txt
git log --oneline -10 >> upgrade_diagnostic.txt
pip list >> upgrade_diagnostic.txt
python --version >> upgrade_diagnostic.txt

# Include this file when reporting upgrade issues
```

---

> **Remember**: This Windows fork builds on the incredible foundation provided by [Dicklesworthstone](https://github.com/Dicklesworthstone). When reporting issues or contributing improvements, consider whether they should also be shared with the original project to benefit the entire MCP community! 