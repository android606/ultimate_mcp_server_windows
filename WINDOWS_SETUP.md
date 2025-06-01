# Windows Setup Guide for Ultimate MCP Server

This guide provides Windows-specific setup instructions for the Ultimate MCP Server, including compatibility fixes and integration with Cursor IDE.

## Prerequisites

### Python 3.13+
- Download and install Python 3.13+ from [python.org](https://www.python.org/downloads/)
- **Important**: During installation, check "Add Python to PATH"
- Verify installation: `python --version`

### Git for Windows
- Download from [git-scm.com](https://git-scm.com/download/win)
- Use Git Bash for command line operations (recommended)

### Optional: Windows Subsystem for Linux (WSL)
- For better Unix tool compatibility, consider installing WSL2
- Some text processing tools work better in WSL environment

## Installation

### 1. Clone the Repository
```bash
git clone https://github.com/Dicklesworthstone/ultimate_mcp_server.git
cd ultimate_mcp_server
```

### 2. Create Virtual Environment
```bash
python -m venv venv
# Activate virtual environment
venv\Scripts\activate  # Windows Command Prompt
# OR
source venv/Scripts/activate  # Git Bash
```

### 3. Install Dependencies
```bash
pip install -e .
```

### 4. Install Additional Windows Dependencies
Some tools may require additional Windows-specific packages:
```bash
# For document processing
pip install python-docx
# For web automation (requires additional setup)
pip install playwright
playwright install chromium
```

## Windows Compatibility Fixes

This branch includes several Windows compatibility fixes:

### 1. Filesystem Module Fixes
- Fixed `os.path.stat.S_ISLNK()` calls that don't work on Windows
- Replaced with proper `stat.S_ISLNK()` calls from the stat module
- Added missing `import stat` statement

### 2. Process Management
- Proper handling of `preexec_fn` parameter (Unix-only)
- Windows-compatible subprocess creation
- Environment variable handling for both Windows and Unix

### 3. Path Handling
- Uses `os.path.join()` for cross-platform path construction
- Handles both forward and backslash path separators
- Proper Windows drive letter support

## Configuration

### 1. Environment Variables
Create a `.env` file in the project root:
```env
# Server Configuration
SERVER_HOST=127.0.0.1
SERVER_PORT=8080

# AI Provider API Keys (optional)
OPENAI_API_KEY=your_openai_key_here
ANTHROPIC_API_KEY=your_anthropic_key_here
GOOGLE_API_KEY=your_google_key_here

# Gitea Configuration (optional)
GITEA_BASE_URL=https://your-gitea-server.com
GITEA_TOKEN=your_gitea_token_here

# Storage Configuration
STORAGE_DIR=C:\Users\%USERNAME%\ultimate_mcp_server\storage
```

### 2. Cursor IDE Integration
The repository includes pre-configured Cursor IDE settings:

#### Local Settings (`.cursor/settings.json`)
```json
{
  "mcp.servers": {
    "ultimate_mcp_server": {
      "command": "python",
      "args": ["-m", "ultimate_mcp_server", "run", "--transport-mode", "sse", "--host", "127.0.0.1", "--port", "8080"],
      "cwd": "C:\\Users\\android\\ultimate_mcp_server"
    }
  }
}
```

#### Global MCP Configuration (`mcp_config_for_cursor.json`)
Copy this to your Cursor MCP configuration directory or use as reference.

## Running the Server

### 1. Manual Start
```bash
# Activate virtual environment first
source venv/Scripts/activate  # Git Bash
# OR
venv\Scripts\activate  # Command Prompt

# Start server
python -m ultimate_mcp_server run --transport-mode sse --host 127.0.0.1 --port 8080
```

### 2. Automated Startup (Windows)
The repository includes Windows startup automation:

#### Option A: Task Scheduler (Recommended)
1. Run PowerShell as Administrator
2. Execute: `.\setup_mcp_task.ps1`
3. Or import `ultimate_mcp_server_task.xml` directly into Task Scheduler

#### Option B: Batch File
Double-click `start_mcp_server.bat` to start the server

#### Option C: Shell Script (Git Bash)
```bash
./start_mcp_server.sh
```

## Testing the Installation

### 1. Check Server Status
```bash
# Test server health endpoint
curl http://127.0.0.1:8080/health

# Or use the provided script
./check_mcp_server.sh
```

### 2. Test MCP Tools
Open Cursor IDE and try these commands:
- Generate text with AI models
- Search the web
- Process files in the storage directory
- Execute Python code in sandbox

## Troubleshooting

### Common Windows Issues

#### 1. Python Not Found
- Ensure Python is in your PATH
- Restart command prompt/Git Bash after Python installation
- Use full path: `C:\Python313\python.exe`

#### 2. Permission Errors
- Run command prompt as Administrator
- Check antivirus software blocking Python execution
- Ensure user has write permissions to project directory

#### 3. Git Configuration Issues
If you encounter git config errors:
```bash
export HOME=/c/Users/yourusername
git config user.name "Your Name"
git config user.email "your.email@example.com"
```

#### 4. Module Import Errors
- Ensure virtual environment is activated
- Reinstall dependencies: `pip install -e . --force-reinstall`
- Check for conflicting Python installations

#### 5. Port Already in Use
- Change port in `.env` file: `SERVER_PORT=8081`
- Kill existing processes: `taskkill /f /im python.exe`
- Use different port in startup commands

### Text Processing Tools
Some Unix text tools may not be available on Windows:
- Install Git for Windows (includes basic Unix tools)
- Use Windows Subsystem for Linux (WSL)
- Install individual tools via package managers like Chocolatey

## Development

### Creating Windows-Compatible Code
When contributing to this project:

1. **Use `os.path.join()` for paths**
2. **Check `sys.platform` for OS-specific code**
3. **Use `pathlib.Path` for modern path handling**
4. **Test subprocess calls on Windows**
5. **Handle both `/` and `\` path separators**

### Testing on Windows
```bash
# Run tests
python -m pytest tests/

# Test specific Windows functionality
python -m pytest tests/test_windows_compatibility.py
```

## Support

For Windows-specific issues:
1. Check this guide first
2. Search existing GitHub issues
3. Create new issue with "Windows" label
4. Include system information:
   - Windows version
   - Python version
   - Error messages
   - Steps to reproduce

## Contributing

When submitting Windows-related fixes:
1. Create feature branch from `windows-compatibility`
2. Test on multiple Windows versions if possible
3. Update this guide if needed
4. Submit pull request with detailed description

## License

Same as main project - see LICENSE file. 