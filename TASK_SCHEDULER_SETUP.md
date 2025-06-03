# Windows Task Scheduler Setup for Ultimate MCP Server

This guide explains how to set up the Ultimate MCP Server to run automatically using Windows Task Scheduler.

## Scripts Available

### 1. `activate_and_run_scheduled.ps1` (Recommended)
- **Purpose**: PowerShell Task Scheduler script with enhanced functionality
- **Features**:
  - Better error handling and structured logging
  - More robust virtual environment activation
  - Structured parameters support
  - Superior PowerShell integration with Task Scheduler

### 2. `activate_and_run_scheduled.bat` (Alternative)
- **Purpose**: Traditional batch script for compatibility
- **Features**: 
  - Simple batch file approach
  - Comprehensive logging with timestamps
  - Proper error handling and exit codes
  - Works on systems that restrict PowerShell

## Task Scheduler Setup Instructions

### Option 1: Using the PowerShell Script (Recommended)

1. **Set Execution Policy** (if needed)
   ```powershell
   Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
   ```

2. **Open Task Scheduler**
   - Press `Win + R`, type `taskschd.msc`, press Enter

3. **Create Basic Task**
   - Click "Create Basic Task..." in the right panel
   - Name: `Ultimate MCP Server`
   - Description: `Starts the Ultimate MCP Server automatically`

4. **Set Trigger**
   - Choose when you want it to run:
     - `When the computer starts` - for automatic startup
     - `Daily` - for scheduled runs
     - `When I log on` - for user session starts

5. **Set Action**
   - Action: `Start a program`
   - Program/script: `powershell.exe`
   - Add arguments: `-ExecutionPolicy Bypass -File "C:\full\path\to\your\ultimate_mcp_server\activate_and_run_scheduled.ps1"`
   - Start in: `C:\full\path\to\your\ultimate_mcp_server`

6. **Advanced Settings**
   - Right-click the created task → Properties
   - **General tab**:
     - ☑ Run whether user is logged on or not
     - ☑ Run with highest privileges
   - **Settings tab**:
     - ☑ Allow task to be run on demand
     - ☐ Stop the task if it runs longer than: (uncheck or set high value)
     - ☑ If the running task does not end when requested, force it to stop

### Option 2: Using the Batch Script (Compatibility Alternative)

1. **Follow steps 2-4 from Option 1**

2. **Set Action**
   - Action: `Start a program`
   - Program/script: `C:\full\path\to\your\ultimate_mcp_server\activate_and_run_scheduled.bat`
   - Start in: `C:\full\path\to\your\ultimate_mcp_server`

3. **Follow step 6 from Option 1**

## Command Line Usage

### PowerShell Script (Recommended)
```powershell
# Default (runs with --debug)
.\activate_and_run_scheduled.ps1

# Custom command and arguments
.\activate_and_run_scheduled.ps1 -Command "run" -Args @("--port", "8014")
.\activate_and_run_scheduled.ps1 -Command "env" -Args @("--verbose")
```

### Batch Script (Alternative)
```cmd
# Default (runs with --debug)
activate_and_run_scheduled.bat

# Custom command
activate_and_run_scheduled.bat run --port 8014
activate_and_run_scheduled.bat env --verbose
```

## Logging

Both scripts create comprehensive logs in the `logs` directory:

- **Timestamped logs**: `server_startup_YYYYMMDD_HHMMSS.log`
- **Latest log**: `latest.log` (always contains the most recent run)

### Log Content Includes:
- Script start/end times
- Virtual environment activation status
- Environment validation results
- Server startup command and arguments
- Exit codes and error messages
- Full execution trace

## Troubleshooting

### Common Issues

1. **"Virtual environment not found"**
   - Ensure the `.venv` directory exists in the project root
   - Run: `python -m venv .venv` and `pip install -e .`

2. **Environment validation failed**
   - Check the log for specific error messages
   - Run manually: `python -m ultimate_mcp_server.cli env --verbose --suggest`

3. **Task doesn't start**
   - Verify the "Start in" directory is correct
   - Check Task Scheduler's "Last Run Result" column
   - Ensure "Run with highest privileges" is checked

4. **PowerShell execution policy errors**
   - Run as administrator: `Set-ExecutionPolicy -ExecutionPolicy RemoteSigned`
   - Or use `-ExecutionPolicy Bypass` in the task arguments

### Checking Logs

```cmd
# View latest log
type logs\latest.log

# View all recent logs
dir logs\server_startup_*.log
```

### Manual Testing

Before setting up Task Scheduler, test the scripts manually:

```powershell
# Test PowerShell script (recommended)
.\activate_and_run_scheduled.ps1
```

```cmd
# Test batch script (alternative)
activate_and_run_scheduled.bat
```

## Environment Variables

The scripts will use environment variables from:
1. The virtual environment (`.venv`)
2. System environment variables
3. The `.env` file in the project directory

Ensure your `.env` file contains all necessary configuration:
```env
# API keys, database URLs, etc.
OPENAI_API_KEY=your_key_here
# ... other configuration
```

## Security Considerations

- **API Keys**: Ensure `.env` file permissions are secure
- **Run as Service**: Consider using Windows Services for production
- **Logging**: Be careful not to log sensitive information
- **Network Access**: Ensure firewall rules allow the server port

## Service Alternative

For production environments, consider creating a Windows Service instead:
```cmd
# Using NSSM (Non-Sucking Service Manager) with PowerShell script
nssm install "Ultimate MCP Server" powershell.exe
nssm set "Ultimate MCP Server" Arguments "-ExecutionPolicy Bypass -File \"C:\path\to\activate_and_run_scheduled.ps1\""

# Or with batch script
nssm install "Ultimate MCP Server" "C:\path\to\activate_and_run_scheduled.bat"
``` 