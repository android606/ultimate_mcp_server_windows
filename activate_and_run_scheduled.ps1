# Ultimate MCP Server - Task Scheduler Startup Script (PowerShell)
# This script is designed to run from Windows Task Scheduler without user interaction

param(
    [string[]]$Args = @("run", "--load-all-tools", "--host", "0.0.0.0", "--port", "8013")
)

# Set up logging
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$LogDir = Join-Path $ScriptDir "logs"
$Timestamp = Get-Date -Format "yyyyMMdd_HHmmss"
$LogFile = Join-Path $LogDir "server_startup_$Timestamp.log"
$LatestLogFile = Join-Path $LogDir "latest.log"

# Create logs directory if it doesn't exist
if (-not (Test-Path $LogDir)) {
    New-Item -ItemType Directory -Path $LogDir -Force | Out-Null
}

# Function to write to both console and log
function Write-Log {
    param([string]$Message)
    $LogMessage = "$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss'): $Message"
    Write-Host $LogMessage
    Add-Content -Path $LogFile -Value $LogMessage
}

try {
    Write-Log "====================================================================="
    Write-Log " Ultimate MCP Server - Task Scheduler Startup Script (PowerShell)"
    Write-Log "====================================================================="
    Write-Log ""

    # Change to script directory
    Set-Location $ScriptDir
    Write-Log "Current directory: $(Get-Location)"

    # Check if virtual environment exists
    $VenvActivate = Join-Path $ScriptDir ".venv\Scripts\Activate.ps1"
    if (Test-Path $VenvActivate) {
        Write-Log "[OK] Found virtual environment at .venv"
        
        # Activate the virtual environment
        Write-Log ""
        Write-Log "Activating virtual environment..."
        
        # Set execution policy for this session to allow the activation script
        Set-ExecutionPolicy -ExecutionPolicy Bypass -Scope Process -Force
        
        # Activate the virtual environment
        & $VenvActivate
        
        # Verify activation worked
        $PythonPath = Get-Command python -ErrorAction SilentlyContinue
        if (-not $PythonPath) {
            Write-Log "[ERROR] Failed to activate virtual environment"
            Write-Log "Virtual environment activation failed - exiting"
            exit 1
        }
        
        Write-Log "[OK] Virtual environment activated"
        Write-Log "Python path: $($PythonPath.Source)"
        
    } else {
        Write-Log "[ERROR] Virtual environment not found at .venv"
        Write-Log "Please create a virtual environment first:"
        Write-Log "  python -m venv .venv"
        Write-Log "  .venv\Scripts\Activate.ps1"
        Write-Log "  pip install -e ."
        Write-Log ""
        exit 1
    }

    # Check environment using our validation tool
    Write-Log ""
    Write-Log "Checking environment status..."
    
    $EnvCheckResult = & python -m ultimate_mcp_server env --check-only 2>&1
    if ($LASTEXITCODE -ne 0) {
        Write-Log ""
        Write-Log "[ERROR] Environment validation failed!"
        Write-Log "Output: $EnvCheckResult"
        Write-Log "Run this command for detailed diagnostics:"
        Write-Log "  python -m ultimate_mcp_server env --verbose --suggest"
        Write-Log ""
        exit 1
    }

    Write-Log "[OK] Environment validation passed"

    # Prepare command arguments properly
    $ArgumentList = @("-m", "ultimate_mcp_server")
    if ($Args -and $Args.Count -gt 0) {
        $ArgumentList += $Args
    }
    
    $CmdString = "python " + ($ArgumentList -join " ")

    # Show what we're about to run
    Write-Log ""
    Write-Log "Starting Ultimate MCP Server with command: $CmdString"
    Write-Log "Argument list: $($ArgumentList -join ', ')"
    Write-Log ""
    Write-Log "====================================================================="

    # Run the Ultimate MCP Server directly (not through CLI)
    Write-Log "Server starting at: $(Get-Date)"
    
    $ServerProcess = Start-Process -FilePath "python" -ArgumentList $ArgumentList -NoNewWindow -PassThru -Wait
    $ExitCode = $ServerProcess.ExitCode

    # Check exit code
    if ($ExitCode -ne 0) {
        Write-Log ""
        Write-Log "[ERROR] Server exited with error code $ExitCode at $(Get-Date)"
    } else {
        Write-Log ""
        Write-Log "[OK] Server exited normally at $(Get-Date)"
    }

    Write-Log ""
    Write-Log "====================================================================="
    Write-Log "Script completed at: $(Get-Date)"
    Write-Log "Exit code: $ExitCode"
    Write-Log "====================================================================="

    # Copy to latest.log for easy access
    if (Test-Path $LogFile) {
        Copy-Item $LogFile $LatestLogFile -Force
    }

    exit $ExitCode

} catch {
    $ErrorMessage = $_.Exception.Message
    Write-Log "[FATAL ERROR] $ErrorMessage"
    Write-Log "Stack trace: $($_.Exception.StackTrace)"
    
    # Copy to latest.log for easy access
    if (Test-Path $LogFile) {
        Copy-Item $LogFile $LatestLogFile -Force
    }
    
    exit 1
} 