# Setup Virtual Environment for Ultimate MCP Server using PowerShell
Write-Host "Setting up Virtual Environment for Ultimate MCP Server (PowerShell version)..." -ForegroundColor Cyan

# Check if running as administrator
$isAdmin = ([Security.Principal.WindowsPrincipal] [Security.Principal.WindowsIdentity]::GetCurrent()).IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)
if (-not $isAdmin) {
    Write-Host "This script is not running as administrator." -ForegroundColor Yellow
    Write-Host "Some operations might fail due to permission issues." -ForegroundColor Yellow
    Write-Host "Consider restarting this script as administrator if you encounter problems." -ForegroundColor Yellow
    Write-Host ""
}

# Check if Python is available
try {
    $pythonVersion = python --version
    Write-Host "Found Python: $pythonVersion" -ForegroundColor Green
} catch {
    Write-Host "Python not found in PATH. Please install Python 3.13+ and try again." -ForegroundColor Red
    exit 1
}

# Check for existing .venv directory
if (Test-Path .venv) {
    Write-Host "Found existing .venv directory." -ForegroundColor Yellow
    $remove = Read-Host "Do you want to remove it and create a fresh environment? (y/N)"
    if ($remove -eq "y" -or $remove -eq "Y") {
        Write-Host "Removing existing virtual environment..." -ForegroundColor Yellow
        try {
            Remove-Item -Recurse -Force .venv
        } catch {
            Write-Host "Failed to remove .venv directory. Trying alternative method..." -ForegroundColor Red
            cmd /c "rmdir /S /Q .venv"
        }
    }
}

# Create venv
Write-Host "Creating virtual environment..." -ForegroundColor Cyan
try {
    python -m venv .venv --clear
} catch {
    Write-Host "Failed to create virtual environment. Trying alternative approach..." -ForegroundColor Yellow
    cmd /c "python -m venv .venv --clear"
    
    if (-not (Test-Path .venv\Scripts\python.exe)) {
        Write-Host "Virtual environment setup failed." -ForegroundColor Red
        Write-Host "Please try running this script as administrator." -ForegroundColor Red
        exit 1
    }
}

# Activate the environment
Write-Host "Activating environment..." -ForegroundColor Cyan
try {
    & .venv\Scripts\Activate.ps1
} catch {
    Write-Host "Failed to activate environment directly." -ForegroundColor Yellow
    Write-Host "Will use full paths instead." -ForegroundColor Yellow
}

# Install the package in development mode
Write-Host "Installing Ultimate MCP Server in development mode..." -ForegroundColor Cyan
try {
    & .venv\Scripts\python.exe -m pip install -U pip
    & .venv\Scripts\python.exe -m pip install -e ".[test]"
} catch {
    Write-Host "Package installation failed: $_" -ForegroundColor Red
    exit 1
}

# Customize the activation scripts
Write-Host "Customizing activation scripts for better Windows compatibility..." -ForegroundColor Cyan
# Add our custom PATH handling to the PowerShell activation script
$activatePs1 = ".venv\Scripts\Activate.ps1"
if (Test-Path $activatePs1) {
    $content = Get-Content $activatePs1 -Raw
    if (-not $content.Contains("# Ultimate MCP Server customization")) {
        $customContent = @"

# Ultimate MCP Server customization
`$env:PATH = "`$VirtualEnv\Scripts;" + `$env:PATH

# Ensure HOME is set for Git
if (-not (Test-Path env:HOME)) {
    if (Test-Path env:USERPROFILE) {
        `$env:HOME = `$env:USERPROFILE
    } elseif ((Test-Path env:HOMEDRIVE) -and (Test-Path env:HOMEPATH)) {
        `$env:HOME = `$env:HOMEDRIVE + `$env:HOMEPATH
    }
}

# Set environment variable to skip environment checks
`$env:UMCP_SKIP_ENV_CHECK = 1
"@
        # Find insertion point
        $lines = $content -split "`n"
        $updated = $false
        for ($i = 0; $i -lt $lines.Length; $i++) {
            if ($lines[$i].Trim() -eq '$VirtualEnv = $PSScriptRoot | Split-Path') {
                # Find end of initialization block
                for ($j = $i; $j -lt $lines.Length; $j++) {
                    if ($lines[$j].Trim() -eq '') {
                        $lines[$j] = $customContent + "`n"
                        $updated = $true
                        break
                    }
                }
                if ($updated) { break }
            }
        }
        
        if ($updated) {
            $lines -join "`n" | Set-Content $activatePs1 -Force
            Write-Host "Successfully customized PowerShell activation script." -ForegroundColor Green
        } else {
            Add-Content $activatePs1 $customContent
            Write-Host "Added customization at the end of PowerShell activation script." -ForegroundColor Yellow
        }
    } else {
        Write-Host "PowerShell activation script already customized." -ForegroundColor Green
    }
}

# Done
Write-Host "`n✨ Setup complete! ✨" -ForegroundColor Green
Write-Host "`nTo activate the environment in PowerShell:"
Write-Host "    .venv\Scripts\Activate.ps1" -ForegroundColor Cyan

Write-Host "`nThen run the server with:"
Write-Host "    python -m ultimate_mcp_server run --port 8014 --host 127.0.0.1 --debug" -ForegroundColor Cyan

Write-Host "`nTo run tests:"
Write-Host "    python -m pytest tests/test_server_startup.py -v" -ForegroundColor Cyan 
Write-Host "    python -m pytest tests/environment/ -v" -ForegroundColor Cyan 