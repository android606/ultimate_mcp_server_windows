# Check if running as administrator
if (-NOT ([Security.Principal.WindowsPrincipal] [Security.Principal.WindowsIdentity]::GetCurrent()).IsInRole([Security.Principal.WindowsBuiltInRole] "Administrator")) {
    Write-Warning "Please run this script as Administrator!"
    Read-Host "Press Enter to exit"
    exit 1
}

$TaskName = "Ultimate MCP Server"
$TaskPath = "\Ultimate MCP Server\"

# Remove existing task if it exists
if (Get-ScheduledTask -TaskName $TaskName -ErrorAction SilentlyContinue) {
    Write-Host "Removing existing task..."
    Unregister-ScheduledTask -TaskName $TaskName -Confirm:$false
}

# Create the task action
$Action = New-ScheduledTaskAction -Execute "C:\Users\android\ultimate_mcp_server\start_mcp_server.bat"

# Create the task trigger
$Trigger = New-ScheduledTaskTrigger -AtLogon

# Create the task principal
$Principal = New-ScheduledTaskPrincipal -UserId "android" -LogonType Interactive -RunLevel Highest

# Create the task settings
$Settings = New-ScheduledTaskSettingsSet -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries -StartWhenAvailable -ExecutionTimeLimit (New-TimeSpan -Hours 0) # No time limit

# Register the task
try {
    Register-ScheduledTask -TaskName $TaskName -Action $Action -Trigger $Trigger -Principal $Principal -Settings $Settings -Description "Starts the Ultimate MCP Server when logging in"
    Write-Host "Task created successfully!" -ForegroundColor Green
    Write-Host "The MCP server will start automatically when you log in."
} catch {
    Write-Host "Error creating task: $_" -ForegroundColor Red
    Read-Host "Press Enter to exit"
    exit 1
}

Read-Host "Press Enter to exit" 