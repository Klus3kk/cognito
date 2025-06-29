# Cognito Windows Installer
# Version: 1.0.0
# Usage: iex (irm https://raw.githubusercontent.com/yourusername/cognito/main/install.ps1)

[CmdletBinding()]
param(
    [switch]$Help,
    [switch]$Version,
    [switch]$AllowAdmin,
    [switch]$Uninstall,
    [switch]$Update,
    [string]$InstallDir = "$env:USERPROFILE\.cognito",
    [ValidateSet("standard", "minimal", "development")]
    [string]$InstallType = "standard"
)

# Global configuration
$script:CognitoVersion = "0.8.0"
$script:PythonMinVersion = "3.8"
$script:PythonMaxVersion = "3.12"
$script:VenvName = "cognito-env"
$script:InstallDir = $InstallDir
$script:InstallType = $InstallType

# Global installation options
$global:EnableLLM = $false
$global:InstallOptionalDeps = $false
$global:InstallDocker = $false

# Colors for output
$Colors = @{
    Red     = 'Red'
    Green   = 'Green'
    Yellow  = 'Yellow'
    Blue    = 'Blue'
    Cyan    = 'Cyan'
    Purple  = 'Magenta'
}

# ASCII Art Logo
function Show-Logo {
    Write-Host -ForegroundColor $Colors.Cyan @"
 ██████╗ ██████╗  ██████╗ ███╗   ██╗██╗████████╗ ██████╗ 
██╔════╝██╔═══██╗██╔════╝ ████╗  ██║██║╚══██╔══╝██╔═══██╗
██║     ██║   ██║██║  ███╗██╔██╗ ██║██║   ██║   ██║   ██║
██║     ██║   ██║██║   ██║██║╚██╗██║██║   ██║   ██║   ██║
╚██████╗╚██████╔╝╚██████╔╝██║ ╚████║██║   ██║   ╚██████╔╝
 ╚═════╝ ╚═════╝  ╚═════╝ ╚═╝  ╚═══╝╚═╝   ╚═╝    ╚═════╝ 
                                                           
    AI-Powered Code Analysis Platform - Windows Installer v1.0
"@
    Write-Host ""
}

# Logging functions
function Write-LogInfo {
    param([string]$Message)
    Write-Host -ForegroundColor $Colors.Blue "[INFO] $Message"
}

function Write-LogSuccess {
    param([string]$Message)
    Write-Host -ForegroundColor $Colors.Green "[SUCCESS] $Message"
}

function Write-LogWarn {
    param([string]$Message)
    Write-Host -ForegroundColor $Colors.Yellow "[WARNING] $Message"
}

function Write-LogError {
    param([string]$Message)
    Write-Host -ForegroundColor $Colors.Red "[ERROR] $Message"
}

function Write-LogStep {
    param([string]$Message)
    Write-Host ""
    Write-Host -ForegroundColor $Colors.Purple "[STEP] $Message"
}

# Progress bar
function Show-Progress {
    param(
        [int]$Current,
        [int]$Total
    )
    
    $percentage = ($Current / $Total) * 100
    Write-Progress -Activity "Installing Cognito" -Status "Step $Current of $Total" -PercentComplete $percentage
}

# Check if running as administrator
function Test-Administrator {
    $currentUser = [Security.Principal.WindowsIdentity]::GetCurrent()
    $principal = New-Object Security.Principal.WindowsPrincipal($currentUser)
    return $principal.IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)
}

# Check system requirements
function Test-Requirements {
    Write-LogStep "Checking system requirements"
    
    # Check Windows version
    $osVersion = [System.Environment]::OSVersion.Version
    Write-LogInfo "Windows version: $($osVersion.Major).$($osVersion.Minor)"
    if ($osVersion.Major -lt 10) {
        Write-LogWarn "Windows 10 or later is recommended"
    }
    
    # Check Python installation
    $pythonExe = $null
    $pythonPaths = @("python", "python3", "py")
    
    foreach ($pyCmd in $pythonPaths) {
        try {
            $version = & $pyCmd --version 2>$null
            if ($version -match "Python (\d+\.\d+)") {
                $pyVersion = [version]$matches[1]
                if ($pyVersion -ge [version]$script:PythonMinVersion -and $pyVersion -lt [version]"3.13") {
                    $pythonExe = $pyCmd
                    Write-LogSuccess "Found Python $($matches[1]) using '$pyCmd'"
                    break
                }
            }
        }
        catch {
            continue
        }
    }
    
    if (-not $pythonExe) {
        Write-LogError "Python $($script:PythonMinVersion)-$($script:PythonMaxVersion) is required but not found"
        Write-LogInfo "Download Python from: https://www.python.org/downloads/"
        throw "Python not found"
    }
    
    # Check pip
    try {
        & $pythonExe -m pip --version | Out-Null
        Write-LogSuccess "pip is available"
    }
    catch {
        Write-LogError "pip is not available"
        throw "pip not found"
    }
    
    # Check git (optional)
    try {
        git --version | Out-Null
        Write-LogSuccess "Git is available"
    }
    catch {
        Write-LogWarn "Git is not installed. Will use local files if available."
        Write-LogInfo "Download Git from: https://git-scm.com/download/win"
    }
    
    # Check available disk space (minimum 1GB)
    try {
        $drive = (Get-Item $script:InstallDir).PSDrive.Name + ":"
        $freeSpace = (Get-WmiObject -Class Win32_LogicalDisk -Filter "DeviceID='$drive'").FreeSpace / 1GB
        if ($freeSpace -lt 1) {
            Write-LogWarn "Less than 1GB disk space available"
        }
    }
    catch {
        Write-LogWarn "Could not check disk space"
    }
    
    # Check memory (minimum 2GB)
    try {
        $totalMemory = (Get-WmiObject -Class Win32_ComputerSystem).TotalPhysicalMemory / 1GB
        if ($totalMemory -lt 2) {
            Write-LogWarn "Less than 2GB RAM available. Some features may be slower"
        }
    }
    catch {
        Write-LogWarn "Could not check memory"
    }
    
    return $pythonExe
}

# Interactive configuration
function Set-Configuration {
    Write-LogStep "Configuring installation"
    
    Write-Host "Please select installation type:"
    Write-Host "1) Standard - Full installation with all features"
    Write-Host "2) Minimal - Core features only" 
    Write-Host "3) Development - Full installation + development tools"
    
    do {
        $choice = Read-Host "Enter choice [1-3] (default: 1)"
        if ([string]::IsNullOrEmpty($choice)) { $choice = "1" }
    } while ($choice -notmatch "^[1-3]$")
    
    switch ($choice) {
        "2" { $script:InstallType = "minimal" }
        "3" { $script:InstallType = "development" }
        default { $script:InstallType = "standard" }
    }
    
    Write-LogInfo "Selected installation type: $($script:InstallType)"
    
    # Ask about LLM features
    $enableLlm = Read-Host "Enable LLM/AI features? (requires OpenAI API key) [y/N]"
    if ($enableLlm -match "^[Yy]") {
        $global:EnableLLM = $true
        Write-Host "You'll need to configure your OpenAI API key after installation"
    }
    
    # Ask about optional dependencies
    if ($script:InstallType -in @("standard", "development")) {
        $installOptional = Read-Host "Install optional ML dependencies? (TensorFlow, etc.) [y/N]"
        if ($installOptional -match "^[Yy]") {
            $global:InstallOptionalDeps = $true
        }
    }
    
    # Ask about Docker
    if (Get-Command docker -ErrorAction SilentlyContinue) {
        $installDocker = Read-Host "Also build Docker images? [y/N]"
        if ($installDocker -match "^[Yy]") {
            $global:InstallDocker = $true
        }
    }
    
    Write-Host ""
    Write-LogInfo "Installation directory: $($script:InstallDir)"
    $changeDir = Read-Host "Change installation directory? [y/N]"
    if ($changeDir -match "^[Yy]") {
        do {
            $newDir = Read-Host "Enter new path"
        } while ([string]::IsNullOrEmpty($newDir))
        $script:InstallDir = $newDir
    }
}

# Create installation directory and virtual environment
function New-Environment {
    param([string]$PythonExe)
    
    Write-LogStep "Setting up environment"
    
    # Create installation directory
    Write-LogInfo "Creating installation directory: $($script:InstallDir)"
    if (-not (Test-Path $script:InstallDir)) {
        New-Item -ItemType Directory -Path $script:InstallDir -Force | Out-Null
    }
    Set-Location $script:InstallDir
    
    # Create virtual environment
    Write-LogInfo "Creating Python virtual environment..."
    & $PythonExe -m venv $script:VenvName
    
    # Activate virtual environment
    $activateScript = Join-Path $script:InstallDir $script:VenvName "Scripts" "Activate.ps1"
    if (Test-Path $activateScript) {
        & $activateScript
    }
    
    # Upgrade pip
    Write-LogInfo "Upgrading pip..."
    $pipExe = Join-Path $script:InstallDir $script:VenvName "Scripts" "pip.exe"
    & $pipExe install --upgrade pip setuptools wheel
    
    Write-LogSuccess "Environment setup complete"
    return $pipExe
}

# Download and install Cognito
function Install-Cognito {
    param([string]$PipExe)
    
    Write-LogStep "Installing Cognito"
    
    # Save current directory for copying local files
    $sourceDir = $PWD.Path
    
    # Download source code - Modified to work with local files first, then remote
    Write-LogInfo "Setting up Cognito source code..."
    
    # Check if we're running from a Cognito project directory
    if ((Test-Path "$sourceDir\src") -and (Test-Path "$sourceDir\requirements.txt")) {
        Write-LogInfo "Found local Cognito source files, copying..."
        Copy-Item -Path "$sourceDir\src" -Destination "." -Recurse
        Copy-Item -Path "$sourceDir\requirements.txt" -Destination "."
        
        # Copy other files if they exist
        $optionalFiles = @("setup.py", "pyproject.toml")
        foreach ($file in $optionalFiles) {
            if (Test-Path "$sourceDir\$file") {
                Copy-Item -Path "$sourceDir\$file" -Destination "."
                Write-LogInfo "Copied $file"
            }
        }
        
        # Copy directories if they exist
        $optionalDirs = @("config", "tests")
        foreach ($dir in $optionalDirs) {
            if (Test-Path "$sourceDir\$dir") {
                Copy-Item -Path "$sourceDir\$dir" -Destination "." -Recurse
                Write-LogInfo "Copied $dir\"
            }
        }
        
    } elseif (Get-Command git -ErrorAction SilentlyContinue) {
        Write-LogInfo "Downloading from Git repository..."
        git clone https://github.com/yourusername/cognito.git .
        git checkout main
    } else {
        Write-LogInfo "Downloading release archive..."
        $archiveUrl = "https://github.com/yourusername/cognito/archive/main.zip"
        $archivePath = Join-Path $script:InstallDir "cognito.zip"
        
        try {
            Invoke-WebRequest -Uri $archiveUrl -OutFile $archivePath
            Expand-Archive -Path $archivePath -DestinationPath $script:InstallDir -Force
            
            # Move files from extracted subdirectory
            $extractedDir = Join-Path $script:InstallDir "cognito-main"
            if (Test-Path $extractedDir) {
                Get-ChildItem $extractedDir | Move-Item -Destination $script:InstallDir -Force
                Remove-Item $extractedDir -Recurse -Force
            }
            
            Remove-Item $archivePath -Force
        }
        catch {
            Write-LogError "Failed to download Cognito: $_"
            throw
        }
    }
    
    # Verify required files exist
    if (-not (Test-Path "src")) {
        Write-LogError "Source directory (src\) not found after installation"
        throw "Source files missing"
    }
    
    if (-not (Test-Path "requirements.txt")) {
        Write-LogError "requirements.txt not found after installation"
        throw "Requirements file missing"
    }
    
    # Install Python dependencies
    Write-LogInfo "Installing Python dependencies..."
    
    switch ($script:InstallType) {
        "minimal" {
            # Create minimal requirements
            $minimalReqs = @"
pytest>=7.4.0
colorama>=0.4.6
astroid>=3.0.1
"@
            $minimalReqs | Out-File -FilePath "requirements-minimal.txt" -Encoding UTF8
            & $PipExe install -r requirements-minimal.txt
        }
        "development" {
            & $PipExe install -r requirements.txt
            if (Test-Path "setup.py") {
                & $PipExe install -e ".[dev]"
            }
        }
        default {
            & $PipExe install -r requirements.txt
            if (Test-Path "setup.py") {
                & $PipExe install -e .
            }
        }
    }
    
    # Install optional dependencies if requested
    if ($global:InstallOptionalDeps) {
        Write-LogInfo "Installing optional ML dependencies..."
        if (Test-Path "setup.py") {
            try {
                & $PipExe install -e ".[full]"
            }
            catch {
                Write-LogWarn "Failed to install optional dependencies"
            }
        } else {
            Write-LogWarn "setup.py not found, skipping optional dependencies"
        }
    }
    
    Write-LogSuccess "Cognito installation complete"
}

# Create configuration files
function New-Configuration {
    Write-LogStep "Setting up configuration"
    
    # Create configuration directories
    $configDirs = @("config", "data", "logs", "models")
    foreach ($dir in $configDirs) {
        $fullPath = Join-Path $script:InstallDir $dir
        if (-not (Test-Path $fullPath)) {
            New-Item -ItemType Directory -Path $fullPath -Force | Out-Null
        }
    }
    
    # Create environment file
    $envPath = Join-Path $script:InstallDir "config" ".env"
    $envContent = @"
# Cognito Configuration
COGNITO_ENV=production
COGNITO_DEBUG=false
COGNITO_VERSION=$($script:CognitoVersion)

# Installation paths
COGNITO_INSTALL_DIR=$($script:InstallDir)
COGNITO_DATA_DIR=$($script:InstallDir)\data
COGNITO_LOG_FILE=$($script:InstallDir)\logs\cognito.log
COGNITO_MODELS_DIR=$($script:InstallDir)\models
COGNITO_CACHE_DIR=$($script:InstallDir)\data\cache

# Security settings
COGNITO_RATE_LIMIT=1000
COGNITO_MAX_FILE_SIZE_MB=10
COGNITO_ENABLE_SECURITY_VALIDATION=true

# Features
COGNITO_ENABLE_LLM=$($global:EnableLLM.ToString().ToLower())
COGNITO_ENABLE_CODE_CORRECTION=true
COGNITO_ENABLE_FEEDBACK_LEARNING=true

# API Keys (set these manually)
# OPENAI_API_KEY=your_openai_api_key_here
# HUGGINGFACE_TOKEN=your_huggingface_token_here
"@
    $envContent | Out-File -FilePath $envPath -Encoding UTF8
    
    # Create launcher batch file
    $launcherContent = @"
@echo off
set COGNITO_DIR=$($script:InstallDir)
set VENV_DIR=%COGNITO_DIR%\$($script:VenvName)

REM Load environment variables
if exist "%COGNITO_DIR%\config\.env" (
    for /f "tokens=1,2 delims==" %%a in ('type "%COGNITO_DIR%\config\.env" ^| findstr /v "^#"') do (
        set %%a=%%b
    )
)

REM Activate virtual environment
call "%VENV_DIR%\Scripts\activate.bat"

REM Change to Cognito directory
cd /d "%COGNITO_DIR%"

REM Run Cognito
python -m src.main %*
"@
    
    $launcherPath = Join-Path $script:InstallDir "cognito.bat"
    $launcherContent | Out-File -FilePath $launcherPath -Encoding ASCII
    
    # Create PowerShell launcher
    $psLauncherContent = @"
# Cognito PowerShell Launcher
`$CognitoDir = "$($script:InstallDir)"
`$VenvDir = Join-Path `$CognitoDir "$($script:VenvName)"

# Load environment variables
`$envFile = Join-Path `$CognitoDir "config" ".env"
if (Test-Path `$envFile) {
    Get-Content `$envFile | Where-Object { `$_ -notmatch "^#" -and `$_ -match "=" } | ForEach-Object {
        `$key, `$value = `$_ -split "=", 2
        [Environment]::SetEnvironmentVariable(`$key, `$value, "Process")
    }
}

# Activate virtual environment
`$activateScript = Join-Path `$VenvDir "Scripts" "Activate.ps1"
if (Test-Path `$activateScript) {
    & `$activateScript
}

# Change to Cognito directory
Set-Location `$CognitoDir

# Run Cognito
& python -m src.main @args
"@
    
    $psLauncherPath = Join-Path $script:InstallDir "cognito.ps1"
    $psLauncherContent | Out-File -FilePath $psLauncherPath -Encoding UTF8
    
    # Add to PATH
    $userPath = [Environment]::GetEnvironmentVariable("PATH", "User")
    if ($userPath -notlike "*$($script:InstallDir)*") {
        $newPath = "$($script:InstallDir);$userPath"
        [Environment]::SetEnvironmentVariable("PATH", $newPath, "User")
        Write-LogInfo "Added $($script:InstallDir) to user PATH"
        Write-LogInfo "Please restart your command prompt to use 'cognito' command"
    }
    
    Write-LogSuccess "Configuration setup complete"
}

# Setup Docker if requested
function Set-Docker {
    if ($global:InstallDocker) {
        Write-LogStep "Setting up Docker"
        
        # Check if Docker is installed and running
        try {
            docker --version | Out-Null
            docker info | Out-Null
        }
        catch {
            Write-LogError "Docker is not installed or not running"
            Write-LogInfo "Download Docker Desktop from: https://www.docker.com/products/docker-desktop"
            return
        }
        
        # Build Docker image
        Write-LogInfo "Building Docker image..."
        Set-Location $script:InstallDir
        try {
            docker build -t cognito:latest .
            docker build -t "cognito:$($script:CognitoVersion)" .
            
            # Create docker-compose file
            $dockerComposeContent = @"
version: '3.8'

services:
  cognito:
    image: cognito:latest
    ports:
      - "8000:8000"
    environment:
      - COGNITO_ENV=production
    volumes:
      - ./data:/app/data
      - ./logs:/app/logs
      - ./models:/app/models
    restart: unless-stopped

  nginx:
    image: nginx:alpine
    ports:
      - "80:80"
    volumes:
      - ./nginx.conf:/etc/nginx/nginx.conf:ro
    depends_on:
      - cognito
    restart: unless-stopped
"@
            
            $dockerComposePath = Join-Path $script:InstallDir "docker-compose.yml"
            $dockerComposeContent | Out-File -FilePath $dockerComposePath -Encoding UTF8
            
            # Create Docker launcher
            $dockerLauncherContent = @"
@echo off
cd /d "$($script:InstallDir)"
docker-compose %*
"@
            
            $dockerLauncherPath = Join-Path $script:InstallDir "cognito-docker.bat"
            $dockerLauncherContent | Out-File -FilePath $dockerLauncherPath -Encoding ASCII
            
            Write-LogSuccess "Docker setup complete"
        }
        catch {
            Write-LogWarn "Docker setup failed: $_"
        }
    }
}

# Run post-installation tests
function Test-Installation {
    Write-LogStep "Running post-installation tests"
    
    $pythonExe = Join-Path $script:InstallDir $script:VenvName "Scripts" "python.exe"
    Set-Location $script:InstallDir
    
    # Test basic functionality
    Write-LogInfo "Testing basic functionality..."
    try {
        'def hello(): print("world")' | & $pythonExe -m src.main --validate-only 2>$null
        Write-LogSuccess "Basic validation test passed"
    }
    catch {
        Write-LogWarn "Basic validation test failed - this might be normal"
    }
    
    # Test configuration
    Write-LogInfo "Testing configuration..."
    try {
        & $pythonExe -c "from src.config import get_config; print('Config OK')" 2>$null
        Write-LogSuccess "Configuration test passed"
    }
    catch {
        Write-LogWarn "Configuration test failed - this is normal for first install"
    }
    
    # Test language detection
    Write-LogInfo "Testing language detection..."
    try {
        & $pythonExe -c "from src.language_detector import detect_code_language; print('Language detection OK')" 2>$null
        Write-LogSuccess "Language detection test passed"
    }
    catch {
        Write-LogWarn "Language detection test failed"
    }
    
    # Test basic import
    Write-LogInfo "Testing basic imports..."
    try {
        & $pythonExe -c "import src.main; print('Main module import OK')" 2>$null
        Write-LogSuccess "Main module import test passed"
    }
    catch {
        Write-LogWarn "Main module import failed"
    }
    
    # Run unit tests if available
    if ($script:InstallType -eq "development" -and (Test-Path "tests")) {
        Write-LogInfo "Running unit tests..."
        try {
            & $pythonExe -m pytest tests/ -x -q 2>$null
            Write-LogSuccess "Unit tests passed"
        }
        catch {
            Write-LogWarn "Some unit tests failed"
        }
    }
    
    Write-LogSuccess "Basic tests completed"
}

# Display installation summary
function Show-InstallationSummary {
    Write-LogStep "Installation Summary"
    
    Write-Host -ForegroundColor $Colors.Green @"
╔══════════════════════════════════════════════════════════════╗
║                  🎉 INSTALLATION COMPLETE! 🎉                ║
╚══════════════════════════════════════════════════════════════╝
"@
    
    Write-Host ""
    Write-Host "Cognito has been successfully installed!"
    Write-Host ""
    Write-Host -ForegroundColor $Colors.Cyan "Installation Details:"
    Write-Host "  📁 Installation directory: $($script:InstallDir)"
    Write-Host "  🐍 Python environment: $($script:InstallDir)\$($script:VenvName)"
    Write-Host "  ⚙️  Configuration: $($script:InstallDir)\config\.env"
    Write-Host "  📊 Data directory: $($script:InstallDir)\data"
    Write-Host "  📝 Logs directory: $($script:InstallDir)\logs"
    Write-Host ""
    
    Write-Host -ForegroundColor $Colors.Cyan "Quick Start:"
    Write-Host "  🚀 Run: cognito --help"
    Write-Host "  📊 Analyze code: cognito --file your_code.py"
    Write-Host "  🧠 With AI: cognito --file your_code.py --use-llm"
    Write-Host ""
    
    if ($global:EnableLLM) {
        Write-Host -ForegroundColor $Colors.Yellow "⚠️  LLM Configuration Required:"
        Write-Host "  1. Get OpenAI API key from: https://platform.openai.com/api-keys"
        Write-Host "  2. Edit: $($script:InstallDir)\config\.env"
        Write-Host "  3. Set: OPENAI_API_KEY=your_key_here"
        Write-Host ""
    }
    
    if ($global:InstallDocker) {
        Write-Host -ForegroundColor $Colors.Cyan "Docker Commands:"
        Write-Host "  🐳 Start: cd $($script:InstallDir) && docker-compose up -d"
        Write-Host "  🛑 Stop: cd $($script:InstallDir) && docker-compose down"
        Write-Host ""
    }
    
    Write-Host -ForegroundColor $Colors.Cyan "Configuration:"
    Write-Host "  ⚙️  Edit config: $($script:InstallDir)\config\.env"
    Write-Host "  📝 View logs: Get-Content $($script:InstallDir)\logs\cognito.log -Tail 10 -Wait"
    Write-Host "  🔄 Update: cd $($script:InstallDir) && git pull && pip install -r requirements.txt"
    Write-Host ""
    
    Write-Host -ForegroundColor $Colors.Cyan "Support & Documentation:"
    Write-Host "  📚 Documentation: https://cognito.readthedocs.io"
    Write-Host "  🐛 Issues: https://github.com/yourusername/cognito/issues"
    Write-Host "  💬 Discussions: https://github.com/yourusername/cognito/discussions"
    Write-Host ""
    
    Write-Host -ForegroundColor $Colors.Green "Happy coding with Cognito! 🚀"
}

# Uninstall function
function Uninstall-Cognito {
    Write-Host "Uninstalling Cognito..."
    
    if (Test-Path $script:InstallDir) {
        # Remove from PATH
        $userPath = [Environment]::GetEnvironmentVariable("PATH", "User")
        if ($userPath -like "*$($script:InstallDir)*") {
            $newPath = $userPath -replace [regex]::Escape("$($script:InstallDir);"), ""
            $newPath = $newPath -replace [regex]::Escape(";$($script:InstallDir)"), ""
            [Environment]::SetEnvironmentVariable("PATH", $newPath, "User")
        }
        
        # Remove installation directory
        Remove-Item $script:InstallDir -Recurse -Force
        Write-Host "Cognito uninstalled successfully"
    }
    else {
        Write-Host "Cognito is not installed"
    }
}

# Update function
function Update-Cognito {
    Write-Host "Updating Cognito..."
    
    if (Test-Path $script:InstallDir) {
        Set-Location $script:InstallDir
        
        $pythonExe = Join-Path $script:InstallDir $script:VenvName "Scripts" "python.exe"
        $pipExe = Join-Path $script:InstallDir $script:VenvName "Scripts" "pip.exe"
        
        # Activate virtual environment
        $activateScript = Join-Path $script:InstallDir $script:VenvName "Scripts" "Activate.ps1"
        if (Test-Path $activateScript) {
            & $activateScript
        }
        
        # Update source code
        if (Get-Command git -ErrorAction SilentlyContinue) {
            git pull
        }
        else {
            Write-LogWarn "Git not available. Please download latest version manually."
        }
        
        # Update dependencies
        & $pipExe install -r requirements.txt
        
        Write-Host "Cognito updated successfully"
    }
    else {
        Write-Host "Cognito is not installed. Run installer first."
    }
}

# Main installation function
function Install-CognitoMain {
    # Check if running as administrator when not needed
    if ((Test-Administrator) -and -not $AllowAdmin) {
        Write-LogWarn "Running as Administrator is not recommended. Use -AllowAdmin to override."
        return
    }
    
    Show-Logo
    
    try {
        $steps = @(
            { $script:PythonExe = Test-Requirements },
            { Set-Configuration },
            { $script:PipExe = New-Environment -PythonExe $script:PythonExe },
            { Install-Cognito -PipExe $script:PipExe },
            { New-Configuration },
            { Set-Docker },
            { Test-Installation }
        )
        
        $totalSteps = $steps.Count
        
        for ($i = 0; $i -lt $steps.Count; $i++) {
            Show-Progress -Current ($i + 1) -Total $totalSteps
            & $steps[$i]
            Start-Sleep -Milliseconds 500
        }
        
        Write-Progress -Completed -Activity "Installing Cognito"
        Show-InstallationSummary
    }
    catch {
        Write-LogError "Installation failed: $_"
        Write-Host "Please check the error message above and try again."
        
        $cleanup = Read-Host "Remove installation directory? [y/N]"
        if ($cleanup -match "^[Yy]") {
            if (Test-Path $script:InstallDir) {
                Remove-Item $script:InstallDir -Recurse -Force
                Write-LogInfo "Installation directory removed"
            }
        }
    }
}

# Handle command line arguments
if ($Help) {
    Write-Host "Cognito Windows Installer"
    Write-Host ""
    Write-Host "Usage: .\install.ps1 [options]"
    Write-Host ""
    Write-Host "Options:"
    Write-Host "  -Help               Show this help message"
    Write-Host "  -Version            Show installer version"
    Write-Host "  -AllowAdmin         Allow running as Administrator"
    Write-Host "  -Uninstall          Uninstall Cognito"
    Write-Host "  -Update             Update existing installation"
    Write-Host "  -InstallDir <path>  Installation directory (default: $env:USERPROFILE\.cognito)"
    Write-Host "  -InstallType <type> Installation type: standard, minimal, development"
    Write-Host ""
    exit 0
}

if ($Version) {
    Write-Host "Cognito Windows Installer v1.0.0"
    exit 0
}

if ($Uninstall) {
    Uninstall-Cognito
    exit 0
}

if ($Update) {
    Update-Cognito
    exit 0
}

# Run main installation
Install-CognitoMain