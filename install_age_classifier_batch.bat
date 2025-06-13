@echo off
REM ================================================================
REM    install_age_classifier_batch.bat - Version 3.0 SMART
REM    Age Classifier - Smart Installer (Non-Invasive)
REM    Created: 2024 - kvmierlo3
REM    This is the LATEST VERSION - Use this file
REM ================================================================

echo ================================================================
echo        Age Classifier - Smart Installer v3.0
echo ================================================================
echo.
echo Smart, non-invasive installation:
echo ✓ Detects existing Python (3.8+ compatible)
echo ✓ Only installs if needed
echo ✓ Offers portable vs system installation
echo ✓ Respects your existing setup
echo ✓ Network-resilient package installation
echo.
pause

REM Get the directory where this batch file is located
set "INSTALL_DIR=%~dp0"
cd /d "%INSTALL_DIR%"

echo Current directory: %CD%
echo.

REM ================================================================
REM                    SYSTEM REQUIREMENTS CHECK
REM ================================================================

echo [1/8] Analyzing system requirements...

REM Check Windows version
echo Checking Windows compatibility...
ver | findstr /i "Windows" >nul
if %errorlevel% neq 0 (
    echo WARNING: This installer is designed for Windows
    echo It may not work properly on other operating systems
    pause
)

REM ================================================================
REM                        GIT CHECK
REM ================================================================

echo [2/8] Checking Git installation...
git --version >nul 2>&1
if %errorlevel% neq 0 (
    echo.
    echo ⚠️  Git is not installed or not in PATH
    echo.
    echo Git is required to download the Age Classifier code.
    echo.
    choice /C YN /M "Open Git download page"
    if errorlevel 2 goto git_skip
    if errorlevel 1 (
        echo Opening Git download page...
        start https://git-scm.com/download/win
        echo.
        echo Please install Git and restart this installer.
        pause
        exit /b 1
    )
    
    :git_skip
    echo.
    echo Manual installation: https://git-scm.com/download/win
    echo Please install Git and restart this installer.
    pause
    exit /b 1
)
echo ✓ Git found

REM ================================================================
REM                    SMART PYTHON DETECTION
REM ================================================================

echo [3/8] Smart Python detection...

REM Check multiple Python commands (python, python3, py)
set "PYTHON_CMD="
set "PYTHON_VERSION="
set "PYTHON_COMPATIBLE=0"

echo Checking for Python installations...

REM Try 'python' command
python --version >nul 2>&1
if %errorlevel% equ 0 (
    for /f "tokens=2" %%i in ('python --version 2^>^&1') do set PYTHON_VERSION=%%i
    set "PYTHON_CMD=python"
    echo Found: python (%PYTHON_VERSION%)
    goto check_compatibility
)

REM Try 'python3' command
python3 --version >nul 2>&1
if %errorlevel% equ 0 (
    for /f "tokens=2" %%i in ('python3 --version 2^>^&1') do set PYTHON_VERSION=%%i
    set "PYTHON_CMD=python3"
    echo Found: python3 (%PYTHON_VERSION%)
    goto check_compatibility
)

REM Try 'py' launcher (Windows Python Launcher)
py --version >nul 2>&1
if %errorlevel% equ 0 (
    for /f "tokens=2" %%i in ('py --version 2^>^&1') do set PYTHON_VERSION=%%i
    set "PYTHON_CMD=py"
    echo Found: py launcher (%PYTHON_VERSION%)
    goto check_compatibility
)

echo ❌ No Python installation detected
goto python_install_options

:check_compatibility
echo.
echo Checking Python compatibility...
echo Found Python version: %PYTHON_VERSION%

REM Basic compatibility check (3.8+)
echo %PYTHON_VERSION% | findstr /r "^3\.[89]" >nul
if %errorlevel% equ 0 (
    set "PYTHON_COMPATIBLE=1"
    goto python_compatible
)

echo %PYTHON_VERSION% | findstr /r "^3\.1[0-9]" >nul
if %errorlevel% equ 0 (
    set "PYTHON_COMPATIBLE=1"
    goto python_compatible
)

echo %PYTHON_VERSION% | findstr /r "^3\.2[0-9]" >nul
if %errorlevel% equ 0 (
    set "PYTHON_COMPATIBLE=1"
    goto python_compatible
)

REM Version too old
echo.
echo ⚠️  Python %PYTHON_VERSION% is too old
echo Age Classifier requires Python 3.8 or newer
echo.
choice /C YN /M "Install newer Python version"
if errorlevel 2 goto python_continue_anyway
if errorlevel 1 goto python_install_options

:python_continue_anyway
echo Continuing with Python %PYTHON_VERSION% (may cause issues)...
set "PYTHON_COMPATIBLE=1"
goto python_compatible

:python_compatible
echo ✓ Compatible Python found: %PYTHON_CMD% (%PYTHON_VERSION%)
goto repository_setup

:python_install_options
echo.
echo ================================================================
echo                     PYTHON INSTALLATION OPTIONS
echo ================================================================
echo.
echo Choose Python installation method:
echo.
echo 1. PORTABLE (Recommended)
echo    - Downloads Python to project folder only
echo    - No system changes
echo    - Safe and contained
echo.
echo 2. SYSTEM-WIDE 
echo    - Installs Python globally on your PC
echo    - Available for other projects
echo    - Modifies system PATH
echo.
echo 3. MANUAL
echo    - Opens download page
echo    - You install manually
echo.
choice /C 123 /M "Choose installation method (1=Portable, 2=System, 3=Manual)"

if errorlevel 3 goto python_manual
if errorlevel 2 goto python_system
if errorlevel 1 goto python_portable

:python_portable
echo.
echo ================================================================
echo                    PORTABLE PYTHON INSTALLATION
echo ================================================================
echo.
echo Installing Python 3.9 portable (no system changes)...
echo.

REM Create python directory
mkdir "python_portable" 2>nul
cd python_portable

echo Downloading Python 3.9 portable...
powershell -Command "& {$url='https://www.python.org/ftp/python/3.9.18/python-3.9.18-embed-amd64.zip'; $output='python_portable.zip'; Invoke-WebRequest -Uri $url -OutFile $output; if (Test-Path $output) {Write-Host 'Download successful'} else {Write-Host 'Download failed'; exit 1}}"

if %errorlevel% neq 0 (
    echo Download failed. Trying alternative method...
    goto python_system
)

echo Extracting Python...
powershell -Command "Expand-Archive -Path 'python_portable.zip' -DestinationPath '.' -Force"

echo Setting up portable Python...
REM Configure portable Python
echo import site >> python39._pth
echo. >> python39._pth

echo Downloading pip for portable Python...
powershell -Command "Invoke-WebRequest -Uri 'https://bootstrap.pypa.io/get-pip.py' -OutFile 'get-pip.py'"
python.exe get-pip.py

cd ..
set "PYTHON_CMD=python_portable\python.exe"
set "PYTHON_VERSION=3.9.18"
set "PYTHON_COMPATIBLE=1"
echo ✓ Portable Python 3.9 installed successfully
goto repository_setup

:python_system
echo.
echo ================================================================
echo                   SYSTEM-WIDE PYTHON INSTALLATION  
echo ================================================================
echo.
echo Installing Python 3.9 system-wide...
echo This will modify your system PATH.
echo.

REM Check if winget is available (Windows 10/11)
winget --version >nul 2>&1
if %errorlevel% equ 0 (
    echo Using Windows Package Manager (winget)...
    winget install Python.Python.3.9 --silent --accept-package-agreements --accept-source-agreements
    if %errorlevel% equ 0 (
        echo ✓ Python installed successfully!
        echo Please restart this installer.
        pause
        exit /b 0
    )
)

echo Downloading Python 3.9 installer...
mkdir "%TEMP%\age_classifier_setup" 2>nul
cd /d "%TEMP%\age_classifier_setup"

powershell -Command "Invoke-WebRequest -Uri 'https://www.python.org/ftp/python/3.9.18/python-3.9.18-amd64.exe' -OutFile 'python_installer.exe'"

if %errorlevel% neq 0 (
    echo Download failed. Opening manual download page...
    start https://www.python.org/downloads/release/python-3918/
    goto python_manual
)

echo Installing Python 3.9...
python_installer.exe /quiet InstallAllUsers=0 PrependPath=1 Include_pip=1

if %errorlevel% equ 0 (
    echo ✓ Python installed successfully!
    cd /d "%INSTALL_DIR%"
    rmdir /s /q "%TEMP%\age_classifier_setup" 2>nul
    echo Please restart this installer.
    pause
    exit /b 0
) else (
    echo Installation failed.
    goto python_manual
)

:python_manual
echo.
echo ================================================================
echo                    MANUAL PYTHON INSTALLATION
echo ================================================================
echo.
echo Please install Python manually:
echo.
echo RECOMMENDED: Python 3.9.18 (best compatibility)
echo Minimum: Python 3.8+
echo.
echo 1. Go to: https://www.python.org/downloads/release/python-3918/
echo 2. Download "Windows installer (64-bit)"
echo 3. During installation, CHECK "Add Python to PATH"
echo 4. Restart this installer
echo.
start https://www.python.org/downloads/release/python-3918/
pause
exit /b 1

REM ================================================================
REM                    REPOSITORY SETUP
REM ================================================================

:repository_setup
echo [4/8] Setting up Age Classifier repository...
cd /d "%INSTALL_DIR%"

if exist "age_classifier" (
    echo Repository already exists. Updating...
    cd age_classifier
    git pull origin main >nul 2>&1
    cd ..
) else (
    echo Cloning from GitHub...
    git clone https://github.com/kvmierlo3/age_classifier.git
    if %errorlevel% neq 0 (
        echo ERROR: Failed to clone repository
        pause
        exit /b 1
    )
)
echo ✓ Repository ready

cd age_classifier

REM ================================================================
REM                   VIRTUAL ENVIRONMENT
REM ================================================================

echo [5/8] Creating Python virtual environment...
if exist "venv_age_classifier" (
    rmdir /s /q venv_age_classifier
)

%PYTHON_CMD% -m venv venv_age_classifier
if %errorlevel% neq 0 (
    echo ERROR: Failed to create virtual environment
    pause
    exit /b 1
)
echo ✓ Virtual environment created with %PYTHON_CMD%

REM Activate virtual environment
call venv_age_classifier\Scripts\activate.bat

REM ================================================================
REM                   ROBUST PACKAGE INSTALLATION
REM ================================================================

echo [6/8] Installing packages (network-resilient)...

echo Upgrading pip...
python -m pip install --upgrade pip --timeout 120 --retries 3 >nul

echo Installing PyTorch (Method 1/4)...
pip install torch torchvision --timeout 300 --retries 5 --index-url https://download.pytorch.org/whl/cpu >nul 2>&1
if %errorlevel% neq 0 (
    echo Method 1 failed, trying Method 2...
    pip install torch torchvision --timeout 300 --retries 5 >nul 2>&1
    if %errorlevel% neq 0 (
        echo Method 2 failed, trying Method 3...
        pip install torch torchvision --timeout 300 --retries 5 --trusted-host pypi.org --trusted-host pypi.python.org --trusted-host files.pythonhosted.org >nul 2>&1
    )
)

echo Installing other packages...
pip install transformers gradio pillow numpy pandas matplotlib seaborn requests safetensors huggingface-hub accelerate scipy --timeout 300 --retries 3 >nul

echo [7/8] Testing installation...
python -c "
try:
    import torch, transformers, gradio
    print('✓ All critical packages imported successfully')
except ImportError as e:
    print(f'⚠️ Import failed: {e}')
"

echo [8/8] Creating shortcuts...
echo @echo off > start_age_classifier_batch.bat
echo cd /d "%%~dp0" >> start_age_classifier_batch.bat
echo call venv_age_classifier\Scripts\activate.bat >> start_age_classifier_batch.bat
echo python multi_model_age_classifier_cleaned.py >> start_age_classifier_batch.bat
echo pause >> start_age_classifier_batch.bat

echo.
echo ================================================================
echo              Installation Complete! v3.0 SMART
echo ================================================================
echo.
echo ✅ SMART INSTALLATION SUMMARY:
echo ✓ Python: %PYTHON_CMD% (%PYTHON_VERSION%)
echo ✓ Installation method: Non-invasive detection
echo ✓ Repository: Cloned and ready
echo ✓ Dependencies: Installed with network resilience
echo ✓ Models: Will download on first use
echo.
echo 🚀 TO START: Double-click "start_age_classifier_batch.bat"
echo 🌐 WEB INTERFACE: http://127.0.0.1:7861
echo.
echo 📧 SUPPORT: https://github.com/kvmierlo3/age_classifier
echo 📅 Version: 3.0 SMART - December 2024
echo.
pause
