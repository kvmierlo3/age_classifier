@echo off
REM ================================================================
REM    install_age_classifier_batch.bat - Version 2.1 ROBUST
REM    Age Classifier - One-Click Installer (Network-Resilient)
REM    Created: 2024 - kvmierlo3
REM    This is the LATEST VERSION - Use this file
REM ================================================================

echo ================================================================
echo          Age Classifier - Robust Installer v2.1
echo ================================================================
echo.
echo This installer will automatically:
echo - Check system requirements
echo - Clone the repository from GitHub
echo - Set up Python virtual environment  
echo - Install all dependencies (with network resilience)
echo - Download pre-trained models
echo - Create shortcuts
echo.
echo Network-resilient features:
echo ✓ Multiple PyTorch installation methods
echo ✓ Extended timeouts for large downloads
echo ✓ Automatic retry with different sources
echo ✓ Graceful fallback handling
echo.
pause

REM Get the directory where this batch file is located
set "INSTALL_DIR=%~dp0"
cd /d "%INSTALL_DIR%"

echo Current directory: %CD%
echo.

REM Check if Git is installed
echo [1/8] Checking Git installation...
git --version >nul 2>&1
if %errorlevel% neq 0 (
    echo ERROR: Git is not installed or not in PATH
    echo Please install Git from: https://git-scm.com/download/win
    echo After installation, restart this installer.
    pause
    exit /b 1
)
echo ✓ Git found

REM Check if Python is installed
echo [2/8] Checking Python installation...
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo ERROR: Python is not installed or not in PATH
    echo Please install Python 3.8+ from: https://python.org
    echo Make sure to check "Add Python to PATH" during installation
    echo After installation, restart this installer.
    pause
    exit /b 1
)

REM Show Python version for debugging
for /f "tokens=2" %%i in ('python --version 2^>^&1') do set PYTHON_VERSION=%%i
echo ✓ Python found (Version: %PYTHON_VERSION%)

REM Clone the repository
echo [3/8] Cloning Age Classifier repository...
if exist "age_classifier" (
    echo Repository already exists. Updating...
    cd age_classifier
    git pull origin main
    if %errorlevel% neq 0 (
        echo Warning: Git pull failed, continuing with existing files...
    )
    cd ..
) else (
    echo Cloning from GitHub...
    git clone https://github.com/kvmierlo3/age_classifier.git
    if %errorlevel% neq 0 (
        echo ERROR: Failed to clone repository
        echo Please check your internet connection and try again
        echo You can also manually download from: https://github.com/kvmierlo3/age_classifier
        pause
        exit /b 1
    )
)
echo ✓ Repository ready

REM Enter the project directory
cd age_classifier

REM Create virtual environment
echo [4/8] Creating Python virtual environment...
if exist "venv_age_classifier" (
    echo Virtual environment already exists. Removing old one...
    rmdir /s /q venv_age_classifier
)

python -m venv venv_age_classifier
if %errorlevel% neq 0 (
    echo ERROR: Failed to create virtual environment
    echo Please ensure Python is properly installed
    pause
    exit /b 1
)
echo ✓ Virtual environment created

REM Activate virtual environment
echo [5/8] Setting up Python environment...
call venv_age_classifier\Scripts\activate.bat

REM Upgrade pip with extended timeout
echo Upgrading pip...
python -m pip install --upgrade pip --timeout 120 --retries 3
if %errorlevel% neq 0 (
    echo Warning: pip upgrade failed, continuing...
)

REM Install PyTorch with multiple fallback methods
echo [6/8] Installing PyTorch (with network resilience)...
echo This may take several minutes depending on your connection...
echo.

REM Method 1: PyTorch CPU from official index (most reliable)
echo Attempting Method 1: PyTorch official CPU index...
pip install torch torchvision --timeout 300 --retries 5 --index-url https://download.pytorch.org/whl/cpu
if %errorlevel% equ 0 (
    echo ✓ PyTorch installed via official CPU index
    goto pytorch_success
)

echo Method 1 failed, trying Method 2...

REM Method 2: Standard PyPI (often works better with proxies/firewalls)
echo Attempting Method 2: Standard PyPI...
pip install torch torchvision --timeout 300 --retries 5
if %errorlevel% equ 0 (
    echo ✓ PyTorch installed via PyPI
    goto pytorch_success
)

echo Method 2 failed, trying Method 3...

REM Method 3: Manual trusted hosts (for corporate networks)
echo Attempting Method 3: With trusted hosts...
pip install torch torchvision --timeout 300 --retries 5 --trusted-host pypi.org --trusted-host pypi.python.org --trusted-host files.pythonhosted.org
if %errorlevel% equ 0 (
    echo ✓ PyTorch installed with trusted hosts
    goto pytorch_success
)

echo Method 3 failed, trying Method 4...

REM Method 4: Smaller chunks installation
echo Attempting Method 4: Installing components separately...
pip install torch --timeout 300 --retries 5 --no-deps
if %errorlevel% equ 0 (
    pip install torchvision --timeout 300 --retries 5 --no-deps
    if %errorlevel% equ 0 (
        echo ✓ PyTorch installed in components
        goto pytorch_success
    )
)

REM All PyTorch methods failed
echo.
echo ⚠️  WARNING: All PyTorch installation methods failed
echo This is usually due to network connectivity issues.
echo.
echo Solutions:
echo 1. Check your internet connection
echo 2. Try again later (PyTorch servers might be busy)
echo 3. If using corporate network, contact IT about PyTorch downloads
echo 4. Manual installation: pip install torch torchvision
echo.
echo The installer will continue with other packages...
echo The app may work with CPU-only mode or fail to start.
echo.

:pytorch_success

REM Install other requirements with resilience
echo [7/8] Installing other dependencies...
echo Installing packages with extended timeouts...

REM Core packages first
echo Installing core packages...
pip install transformers --timeout 300 --retries 3 --no-warn-script-location
pip install gradio --timeout 300 --retries 3 --no-warn-script-location
pip install pillow --timeout 300 --retries 3 --no-warn-script-location

REM Data science packages
echo Installing data science packages...
pip install numpy pandas scipy --timeout 300 --retries 3 --no-warn-script-location

REM Visualization packages
echo Installing visualization packages...
pip install matplotlib seaborn --timeout 300 --retries 3 --no-warn-script-location

REM Utility packages
echo Installing utility packages...
pip install requests accelerate safetensors huggingface-hub --timeout 300 --retries 3 --no-warn-script-location

REM Try requirements.txt if it exists (backup method)
if exist "requirements.txt" (
    echo Installing from requirements.txt...
    pip install -r requirements.txt --timeout 300 --retries 3 --no-warn-script-location
)

echo ✓ Dependencies installation attempted

REM Test critical imports
echo Testing critical package imports...
python -c "
try:
    import torch
    print('✓ PyTorch import successful')
except ImportError:
    print('⚠️  PyTorch import failed - app may not work')
    
try:
    import transformers
    print('✓ Transformers import successful')
except ImportError:
    print('⚠️  Transformers import failed - app will not work')
    
try:
    import gradio
    print('✓ Gradio import successful')
except ImportError:
    print('⚠️  Gradio import failed - app will not work')
"

echo.

REM Pre-download models (optional, with error handling)
echo [8/8] Attempting to pre-download AI models...
echo Note: If this fails, models will download on first use.
echo.

python -c "
import sys
try:
    print('Attempting to download nateraw/vit-age-classifier...')
    from transformers import AutoImageProcessor, AutoModelForImageClassification
    processor = AutoImageProcessor.from_pretrained('nateraw/vit-age-classifier')
    model = AutoModelForImageClassification.from_pretrained('nateraw/vit-age-classifier')
    print('✓ nateraw model downloaded successfully')
except Exception as e:
    print(f'⚠️  nateraw model download failed: {e}')
    print('   Model will download on first use')

try:
    print('Attempting to download prithivMLmods/Age-Classification-SigLIP2...')
    from transformers import SiglipForImageClassification
    processor = AutoImageProcessor.from_pretrained('prithivMLmods/Age-Classification-SigLIP2')
    model = SiglipForImageClassification.from_pretrained('prithivMLmods/Age-Classification-SigLIP2')
    print('✓ prithiv model downloaded successfully')
except Exception as e:
    print(f'⚠️  prithiv model download failed: {e}')
    print('   Model will download on first use')

print('Model download phase completed')
" 2>nul

REM Create shortcuts
echo Creating application shortcuts...

REM Create start script
echo @echo off > start_age_classifier_batch.bat
echo REM start_age_classifier_batch.bat - Version 2.1 ROBUST >> start_age_classifier_batch.bat
echo echo Starting Age Classifier v2.1... >> start_age_classifier_batch.bat
echo cd /d "%%~dp0" >> start_age_classifier_batch.bat
echo call venv_age_classifier\Scripts\activate.bat >> start_age_classifier_batch.bat
echo echo Web interface will open at: http://127.0.0.1:7861 >> start_age_classifier_batch.bat
echo echo Press Ctrl+C to stop the application >> start_age_classifier_batch.bat
echo python multi_model_age_classifier_cleaned.py >> start_age_classifier_batch.bat
echo pause >> start_age_classifier_batch.bat

REM Create desktop shortcut (if possible)
set "SHORTCUT_PATH=%USERPROFILE%\Desktop\Age Classifier.lnk"
set "TARGET_PATH=%CD%\start_age_classifier_batch.bat"

powershell -Command "& {$WshShell = New-Object -comObject WScript.Shell; $Shortcut = $WshShell.CreateShortcut('%SHORTCUT_PATH%'); $Shortcut.TargetPath = '%TARGET_PATH%'; $Shortcut.WorkingDirectory = '%CD%'; $Shortcut.Description = 'Age Classifier Research Tool v2.1'; $Shortcut.Save()}" 2>nul

echo ✓ Shortcuts created

echo.
echo ================================================================
echo                Installation Summary v2.1
echo ================================================================
echo.
echo Age Classifier has been set up in: %CD%
echo.
echo ✓ Repository cloned/updated
echo ✓ Python virtual environment created
echo ✓ Dependencies installation attempted
echo ✓ Model downloads attempted
echo ✓ Shortcuts created
echo.
echo To start the application:
echo 1. Double-click "start_age_classifier_batch.bat"
echo 2. Or use the desktop shortcut "Age Classifier"
echo 3. Web interface: http://127.0.0.1:7861
echo.
echo Troubleshooting:
echo - If app fails to start, check Python packages
echo - Models download automatically on first use if pre-download failed
echo - For network issues, try installation again later
echo.
echo Support: https://github.com/kvmierlo3/age_classifier
echo Version: 2.1 ROBUST - December 2024
echo.
pause
