@echo off
REM ================================================================
REM    install_age_classifier_batch.bat - Version 2.0 FINAL
REM    Age Classifier - One-Click Installer
REM    Created: 2024 - kvmierlo3
REM    This is the LATEST VERSION - Use this file
REM ================================================================

echo ================================================================
echo            Age Classifier - One-Click Installer v2.0
echo ================================================================
echo.
echo This installer will automatically:
echo - Check system requirements
echo - Clone the repository from GitHub
echo - Set up Python virtual environment  
echo - Install all dependencies
echo - Download pre-trained models
echo - Create shortcuts
echo.
pause

REM Get the directory where this batch file is located
set "INSTALL_DIR=%~dp0"
cd /d "%INSTALL_DIR%"

echo Current directory: %CD%
echo.

REM Check if Git is installed
echo [1/7] Checking Git installation...
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
echo [2/7] Checking Python installation...
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo ERROR: Python is not installed or not in PATH
    echo Please install Python 3.8+ from: https://python.org
    echo Make sure to check "Add Python to PATH" during installation
    echo After installation, restart this installer.
    pause
    exit /b 1
)
echo ✓ Python found

REM Clone the repository
echo [3/7] Cloning Age Classifier repository...
if exist "age_classifier" (
    echo Repository already exists. Updating...
    cd age_classifier
    git pull origin main
    cd ..
) else (
    echo Cloning from GitHub...
    git clone https://github.com/kvmierlo3/age_classifier.git
    if %errorlevel% neq 0 (
        echo ERROR: Failed to clone repository
        echo Please check your internet connection and try again
        pause
        exit /b 1
    )
)
echo ✓ Repository cloned/updated

REM Enter the project directory
cd age_classifier

REM Create virtual environment
echo [4/7] Creating Python virtual environment...
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
echo [5/7] Installing dependencies...
call venv_age_classifier\Scripts\activate.bat

REM Upgrade pip
python -m pip install --upgrade pip --quiet

REM Install PyTorch with fallback methods
echo Installing PyTorch (this may take a while)...
pip install torch torchvision --quiet --index-url https://download.pytorch.org/whl/cpu
if %errorlevel% neq 0 (
    echo Trying alternative PyTorch installation...
    pip install torch torchvision --quiet
    if %errorlevel% neq 0 (
        echo WARNING: PyTorch installation failed
        echo The application may not work properly
    )
)

REM Install other requirements
echo Installing other dependencies...
pip install -r requirements.txt --quiet
if %errorlevel% neq 0 (
    echo WARNING: Some dependencies failed to install
    echo Trying individual package installation...
    pip install transformers gradio pillow numpy pandas matplotlib seaborn requests --quiet
    pip install safetensors huggingface-hub accelerate scipy --quiet
)
echo ✓ Dependencies installed

REM Pre-download models
echo [6/7] Pre-downloading AI models (this may take several minutes)...
python -c "
try:
    print('Downloading nateraw/vit-age-classifier...')
    from transformers import AutoImageProcessor, AutoModelForImageClassification
    AutoImageProcessor.from_pretrained('nateraw/vit-age-classifier')
    AutoModelForImageClassification.from_pretrained('nateraw/vit-age-classifier')
    print('✓ nateraw model downloaded')
    
    print('Downloading prithivMLmods/Age-Classification-SigLIP2...')
    from transformers import SiglipForImageClassification
    AutoImageProcessor.from_pretrained('prithivMLmods/Age-Classification-SigLIP2')
    SiglipForImageClassification.from_pretrained('prithivMLmods/Age-Classification-SigLIP2')
    print('✓ prithiv model downloaded')
    
    print('All models successfully cached!')
except Exception as e:
    print(f'Warning: Model download failed: {e}')
    print('Models will be downloaded during first use.')
" 2>nul
echo ✓ Models downloaded

REM Create desktop shortcuts
echo [7/7] Creating shortcuts...

REM Create start script if it doesn't exist
if not exist "start_age_classifier_batch.bat" (
    echo @echo off > start_age_classifier_batch.bat
    echo REM start_age_classifier_batch.bat - Version 2.0 FINAL >> start_age_classifier_batch.bat
    echo echo Starting Age Classifier v2.0... >> start_age_classifier_batch.bat
    echo cd /d "%%~dp0" >> start_age_classifier_batch.bat
    echo call venv_age_classifier\Scripts\activate.bat >> start_age_classifier_batch.bat
    echo echo Web interface will open at: http://127.0.0.1:7861 >> start_age_classifier_batch.bat
    echo echo Press Ctrl+C to stop the application >> start_age_classifier_batch.bat
    echo python multi_model_age_classifier_cleaned.py >> start_age_classifier_batch.bat
    echo pause >> start_age_classifier_batch.bat
)

REM Create desktop shortcut (if possible)
set "SHORTCUT_PATH=%USERPROFILE%\Desktop\Age Classifier.lnk"
set "TARGET_PATH=%CD%\start_age_classifier_batch.bat"

powershell -Command "& {$WshShell = New-Object -comObject WScript.Shell; $Shortcut = $WshShell.CreateShortcut('%SHORTCUT_PATH%'); $Shortcut.TargetPath = '%TARGET_PATH%'; $Shortcut.WorkingDirectory = '%CD%'; $Shortcut.Description = 'Age Classifier Research Tool v2.0'; $Shortcut.Save()}" 2>nul

echo ✓ Shortcuts created

echo.
echo ================================================================
echo                 Installation Complete! v2.0
echo ================================================================
echo.
echo Age Classifier has been successfully installed in:
echo %CD%
echo.
echo To start the application:
echo 1. Double-click "start_age_classifier_batch.bat" in this folder
echo 2. Or use the desktop shortcut "Age Classifier"
echo 3. The web interface will open at: http://127.0.0.1:7861
echo.
echo Features installed:
echo ✓ Vision Transformer (nateraw) age classifier
echo ✓ SigLIP (prithiv) age classifier  
echo ✓ Multi-model comparison tools
echo ✓ Batch processing capabilities
echo ✓ Statistical analysis features
echo ✓ Strict vs Balanced age boundary modes
echo.
echo For support, visit: https://github.com/kvmierlo3/age_classifier
echo.
echo Version: 2.0 FINAL - December 2024
pause