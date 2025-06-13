@echo off
REM ================================================================
REM    start_age_classifier_batch.bat - Version 2.2 ULTRA-ROBUST
REM    Age Classifier - Application Launcher
REM    Created: 2024 - kvmierlo3
REM    This is the LATEST VERSION - Use this file
REM ================================================================

echo ================================================================
echo            Age Classifier - Starting Application v2.0
echo ================================================================
echo.

REM Get the directory where this batch file is located
cd /d "%~dp0"

REM Check if virtual environment exists
if not exist "venv_age_classifier" (
    echo ERROR: Virtual environment not found!
    echo Please run install_age_classifier_batch.bat first to set up the environment.
    echo.
    echo Download the installer from:
    echo https://github.com/kvmierlo3/age_classifier/raw/main/install_age_classifier_batch.bat
    echo.
    pause
    exit /b 1
)

REM Check if main script exists
if not exist "multi_model_age_classifier.py" (
    echo ERROR: Main application file not found!
    echo Please ensure all files are in the same directory.
    echo Try running install_age_classifier_batch.bat again.
    echo.
    pause
    exit /b 1
)

echo Activating Python environment...
call venv_age_classifier\Scripts\activate.bat

echo Starting Age Classifier Research Tool v2.0...
echo.
echo The web interface will open in your browser at:
echo ┌─────────────────────────────────────┐
echo │     http://127.0.0.1:7861          │
echo └─────────────────────────────────────┘
echo.
echo Loading models:
echo ├─ nateraw/vit-age-classifier (Vision Transformer - 9 age groups)
echo └─ prithivMLmods/Age-Classification-SigLIP2 (SigLIP - 5 age groups)
echo.
echo Features available:
echo ✓ Two-image comparison with detailed analysis
echo ✓ Batch folder processing with progress tracking
echo ✓ Statistical distribution analysis (entropy, confidence)
echo ✓ Model agreement metrics and ensemble predictions
echo ✓ Age boundary adjustment modes:
echo   • Strict: Original model outputs
echo   • Balanced: Improved teenage classification
echo.
echo ================================================================
echo Press Ctrl+C to stop the application
echo ================================================================
echo.

REM Start the application with error handling
python multi_model_age_classifier.py
if %errorlevel% neq 0 (
    echo.
    echo ERROR: Application failed to start!
    echo This might be due to:
    echo - Missing dependencies
    echo - Internet connection issues (for first-time model download)
    echo - Python environment problems
    echo.
    echo Try running install_age_classifier_batch.bat again.
    echo.
)

echo.
echo ================================================================
echo                   Application Stopped v2.0
echo ================================================================
pause
