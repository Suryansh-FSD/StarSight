@echo off
echo ============================================================
echo              STARSIGHT WINDOWS BATCH SETUP
echo ============================================================

where python >nul 2>nul
if %errorlevel% neq 0 (
    echo Error: Python is not installed or not in PATH. Please install Python 3.12+ and retry.
    exit /b 1
)

if not exist .venv (
    echo Creating virtual environment in .venv...
    python -m venv .venv
) else (
    echo Virtual environment .venv already exists.
)

echo Upgrading pip and installing requirements...
.venv\Scripts\python.exe -m pip install --upgrade pip
.venv\Scripts\pip.exe install -r requirements.txt
.venv\Scripts\pip.exe install -r requirements-dev.txt

echo ============================================================
echo Setup completed successfully!
echo To activate the environment: .venv\Scripts\activate.bat
echo To run the dashboard: streamlit run app/app.py
echo ============================================================
