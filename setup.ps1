Write-Host "============================================================"
Write-Host "             STARSIGHT WINDOWS POWERSHELL SETUP"
Write-Host "============================================================"

$python = Get-Command python.exe -ErrorAction SilentlyContinue
if (-not $python) {
    Write-Error "Python is not installed or not in PATH. Please install Python 3.12+ and retry."
    exit 1
}

if (-not (Test-Path ".venv")) {
    Write-Host "Creating virtual environment in .venv..."
    python -m venv .venv
} else {
    Write-Host "Virtual environment .venv already exists."
}

Write-Host "Upgrading pip and installing requirements..."
& .venv\Scripts\python.exe -m pip install --upgrade pip
& .venv\Scripts\pip.exe install -r requirements.txt
& .venv\Scripts\pip.exe install -r requirements-dev.txt

Write-Host "============================================================"
Write-Host "Setup completed successfully!"
Write-Host "To activate the environment: .\.venv\Scripts\Activate.ps1"
Write-Host "To run the dashboard: streamlit run app/app.py"
Write-Host "============================================================"
