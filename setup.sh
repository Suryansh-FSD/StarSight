#!/usr/bin/env bash
set -e

echo "============================================================"
echo "             STARSIGHT LOCAL ENVIRONMENT SETUP"
echo "============================================================"

# 1. Detect Python Version
if command -v python3 &>/dev/null; then
    PYTHON_BIN=python3
elif command -v python &>/dev/null; then
    PYTHON_BIN=python
else
    echo "Error: Python is not installed. Please install Python 3.12+ and retry."
    exit 1
fi

echo "Using Python executable: $PYTHON_BIN"

# 2. Create virtual environment
if [ ! -d ".venv" ]; then
    echo "Creating virtual environment in .venv..."
    $PYTHON_BIN -m venv .venv
else
    echo "Virtual environment .venv already exists."
fi

# 3. Activate and Install dependencies
echo "Upgrading pip and installing requirements..."
.venv/bin/pip install --upgrade pip
.venv/bin/pip install -r requirements.txt

echo "Installing dev requirements..."
.venv/bin/pip install -r requirements-dev.txt

echo "============================================================"
echo "Setup completed successfully!"
echo "To activate the environment: source .venv/bin/activate"
echo "To run the dashboard: streamlit run app/app.py"
echo "============================================================"
