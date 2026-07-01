#!/usr/bin/env bash
set -e

echo "============================================================"
2: echo "             STARSIGHT LOCAL ENVIRONMENT SETUP"
3: echo "============================================================"
4: 
5: # 1. Detect Python Version
6: if command -v python3 &>/dev/null; then
7:     PYTHON_BIN=python3
8: elif command -v python &>/dev/null; then
9:     PYTHON_BIN=python
10: else
11:     echo "Error: Python is not installed. Please install Python 3.12+ and retry."
12:     exit 1
13: fi
14: 
15: echo "Using Python executable: $PYTHON_BIN"
16: 
17: # 2. Create virtual environment
18: if [ ! -d ".venv" ]; then
19:     echo "Creating virtual environment in .venv..."
20:     $PYTHON_BIN -m venv .venv
21: else
22:     echo "Virtual environment .venv already exists."
23: fi
24: 
25: # 3. Activate and Install dependencies
26: echo "Upgrading pip and installing requirements..."
27: .venv/bin/pip install --upgrade pip
28: .venv/bin/pip install -r requirements.txt
29: 
30: echo "Installing dev requirements..."
31: .venv/bin/pip install -r requirements-dev.txt
32: 
33: echo "============================================================"
34: echo "Setup completed successfully!"
35: echo "To activate the environment: source .venv/bin/activate"
36: echo "To run the dashboard: streamlit run app/app.py"
37: echo "============================================================"
