#!/bin/bash
set -e  # exit immediately on error

echo "[INFO] Setting up environment for Graph Indexing (Q3)"

# Create virtual environment (local, safe)
if [ ! -d "venv" ]; then
    echo "[INFO] Creating virtual environment"
    python3 -m venv venv
fi

# Activate venv
source venv/bin/activate

# Upgrade pip
echo "[INFO] Upgrading pip"
pip install --upgrade pip

# Install required Python libraries
echo "[INFO] Installing Python dependencies"
pip install numpy vf3py

echo "[INFO] Environment setup complete"