#!/bin/bash

# ✅ Python 版本檢查 (>= 3.9)
PYTHON_VERSION=$(python3 -c 'import sys; print(".".join(map(str, sys.version_info[:2])))')
REQUIRED_VERSION="3.9"

if [[ $(echo -e "$PYTHON_VERSION\n$REQUIRED_VERSION" | sort -V | head -n1) != "$REQUIRED_VERSION" ]]; then
  echo "❌ Python 3.9 or higher is required. Current version: $PYTHON_VERSION"
  exit 1
fi

python3 -m venv venv
source venv/bin/activate  # Windows pls change to venv\Scripts\activate

pip install --upgrade pip
pip install -r requirements.txt
