#!/usr/bin/env bash

set -e  # exit on error

# Detect platform
OS="$(uname -s)"
case "${OS}" in
    Linux*)     machine=Linux;;
    Darwin*)    machine=Mac;;
    CYGWIN*|MINGW*|MSYS*)    machine=Windows;;
    *)          machine="UNKNOWN:${OS}"
esac

# Python version check
PYTHON_BIN="python3"
REQUIRED_VERSION="3.9"
CURRENT_VERSION=$($PYTHON_BIN -c 'import sys; print("{}.{}".format(*sys.version_info[:2]))')

if [[ $(echo -e "$CURRENT_VERSION\n$REQUIRED_VERSION" | sort -V | head -n1) != "$REQUIRED_VERSION" ]]; then
  echo "❌ Python >= 3.9 required. Current: $CURRENT_VERSION"
  exit 1
fi

# Setup virtual environment
VENV_DIR=".venv"

if [ ! -d "$VENV_DIR" ]; then
  $PYTHON_BIN -m venv $VENV_DIR
fi

# Activate virtual environment
if [[ "$machine" == "Windows" ]]; then
    source "$VENV_DIR/Scripts/activate"
else
    source "$VENV_DIR/bin/activate"
fi

# Upgrade pip safely
python -m pip install --upgrade pip

# Install requirements
python -m pip install --upgrade --no-deps -r requirements.txt

# Add test support
mkdir -p tests
cat <<EOF > tests/test_basic.py
import pytest

def test_example():
    assert 1 + 1 == 2
EOF

# Confirm
echo "✅ Environment setup complete. To activate:"
echo "   source .venv/bin/activate"
