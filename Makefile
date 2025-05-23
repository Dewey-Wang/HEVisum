.PHONY: help init test clean reset lab

help:
	@echo "Usage: make [target]"
	@echo "Targets:"
	@echo "  init   - Set up virtual environment and install dependencies"
	@echo "  lab    - Start JupyterLab"
	@echo "  clean  - Remove virtual environment and unregister Jupyter kernel"
	@echo "  reset  - Clean and reinitialize environment"

init:
	@echo "[INIT] Creating virtual environment and installing dependencies..."
	@bash install_host.sh
	@.venv/bin/python -m ipykernel install --user --name=spatialhackathon --display-name "Python (.venv) spatialhackathon" || true

clean:
	@echo "[CLEAN] Removing virtual environment and unregistering Jupyter kernel..."
	@rm -rf .venv || rmdir /S /Q .venv
	@jupyter kernelspec uninstall -f spatialhackathon || true

lab:
	@echo "[LAB] Starting JupyterLab..."
	@.venv/bin/jupyter lab

reset: clean init
