.PHONY: help install test clean

help:
	@echo "Available targets:"
	@echo "  install - Install package in editable mode"
	@echo "  test    - Run tests with coverage"
	@echo "  clean   - Clean build artifacts"

install:
	@echo "Installing package in editable mode..."
	UV_PROJECT_ENVIRONMENT=../.venv uv sync --inexact

test:
	@echo "Running tests with coverage..."
	UV_PROJECT_ENVIRONMENT=../.venv uv run pytest --cov=metta.sim --cov-report=term-missing

clean:
	@echo "Cleaning build artifacts..."
	@rm -rf .venv
	@rm -f uv.lock
	@find . -type d -name __pycache__ -exec rm -rf {} +
	@find . -type f -name "*.pyc" -delete
