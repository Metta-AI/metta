
.PHONY: help install test

help:
	@echo "Available targets:"
	@echo "  install - Install package in editable mode"
	@echo "  test    - Run tests with coverage"

install:
	@echo "Installing package in editable mode..."
	uv sync --inexact

test:
	@echo "Running tests with coverage..."
	uv run pytest --cov=metta.common --cov-report=term-missing

