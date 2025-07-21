.PHONY: help install test clean docker

help:
	@echo "Available targets:"
	@echo "  install - Install package in editable mode"
	@echo "  test    - Run tests with coverage"
	@echo "  docker  - Build Docker image"

install:
	@echo "Installing package in editable mode..."
	UV_PROJECT_ENVIRONMENT=../.venv uv sync --inexact --active

test:
	@echo "Running tests with coverage..."
	UV_PROJECT_ENVIRONMENT=../.venv uv run --active pytest --cov=metta.app_backend --cov-report=term-missing

clean:
	@echo "Cleaning extra venv..."
	@rm -rf .venv
	@rm -f uv.lock

docker:
	@echo "Building Docker image..."
	docker build -t metta-app-backend:latest -f Dockerfile ..
