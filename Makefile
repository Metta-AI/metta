.PHONY: help all dev test clean install


# Default target when just running 'make'
help:
	@echo "Available targets:"
	@echo " dev - Prepare the dev environment"
	@echo " test - Run all unit tests"
	@echo " all - Run dev and test"
	@echo " clean - Remove cmake build artifacts and temporary files"


# Clean cmake build artifacts
clean:
	@echo "(Metta) Cleaning root build artifacts..."
	rm -rf build
	@echo "(Metta) Cleaning mettagrid build artifacts..."
	cd mettagrid && rm -rf build-debug build-release
	@echo "(Metta) Clean completed successfully"

# Dev all project dependencies and external components
dev:
	@echo "Running full devops/setup_dev.sh installation script..."
	@bash devops/setup_dev.sh

test:
	@echo "Running python tests with coverage"
	uv run pytest --cov=metta --cov-report=term-missing

install:
	@echo "Installing package in editable mode..."
	uv sync --inexact

pytest: install
	@echo "Running Python tests..."
	uv run pytest

all: dev test
