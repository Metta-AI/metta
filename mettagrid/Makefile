.PHONY: help build lint test clean install pytest

# Default target
help:
	@echo "Available targets:"
	@echo "  build   - Build project with tests"
	@echo "  lint    - Run cpplint on all C++ source files"
	@echo "  test    - Build and run all C++ tests"
	@echo "  pytest  - Install package and run all Python tests"
	@echo "  clean   - Clean all build artifacts"
	@echo "  install - Install package in editable mode"

# Build project with tests
build:
	@echo "Building in debug mode..."
	cmake --preset debug
	cmake --build build-debug

# Run cpplint on all C++ source files
lint:
	@echo "Running cpplint..."
	@bash ./tests/cpplint.sh

# Build and run all tests
test: build lint
	@echo "Running C++ tests..."
	ctest --test-dir build-debug -L benchmark -V --rerun-failed --output-on-failure

# Clean all build artifacts
clean:
	@echo "Cleaning build artifacts..."
	rm -rf build-debug build-release build

# Install package in editable mode
install:
	@echo "Installing package in editable mode..."
	UV_PROJECT_ENVIRONMENT=../.venv uv sync --inexact

pytest: install
	@echo "Running Python tests..."
	UV_PROJECT_ENVIRONMENT=../.venv uv run pytest
