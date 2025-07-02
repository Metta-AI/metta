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

test: build lint
	@echo "Running C++ tests..."
	ctest --test-dir build-debug -L benchmark -V --rerun-failed --output-on-failure

clean:
	@echo "Cleaning extra venv..."
	rm -rf .venv
	rm uv.lock
	@echo "Cleaning build artifacts..."
	rm -rf build-debug build-release build

install:
	@echo "Installing package in editable mode..."
	UV_PROJECT_ENVIRONMENT=../.venv uv sync --inexact

pytest: install
	@echo "Running Python tests..."
	UV_PROJECT_ENVIRONMENT=../.venv uv run pytest
