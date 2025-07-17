.PHONY: help build lint test clean install pytest tidy coverage

# Default target
help:
	@echo "Available targets:"
	@echo "  build          - Build project for tests"
	@echo "  build-prod     - Build project for production"
	@echo "  lint           - Run cpplint on all C++ source files"
	@echo "  tidy           - Run clang-tidy static analysis"
	@echo "  test           - Build and run all C++ tests"
	@echo "  pytest         - Install package and run all Python tests"
	@echo "  coverage       - Generate C++ coverage report"
	@echo "  clean          - Clean all build artifacts"
	@echo "  install        - Install package in editable mode"

# Build project with tests
build:
	@echo "Building in debug mode..."
	cmake --preset debug
	cmake --build build-debug

build-prod:
	@echo "Building in release mode..."
	cmake --preset release-no-tests
	cmake --build build-release

# Run cpplint on all C++ source files
lint:
	@echo "Running cpplint..."
	@bash ./tests/cpplint.sh

test: build lint
	@echo "Running C++ tests..."
	ctest --test-dir build-debug -V --output-on-failure

coverage:
	@echo "Generating C++ coverage..."
	@bash ./generate-coverage.sh

clean:
	@echo "Cleaning extra venv..."
	@rm -rf .venv
	@rm -f uv.lock
	@echo "Cleaning build artifacts..."
	rm -rf build-debug build-release build build-coverage
	@echo "Cleaning coverage files..."
	rm -f coverage.info *.gcda *.gcno *.profraw *.profdata

install:
	@echo "Installing package in editable mode..."
	UV_PROJECT_ENVIRONMENT=../.venv uv sync --inexact --active

pytest: install
	@echo "Running Python tests..."
	UV_PROJECT_ENVIRONMENT=../.venv uv run --active pytest

tidy: build
	@echo "Running clang-tidy..."
	@bash clang-tidy.sh
