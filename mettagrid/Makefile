.PHONY: help build build-prod benchmark test coverage lint tidy clean install pytest

# Default target
help:
	@echo "Available targets:"
	@echo "  build          - Debug build with tests and coverage"
	@echo "  build-prod     - Release build (no tests)"
	@echo "  benchmark      - Build and run benchmarks"
	@echo "  test           - Run unit tests"
	@echo "  coverage       - Generate coverage report"
	@echo "  lint           - Run cpplint"
	@echo "  tidy           - Run clang-tidy"
	@echo "  pytest         - Run Python tests"
	@echo "  clean          - Clean all build artifacts"
	@echo "  install        - Install package in editable mode"

# Debug build with tests and coverage
build:
	@echo "🔨 Building debug with tests and coverage..."
	cmake --preset debug
	cmake --build build-debug

# Production release build
build-prod:
	@echo "🔨 Building release..."
	cmake --preset release
	cmake --build build-release

# Build and run benchmarks
benchmark:
	@echo "🏃 Building and running benchmarks..."
	cmake --preset benchmark
	cmake --build build-release
	cd build-release && ctest -L benchmark --output-on-failure

# Run unit tests
test: build
	@echo "🧪 Running unit tests..."
	cd build-debug && ctest -L test --output-on-failure

# Generate coverage report
coverage: test
	@echo "📊 Generating coverage report..."
	@./generate_coverage.py

# Code quality
lint:
	@echo "🔍 Running cpplint..."
	@bash ./tests/cpplint.sh

tidy: build
	@echo "🔍 Running clang-tidy..."
	@bash clang-tidy.sh

# Python
install:
	@echo "📦 Installing package..."
	UV_PROJECT_ENVIRONMENT=../.venv uv sync --inexact --active

pytest: install
	@echo "🐍 Running Python tests..."
	UV_PROJECT_ENVIRONMENT=../.venv uv run --active pytest

# Cleanup
clean:
	@echo "🧹 Cleaning build artifacts..."
	rm -rf build-debug build-release .venv uv.lock
	rm -f coverage.info *.gcda *.gcno *.profraw *.profdata
