.PHONY: help build build-prod benchmark test coverage tidy tidy-verbose clean install pytest

# Default target
help:
	@echo "Available targets:"
	@echo "  build          - Debug build with tests and coverage"
	@echo "  build-prod     - Release build (no tests)"
	@echo "  benchmark      - Build and run benchmarks"
	@echo "  test           - Run unit tests"
	@echo "  coverage       - Generate coverage report"
	@echo "  tidy           - Run clang-tidy (fails on errors only)"
	@echo "  tidy-verbose   - Run clang-tidy with full output"
	@echo "  pytest         - Run Python tests"
	@echo "  clean          - Clean all build artifacts"
	@echo "  install        - Install package in editable mode"

# Debug build with tests and coverage
build:
	@echo "🔨 Building debug with tests and coverage..."
	bazel build --config=dbg //:mettagrid_c

# Production release build
build-prod:
	@echo "🔨 Building release..."
	bazel build --config=opt //:mettagrid_c

# Build and run benchmarks
benchmark:
	@echo "🔨 Building C++ benchmarks..."
	bazel build --config=opt //benchmarks:test_mettagrid_env_benchmark
	@echo "📂 Creating build-release directory..."
	mkdir -p build-release
	@echo "📋 Copying benchmark binaries..."
	cp -f bazel-bin/benchmarks/test_mettagrid_env_benchmark build-release/
	chmod +x build-release/test_mettagrid_env_benchmark
	@echo "🏃 Running C++ benchmarks..."
	bazel run //benchmarks:test_mettagrid_env_benchmark
	@echo "🏃 Running Python benchmarks..."
	uv run pytest benchmarks/test_mettagrid_env_benchmark.py -v --benchmark-only

# Run unit tests
test: build
	@echo "🧪 Running unit tests..."
	bazel test //tests:tests_all

# Generate coverage report
coverage:
	@echo "🔨 Building with coverage..."
	bazel coverage --combined_report=lcov //tests:tests_all
	@echo "📊 Processing coverage data..."
	@UV_PROJECT_ENVIRONMENT=../.venv uv run generate_coverage.py

# Alias for clang_tidy that only fails on errors (not warnings)
tidy:
	@echo "🔍 Running clang-tidy (errors only)..."
	@bazel test //lint:clang_tidy --test_output=errors  --nocache_test_results
	@echo "✅ Clang-tidy check complete (use 'make tidy-verbose' to see all warnings)"

# Run clang-tidy with full output showing all warnings
tidy-verbose:
	@echo "🔍 Running clang-tidy with full output..."
	@bazel test //lint:clang_tidy --test_output=all --nocache_test_results

# Python
install:
	@echo "📦 Installing package..."
	UV_PROJECT_ENVIRONMENT=../.venv uv sync --active --inexact --frozen

pytest: install
	@echo "🐍 Running Python tests..."
	UV_PROJECT_ENVIRONMENT=../.venv uv run --active pytest

# Cleanup
clean:
	@echo "🧹 Cleaning build artifacts..."
	bazel clean
	rm -rf .venv uv.lock
	rm -f coverage.info *.gcda *.gcno *.profraw *.profdata