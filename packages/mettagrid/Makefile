.PHONY: help build build-prod benchmark test coverage tidy tidy-verbose clean install pytest pytest-coverage

# Default target
help:
	@echo "Available targets:"
	@echo "  build           - Debug build"
	@echo "  build-prod      - Release build (optimized)"
	@echo "  benchmark       - Build and run benchmarks"
	@echo "  test            - Run C++ unit tests"
	@echo "  coverage        - Run tests (C++ coverage not available on macOS)"
	@echo "  tidy            - Run clang-tidy (fails on errors only)"
	@echo "  tidy-verbose    - Run clang-tidy with full output"
	@echo "  format-check   - Verify code is clang-formatted"
	@echo "  format-fix     - Auto-format code with clang-format"
	@echo "  pytest          - Run Python tests"
	@echo "  pytest-coverage - Run Python tests with coverage"
	@echo "  clean           - Clean all build artifacts"
	@echo "  install         - Install package in editable mode"

# Debug build
build:
	@echo "Building debug..."
	bazel build --config=dbg //:mettagrid_c

# Production release build
build-prod:
	@echo "Building release..."
	bazel build --config=opt //:mettagrid_c

# Build and run benchmarks
benchmark:
	@echo "Building C++ benchmarks..."
	bazel build --config=opt //benchmarks:test_mettagrid_env_benchmark
	@echo "Creating build-release directory..."
	mkdir -p build-release
	@echo "Copying benchmark binaries..."
	cp -f bazel-bin/benchmarks/test_mettagrid_env_benchmark build-release/
	chmod +x build-release/test_mettagrid_env_benchmark
	@echo "Running C++ benchmarks..."
	# Note: Using the copied binary instead of 'bazel run' to avoid Python environment issues
	./build-release/test_mettagrid_env_benchmark || echo "C++ benchmark failed - this may be due to Python environment issues"
	@echo "Running Python benchmarks..."
	uv run pytest benchmarks/test_mettagrid_env_benchmark.py -v --benchmark-only

# Run unit tests
test: build
	@echo "Running unit tests..."
	bazel test --config=dbg //tests:tests_all

# Generate coverage report
coverage:
	@echo "Building with coverage..."
	@echo "Note: C++ coverage on macOS is currently not supported by bazel"
	@echo "    Running tests instead..."
	bazel test --config=dbg //tests:tests_all
	@echo "Tests completed successfully"
	@echo "    For Python coverage, use: make pytest-coverage"

# Alias for clang_tidy that only fails on errors (not warnings)
tidy:
	@echo "Running clang-tidy (errors only)..."
	@bazel test //lint:clang_tidy --test_output=errors  --nocache_test_results
	@echo "Clang-tidy check complete (use 'make tidy-verbose' to see all warnings)"

# Run clang-tidy with full output showing all warnings
tidy-verbose:
	@echo "Running clang-tidy with full output..."
	@bazel test //lint:clang_tidy --test_output=all --nocache_test_results

# Formatting
format-check:
	@echo "Checking code formatting..."
	@clang-format --dry-run --Werror -style=file $(shell find src include tests -name '*.c' -o -name '*.h' -o -name '*.cpp' -o -name '*.hpp')
	@echo "All files are correctly formatted"

format-fix:
	@echo "Reformatting code..."
	@clang-format -i -style=file $(shell find src include tests -name '*.c' -o -name '*.h' -o -name '*.cpp' -o -name '*.hpp')
	@echo "Code reformatted"

# Python
install:
	@echo "Installing package..."
	@cd ../.. && uv sync --active --inexact --frozen

pytest:
	@echo "Preparing environment for Python tests (ASAN disabled)..."
	@cd ../.. && DEBUG= uv sync --active --inexact --frozen
	@echo "Building C++ extension without ASAN (opt config)..."
	@cd ../.. && PYTHONPATH=packages/mettagrid DEBUG= uv run python -c "import bazel_build as bb; bb._run_bazel_build()"
	@echo "Running Python tests (ASAN disabled)..."
	@cd ../.. && DEBUG= uv run pytest packages/mettagrid/tests

pytest-coverage:
	@echo "Preparing environment for Python tests (ASAN disabled)..."
	@cd ../.. && DEBUG= uv sync --active --inexact --frozen
	@echo "Building C++ extension without ASAN (opt config)..."
	@cd ../.. && PYTHONPATH=packages/mettagrid DEBUG= uv run python -c "import bazel_build as bb; bb._run_bazel_build()"
	@echo "Running Python tests with coverage (ASAN disabled)..."
	@cd ../.. && DEBUG= uv run pytest packages/mettagrid/tests --cov=mettagrid --cov-report=term-missing --cov-report=html:packages/mettagrid/htmlcov

# Cleanup
clean:
	@echo "Cleaning build artifacts..."
	bazel clean
	rm -rf .venv uv.lock
	rm -f coverage.info *.gcda *.gcno *.profraw *.profdata
