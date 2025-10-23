.PHONY: help build build-prod benchmark test coverage

# Default target
help:
	@echo "Available targets:"
	@echo "  build           - Debug build"
	@echo "  build-prod      - Release build (optimized)"
	@echo "  benchmark       - Build and run benchmarks"
	@echo "  test            - Run C++ unit tests"
	@echo "  coverage        - Run tests (C++ coverage not available on macOS)"

# Debug build
build:
	@echo "Building debug..."
	bazel build --config=dbg //:mettagrid_c

# Production release build
build-prod:
	@echo "Building release..."
	bazel build --config=opt //:mettagrid_c

# Build and run benchmarks
PYTEST_FLAGS ?=
ifneq ($(strip $(VERBOSE)),)
PYTEST_FLAGS += -s
endif

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
	uv run pytest benchmarks/test_mettagrid_env_benchmark.py -v --benchmark-only $(PYTEST_FLAGS)

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
