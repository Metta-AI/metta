.PHONY: help build build-prod benchmark test coverage format-check format-fix

# Default target
help:
	@echo "Available targets:"
	@echo "  build           - Debug build"
	@echo "  build-prod      - Release build (optimized)"
	@echo "  benchmark       - Build and run benchmarks"
	@echo "  test            - Run C++ unit tests"
	@echo "  coverage        - Run tests (C++ coverage not available on macOS)"
	@echo "  format-check    - Verify code is clang-formatted"
	@echo "  format-fix      - Auto-format code with clang-format"

# Debug build
build:
	@echo "Building debug..."
	bazel build --config=debug //:mettagrid_c

# Production release build
build-prod:
	@echo "Building release..."
	bazel build --config=release //:mettagrid_c

# Build and run benchmarks
PYTEST_FLAGS ?=
ifneq ($(strip $(VERBOSE)),)
PYTEST_FLAGS += -s
endif

# Discover uv-managed Python runtime locations for embedded interpreter use
UV_PY_PREFIX := $(shell uv run -q python -c 'import sys; print(sys.base_prefix)')
UV_PY_LIBDIR := $(shell uv run -q python -c 'import sysconfig;print(sysconfig.get_config_var("LIBDIR") or sysconfig.get_config_var("LIBPL"))')
UV_SITEPKG  := $(shell uv run -q python -c 'import site; print(site.getsitepackages()[0])')

benchmark:
	@echo "Building C++ benchmarks..."
	bazel build --config=release //benchmarks:test_mettagrid_env_benchmark
	@echo "Creating build-release directory..."
	mkdir -p build-release
	@echo "Copying benchmark binaries..."
	cp -f bazel-bin/benchmarks/test_mettagrid_env_benchmark build-release/
	chmod +x build-release/test_mettagrid_env_benchmark
	@echo "Running C++ benchmarks..."
	# Run binary with correct Python runtime env so libpython and stdlib/site-packages are discoverable
	LD_LIBRARY_PATH="$(UV_PY_LIBDIR):$${LD_LIBRARY_PATH}" \
	PYTHONHOME="$(UV_PY_PREFIX)" \
	PYTHONPATH="$(UV_SITEPKG):$(UV_PY_PREFIX)/lib/python3.12:$(UV_PY_PREFIX)/lib/python3.12/lib-dynload" \
	./build-release/test_mettagrid_env_benchmark --benchmark_time_unit=us
	@echo "Running Python benchmarks..."
	uv run pytest benchmarks/test_mettagrid_env_benchmark.py -v --benchmark-only $(PYTEST_FLAGS)

# Run unit tests
test: build
	@echo "Running unit tests..."
	bazel test --config=debug //tests:tests_all

# Formatting
format-check:
	@echo "Checking code formatting..."
	@clang-format --dry-run --Werror -style=file $(shell find cpp/src cpp/include tests benchmarks -name '*.c' -o -name '*.h' -o -name '*.cpp' -o -name '*.hpp' 2>/dev/null)
	@echo "All files are correctly formatted"

format-fix:
	@echo "Reformatting code..."
	@clang-format -i -style=file $(shell find cpp/src cpp/include tests benchmarks -name '*.c' -o -name '*.h' -o -name '*.cpp' -o -name '*.hpp' 2>/dev/null)
	@echo "Code reformatted"
