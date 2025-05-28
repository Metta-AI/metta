.PHONY: clean test build help install

# Default target
help:
	@echo "Available targets:"
	@echo "  build - Build in debug mode"
	@echo "  test  - Build and run all tests"
	@echo "  clean - Clean all build artifacts"

# Build in debug mode
build:
	@echo "Building in debug mode..."
	cmake --preset debug
	cmake --build build-debug

# Build and run all tests
test: build
	@echo "Running tests..."
	ctest --test-dir build-debug --rerun-failed --output-on-failure

# Clean all build artifacts
clean:
	@echo "Cleaning build artifacts..."
	rm -rf build-debug build-release

install:
	uv pip install -e .
