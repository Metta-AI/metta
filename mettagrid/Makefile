.PHONY: clean test build help install

# Default target
help:
	@echo "Available targets:"
	@echo "  build   - Build in debug mode"
	@echo "  test    - Build and run all tests"
	@echo "  clean   - Clean all build artifacts"
	@echo "  install - Install package in editable mode"

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

# Install package in editable mode
install:
	@echo "Installing package in editable mode..."
	uv pip install -e .