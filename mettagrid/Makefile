.PHONY: clean test build help install
# Default target
help:
	@echo "Available targets:"
	@echo "  build   - Build project with tests"
	@echo "  test    - Build and run all tests (Python + C++)"
	@echo "  clean   - Clean all build artifacts"
	@echo "  install - Install package in editable mode"

# Build project with tests
build:
	@echo "Building project with tests..."
	cmake -B build -DBUILD_TESTS=ON
	cmake --build build

# Build and run all tests
test: build
	@echo "Running C++ tests..."
	ctest --test-dir build --rerun-failed --output-on-failure

# Clean all build artifacts
clean:
	@echo "Cleaning build artifacts..."
	rm -rf build

# Install package in editable mode
install:
	@echo "Installing package in editable mode..."
	uv pip install -e .
