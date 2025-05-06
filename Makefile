.PHONY: all install build test clean

# Default target
all: test

# Install all project dependencies and external components
install:
	@echo "Running full install..."
	@bash devops/setup_build.sh

# Build the C/C++ extension in-place
build:
	@echo "Building mettagrid extension..."
	cd mettagrid && python setup.py build_ext --inplace
	@echo "Building metta extensions..."
	python setup.py build_ext --inplace

# Run tests with coverage
test: build
	@echo "Running tests with coverage..."
	PYTHONPATH=deps pytest --cov=mettagrid --cov-report=term-missing

# Clean build artifacts
clean:
	@echo "Cleaning build artifacts..."
	find . -type f -name '*.so' -delete
	find . -type d -name '__pycache__' -exec rm -rf {} +
	find . -type d -name 'build' -exec rm -rf {} +
	find . -type d -name '*.egg-info' -exec rm -rf {} +
	find . -type f -name '.coverage' -delete
