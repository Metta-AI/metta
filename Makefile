.PHONY: help all reinstall install build test clean 

# Default target when just running 'make'
help:
	@echo "Available targets:"
	@echo " install - Build mettagrid using the rebuild script"
	@echo " test    - Run all unit tests"
	@echo " all     - Run install and test"
	@echo " clean   - Remove build artifacts and temporary files"

# Clean build artifacts in root directory
clean:
	@echo "Cleaning root build artifacts..."
	find . -type f -name '*.so' -delete
	find . -type d -name 'build' -exec rm -rf {} +
	@echo "Cleaning mettagrid build artifacts..."
	cd mettagrid && $(MAKE) clean
	
# Install all project dependencies and external components
install:
	@echo "Running full install..."
	@bash devops/setup_build.sh

# Run tests with coverage
test:
	@echo "Running tests with coverage..."
	PYTHONPATH=deps pytest --cov=mettagrid --cov-report=term-missing

all: clean install test