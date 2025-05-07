.PHONY: all install build test clean clean-mettagrid clean-root

# Default target
all: test

# Install all project dependencies and external components
install:
	@echo "Running full install..."
	@bash devops/setup_build.sh

# Run tests with coverage
test: build
	@echo "Running tests with coverage..."
	PYTHONPATH=deps pytest --cov=mettagrid --cov-report=term-missing

# Clean build artifacts in root directory
clean-root:
	@echo "Cleaning root build artifacts..."
	find . -type f -name '*.so' -delete
	find . -type d -name '__pycache__' -exec rm -rf {} +
	find . -type d -name 'build' -exec rm -rf {} +
	find . -type d -name '*.egg-info' -exec rm -rf {} +
	find . -type f -name '.coverage' -delete

# Clean build artifacts in mettagrid subdirectory
clean-mettagrid:
	@echo "Cleaning mettagrid build artifacts..."
	cd mettagrid && $(MAKE) clean

# Clean all build artifacts
clean: clean-root clean-mettagrid

# Rebuild
build: clean-mettagrid
	@echo "Building mettagrid extension..."
	cd mettagrid && $(MAKE) build
	@echo "Building metta extensions..."
	$(MAKE) build-metta

# Build metta extensions (new target)
build-metta:
	@echo "Building metta extensions..."
	# Add your metta build commands here