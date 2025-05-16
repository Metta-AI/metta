.PHONY: help all install build test clean test-python check-venv


# Default target when just running 'make'
help:
	@echo "Available targets:"
	@echo " install - Build mettagrid using the rebuild script"
	@echo " test - Run all unit tests"
	@echo " build - Build from setup.py"
	@echo " all - Run install and test"
	@echo " clean - Remove build artifacts and temporary files"


# Check if we're in a virtual environment
check-venv:
	@if [ -z "$$VIRTUAL_ENV" ]; then \
		echo "ERROR: Not in a virtual environment!"; \
		echo "Please activate the virtual environment first with:"; \
		echo "  source .venv/bin/activate"; \
		exit 1; \
	else \
		echo "... in active virtual environment: $$VIRTUAL_ENV"; \
	fi

clean:
	@echo "(Metta) Cleaning root build artifacts..."
	@echo "(Metta) Removing all '.so' files (excluding .venv)"
	@find . -type f -name '*.so' -not -path "./.venv/*" -delete || true
	@echo "(Metta) Removing build directories (excluding .venv)"
	@find . -type d -name 'build' -not -path "./.venv/*" -print0 | xargs -0 rm -rf 2>/dev/null || true
	@echo "(Metta) Cleaning mettagrid build artifacts..."
	@if [ -d "mettagrid" ]; then \
		cd mettagrid && $(MAKE) clean || true; \
	else \
		echo "(Metta) mettagrid directory not found, skipping"; \
	fi
	@echo "(Metta) Clean completed successfully"

# Install all project dependencies and external components
install:
	@echo "Running full devops/setup_build installation script..."
	@bash devops/setup_build.sh

test-python: check-venv
	@echo "Running python tests with coverage"
	pytest --cov=metta --cov-report=term-missing

test: test-python

all: clean install check-venv test

# Build the project using setup.py
build: check-venv
	@echo "Building metta..."
	python setup.py build_ext --inplace
	@echo "Build complete."