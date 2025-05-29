.PHONY: help all install test clean check-venv


# Default target when just running 'make'
help:
	@echo "Available targets:"
	@echo " install - Prepare the dev environment"
	@echo " test - Run all unit tests"
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
	@echo "(Metta) Removing build directories (excluding .venv)"
	rm -rf build
	@echo "(Metta) Cleaning mettagrid build artifacts..."
	cd mettagrid && rm -rf build-debug build-release
	@echo "(Metta) Cleaning uv build cache..."
	rm -rf ~/.cache/uv/builds-v0
	@echo "(Metta) Clean completed successfully"

# Install all project dependencies and external components
install:
	@echo "Running full devops/setup_build installation script..."
	@bash devops/setup_build.sh

test: check-venv
	@echo "Running python tests with coverage"
	pytest --cov=metta --cov-report=term-missing

all: clean install check-venv test
