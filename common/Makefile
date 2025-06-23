## Add `common/Makefile`:

```makefile
.PHONY: help install test lint clean

help:
	@echo "Available targets:"
	@echo "  install - Install package in editable mode"
	@echo "  test    - Run tests with coverage"
	@echo "  lint    - Run linting and type checking"
	@echo "  clean   - Remove build artifacts"

install:
	@echo "Installing metta-common in editable mode..."
	uv pip install -e ".[dev]"

test:
	@echo "Running tests with coverage..."
	uv run pytest --cov=metta.common --cov-report=term-missing

lint:
	@echo "Running ruff..."
	uv run ruff check src tests
	@echo "Running pyright..."
	uv run pyright

clean:
	@echo "Cleaning build artifacts..."
	rm -rf build dist *.egg-info
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	rm -rf .coverage htmlcov .pytest_cache .ruff_cache
