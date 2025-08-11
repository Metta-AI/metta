.PHONY: help install test pytest bench clean dev all

# Default target - show help
help:
	@echo "Usage: make [target]"
	@echo ""
	@echo "Targets:"
	@echo "  install     Install package and dependencies"
	@echo "  reinstall   Re-install package and dependencies"
	@echo "  test-setup  Run setup integration tests"
	@echo "  test        Run tests with coverage"
	@echo "  pytest      Run tests quickly (no coverage, parallel)"
	@echo "  clean       Remove build artifacts"
	@echo "  dev         Set up development environment"
	@echo "  all         Run dev and test"

reinstall:
	uv sync --reinstall

install:
	uv sync

# Run metta setup tests
test-setup:
	@echo "Running setup integration tests..."
	METTA_TEST_ENV=1 \
	METTA_TEST_SETUP=1 \
	METTA_TEST_PROFILE=$(METTA_TEST_PROFILE) \
	AWS_SSO_NONINTERACTIVE=1 \
		uv run metta test tests/setup -v -n auto

# Run tests with coverage and benchmarks
test: install
	uv run pytest \
		tests \
		mettascope/tests \
		agent/tests \
		app_backend/tests \
		common/tests \
		mettagrid/tests \
		--cov=metta \
		--cov-report=term-missing \
		--durations=10 \
		-n 0

# Quick test run without coverage
pytest: install
	uv run pytest \
		tests \
		mettascope/tests \
		agent/tests \
		app_backend/tests \
		common/tests \
		mettagrid/tests \
		--benchmark-disable -n auto

# Clean build artifacts
clean:
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info
	rm -rf .pytest_cache/
	rm -rf .coverage
	rm -rf htmlcov/
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete

# Full development setup
dev:
	@bash devops/setup_dev.sh

# Run everything
all: dev test
