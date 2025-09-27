.PHONY: help install test pytest bench clean dev all lint format

# Default target - show help
help:
	@echo "Usage: make [target]"
	@echo ""
	@echo "Targets:"
	@echo "  install     Install package and dependencies"
	@echo "  reinstall   Re-install package and dependencies"
	@echo "  test        Run tests with coverage"
	@echo "  pytest      Run tests quickly (no coverage, parallel)"
	@echo "  lint        Run ruff check (without modifying files)"
	@echo "  format      Format all file types (Python, JSON, MD, etc.)"
	@echo "  clean       Remove build artifacts"
	@echo "  dev         Set up development environment"
	@echo "  all         Run dev and test"

reinstall:
	uv sync --reinstall

install:
	uv sync

# Run tests with coverage and benchmarks
test: install
	uv run pytest \
		tests \
		mettascope/tests \
		agent/tests \
		app_backend/tests \
		common/tests \
		packages/mettagrid/tests \
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
		packages/mettagrid/tests \
		--benchmark-disable -n auto

# Run linting checks only (no modifications)
lint: install
	@echo "Running Python lint checks..."
	uv run ruff check .
	uv run ruff format --check .
	@echo "Running mettagrid lint..."
	cd packages/mettagrid && make lint

# Format all file types
format: install
	@echo "Formatting Python files..."
	uv run ruff format .
	@echo "Formatting JSON files..."
	@bash devops/tools/format_json.sh
	@echo "Formatting Markdown files..."
	@bash devops/tools/format_md.sh
	@echo "Formatting Shell scripts..."
	@bash devops/tools/format_sh.sh
	@echo "Formatting TOML files..."
	@bash devops/tools/format_toml.sh
	@echo "Formatting YAML files..."
	@bash devops/tools/format_yml.sh
	@echo "Running mettagrid format..."
	cd packages/mettagrid && make format-fix

clean:
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info
	rm -rf .pytest_cache/
	rm -rf .coverage
	rm -rf htmlcov/
	cd packages/mettagrid && make clean

	# Safely remove Python caches, excluding all .bazel_output/* and similar
	find . -type d -name __pycache__ \
		! -path "*/.bazel_output/*" \
		! -path "*/bazel-out/*" \
		! -path "*/bazel-bin/*" \
		-exec rm -rf {} +

	find . -type f -name "*.pyc" \
		! -path "*/.bazel_output/*" \
		! -path "*/bazel-out/*" \
		! -path "*/bazel-bin/*" \
		-delete

# Full development setup
dev:
	@bash devops/setup_dev.sh

# Run everything
all: dev test
