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
	rm -rf build dist htmlcov .pytest_cache .coverage ./*.egg-info || true

	$(MAKE) -C packages/mettagrid clean

	@PRUNES='-name .git -o -name .venv -o -name node_modules -o -name .bazel_output -o -name "bazel-*" -o -name bazel-out -o -name bazel-bin -o -name bazel-testlogs'; \
	find . \( $$PRUNES \) -prune -o \
	  \( -type d -name __pycache__ -print0 -o -type f -name '*.pyc' -print0 \) | xargs -0r rm -rf


# Full development setup
dev:
	@bash devops/setup_dev.sh

# Run everything
all: dev test
