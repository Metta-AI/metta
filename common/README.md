# metta-common

Common utilities shared across metta packages.

## Installation

```bash
# Install using uv
uv pip install -e .

# Install with dev dependencies
uv pip install -e ".[dev]"
```

## Development

```bash
# Run tests
uv run pytest

# Run tests with coverage
uv run pytest --cov=metta.common --cov-report=term-missing

# Run linting
uv run ruff check .

# Run type checking
uv run pyright
```
