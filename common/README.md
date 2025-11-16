# metta-common

Common utilities shared across metta packages.

## Installation

```bash
# Install and then auto-select torch backend
uv sync --inexact
UV_TORCH_BACKEND=auto uv pip install --python .venv/bin/python torch>=2.9.1
```

## Development

```bash
# Run tests
uv run pytest

# Run tests with coverage
uv run pytest --cov=metta.common --cov-report=term-missing

# Run linting/formatting from the repo root
metta lint packages/common

# Apply auto-fixes
metta lint --fix packages/common
```
