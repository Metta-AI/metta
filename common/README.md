# metta-common

Common utilities shared across metta packages.

## Installation

```bash
# Install using uv
uv sync --inexact
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
