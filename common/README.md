# metta-common

Common utilities shared across metta packages.

## Installation

```bash
# Install using uv helper (detects CUDA backend automatically)
./scripts/uv-sync.sh --inexact
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
