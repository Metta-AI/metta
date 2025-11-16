# metta-agent

Agent/Policy utilities for metta packages.

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
uv run pytest --cov=metta.agent --cov-report=term-missing

# Run linting/formatting from the repo root
metta lint packages/agent

# Apply auto-fixes
metta lint --fix packages/agent
```
