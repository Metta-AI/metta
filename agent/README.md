# metta-agent

Agent/Policy utilities for metta packages.

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
uv run pytest --cov=metta.agent --cov-report=term-missing

# Run linting/formatting from the repo root
metta lint packages/agent

# Apply auto-fixes
metta lint --fix packages/agent
```
