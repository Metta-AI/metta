# Adaptive System Smoke Tests

These tests are **standalone** and **not part of the regular test suite**. They make real API calls and require proper credentials/setup.

## Purpose

- **Smoke Tests**: End-to-end tests with real external services (WandB, Skypilot)
- **Integration Tests**: Full system integration with appropriately mocked components
- **Manual Verification**: Tests that require manual inspection or long-running validation

## Running Tests

```bash
# Set up environment variables first
export WANDB_API_KEY=your_key_here
export WANDB_ENTITY=your_entity
export WANDB_PROJECT=adaptive_smoke_test

# Run individual smoke tests
uv run smoke_tests/adaptive/test_wandb_smoke.py
uv run smoke_tests/adaptive/test_full_workflow_smoke.py

# Run integration tests (with mocks)
uv run smoke_tests/adaptive/test_integration_mocked.py
```

## Test Categories

1. **`test_wandb_smoke.py`** - Real WandB API integration
2. **`test_skypilot_smoke.py`** - Real Skypilot dispatcher integration
3. **`test_full_workflow_smoke.py`** - Complete adaptive workflow end-to-end
4. **`test_integration_mocked.py`** - Full integration with properly mocked external services

## Requirements

- Valid WandB account and API key
- Skypilot setup for cloud tests
- Sufficient cloud credits for dispatcher tests