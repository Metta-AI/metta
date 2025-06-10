# Metta Testing Guide

This document outlines our standard practices for testing. The goal is to build a robust test suite that prevents regressions and ensures new features are working correctly.

## Guiding Principle: Debugging into Tests

Every debugging session should result in a new test case. If you find a bug, once you fix it, the first thing you should do is create a test that reproduces the bug and verifies the fix.

This ensures:
1. The bug is actually fixed.
2. The bug never comes back.
3. Our test suite grows to cover more edge cases over time.

## How to Add a New Test

1. **Locate the right test file**: Find the existing test file that corresponds to the module you're working on (e.g., `tests/agent/test_metta_agent.py` for `metta_agent.py`). If one doesn't exist, create a new one.

2. **Create a new test function**: Add a new function that starts with `test_`. Describe what the test is verifying in the function name (e.g., `test_load_puffer_agent_from_uri`).

3. **Write the test**:
   - **Arrange**: Set up the necessary objects, mock data, and configurations.
   - **Act**: Call the function or method you're testing.
   - **Assert**: Use `assert` statements to check that the result is what you expect.

4. **Run the test**: Run your new test to make sure it passes.
   ```bash
   python -m pytest tests/path/to/your/test_file.py::test_your_new_test
   ```

## Example: Creating a Test from a Debugging Session

Let's say we just fixed a bug where loading a Puffer agent was failing. We would add a new test to `tests/agent/test_policy_store.py`:

```python
# tests/agent/test_policy_store.py

# ... other imports ...
import pytest
from metta.agent.policy_store import PolicyStore
from omegaconf import DictConfig

def test_load_puffer_agent_from_uri():
    """Verify that a PufferAgent can be loaded from a puffer:// URI."""

    # Arrange: Create a minimal config with a puffer block
    cfg = DictConfig({
        "device": "cpu",
        "puffer": {
            "_target_": "metta.agent.external.example.Recurrent",
            "hidden_size": 512,
            "cnn_channels": 128,
        },
        "wandb": None,
    })

    # Create the policy store
    policy_store = PolicyStore(cfg, wandb_run=None)

    # Act: Load the agent
    # Note: This requires the checkpoint to exist at this path
    puffer_agent = policy_store.load_from_uri("puffer://checkpoints/metta-new/metta.pt")

    # Assert: Check that the agent was loaded correctly
    assert puffer_agent is not None
    assert puffer_agent.model_type == "puffer"
    assert "pufferlib" in str(type(puffer_agent.model))
    assert puffer_agent.model.hidden_size == 512
```

## Running the Full Test Suite

To run all tests, use:
```bash
python -m pytest tests/
```

By consistently turning our debugging work into permanent tests, we'll build a more stable and reliable codebase.
