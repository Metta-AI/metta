#!/usr/bin/env -S uv run
# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "pytest",
#     "wandb",
# ]
# ///
"""
Integration tests for the wandb debug logger.
These tests actually create wandb runs and verify data is logged correctly.
"""

import os
import time

import pytest
import wandb

from metta.common.wandb.log_wandb import log_debug_info, log_wandb


@pytest.fixture
def wandb_test_env(tmp_path):
    """Set up a test environment for wandb."""
    # Save original env vars
    original_env = {
        "WANDB_MODE": os.environ.get("WANDB_MODE"),
        "WANDB_PROJECT": os.environ.get("WANDB_PROJECT"),
        "WANDB_DIR": os.environ.get("WANDB_DIR"),
        "WANDB_SILENT": os.environ.get("WANDB_SILENT"),
        "METTA_RUN_ID": os.environ.get("METTA_RUN_ID"),
    }

    # Set test env vars
    os.environ["WANDB_MODE"] = "offline"  # Run in offline mode for testing
    os.environ["WANDB_PROJECT"] = "test-project"
    os.environ["WANDB_DIR"] = str(tmp_path)
    os.environ["WANDB_SILENT"] = "true"

    yield tmp_path

    # Restore original env vars
    for key, value in original_env.items():
        if value is None:
            os.environ.pop(key, None)
        else:
            os.environ[key] = value

    # Clean up any active runs
    if wandb.run is not None:
        wandb.finish()


def test_log_wandb_creates_new_run(wandb_test_env):
    """Test that log_wandb creates a new run when METTA_RUN_ID is set."""
    # Set a run ID
    os.environ["METTA_RUN_ID"] = "test-run-001"

    # Log a value
    success = log_wandb("test/metric", 42.0)
    assert success is True

    # Verify the run was created
    assert wandb.run is not None
    assert wandb.run.name == "test-run-001"
    assert wandb.run.id == "test-run-001"

    # Verify the value was logged to summary
    summary_dict = dict(wandb.run.summary)
    assert "test/metric" in summary_dict
    assert summary_dict["test/metric"] == 42.0

    # Clean up
    wandb.finish()


def test_log_wandb_without_run_id(wandb_test_env):
    """Test that log_wandb fails gracefully when no run ID is available."""
    # Remove METTA_RUN_ID
    os.environ.pop("METTA_RUN_ID", None)

    # Try to log a value
    success = log_wandb("test/metric", 42.0)
    assert success is False

    # Verify no run was created
    assert wandb.run is None


def test_log_wandb_resumes_existing_run(wandb_test_env):
    """Test that log_wandb can resume an existing run."""
    run_id = "test-run-002"
    os.environ["METTA_RUN_ID"] = run_id

    # Create an initial run and log some data
    _run = wandb.init(
        project="test-project",
        name=run_id,
        id=run_id,
        mode="offline",
    )
    wandb.log({"initial/metric": 1.0})
    wandb.finish()

    # Now use log_wandb to resume the run
    success = log_wandb("resumed/metric", 2.0)
    assert success is True

    # Verify we resumed the same run
    assert wandb.run is not None
    assert wandb.run.id == run_id

    # Clean up
    wandb.finish()


@pytest.mark.slow
def test_log_wandb_multiple_values(wandb_test_env):
    """Test logging multiple values to the same run."""
    os.environ["METTA_RUN_ID"] = "test-run-003"

    # Log multiple values
    values = {
        "test/int": 42,
        "test/float": 3.14159,
        "test/string": "hello",
        "test/bool": True,
        "test/list": [1, 2, 3],
        "test/dict": {"a": 1, "b": 2},
    }

    for key, value in values.items():
        success = log_wandb(key, value)
        assert success is True

    # Verify all values are in summary
    assert wandb.run is not None

    # Access summary values directly from wandb.run.summary
    # This handles wandb's internal object wrapping
    for key, value in values.items():
        assert wandb.run.summary.get(key) is not None

        # For simple types, direct comparison works
        if isinstance(value, (int, float, str, bool)):
            assert wandb.run.summary[key] == value
        # For complex types, wandb stores them differently
        # We just verify they were stored
        else:
            # Just verify the key exists - wandb wraps complex objects
            assert key in dict(wandb.run.summary)

    # Clean up
    wandb.finish()


def test_log_wandb_with_steps(wandb_test_env):
    """Test logging values at different steps."""
    os.environ["METTA_RUN_ID"] = "test-run-004"

    # Log values at different steps
    for step in range(5):
        success = log_wandb("test/counter", step * 10, step=step, also_summary=False)
        assert success is True

    # Also log a summary value
    success = log_wandb("test/final", 100, step=10, also_summary=True)
    assert success is True

    # Verify summary contains only the final value
    assert wandb.run is not None
    summary_dict = dict(wandb.run.summary)

    assert "test/final" in summary_dict
    assert summary_dict["test/final"] == 100

    # Note: We can't easily verify step-based logging in offline mode
    # without accessing the internal data files

    # Clean up
    wandb.finish()


def test_log_debug_info(wandb_test_env, tmp_path):
    """Test the log_debug_info function."""
    os.environ["METTA_RUN_ID"] = "test-run-005"
    os.environ["SKYPILOT_TASK_ID"] = "sky-2024-01-01-12-00-00-123456_test_1"

    # Create a fake latency file
    latency_dir = tmp_path / ".metta"
    latency_dir.mkdir(exist_ok=True)
    latency_file = latency_dir / "skypilot_latency.json"

    import json

    latency_data = {
        "latency_s": 123.45,
        "task_id": "sky-2024-01-01-12-00-00-123456_test_1",
        "run_id": "test-run-005",
        "timestamp": "2024-01-01T12:00:00",
    }
    with open(latency_file, "w") as f:
        json.dump(latency_data, f)

    # Temporarily override home directory for the test
    original_home = os.environ.get("HOME")
    os.environ["HOME"] = str(tmp_path)

    try:
        # Run log_debug_info
        log_debug_info()

        # Verify debug values were logged
        assert wandb.run is not None
        summary_dict = dict(wandb.run.summary)

        assert "debug/test_value" in summary_dict
        assert summary_dict["debug/test_value"] == 42
        assert "debug/test_float" in summary_dict
        assert summary_dict["debug/test_float"] == 3.14159

        # Verify environment info was logged
        assert "debug/metta_run_id" in summary_dict
        assert summary_dict["debug/metta_run_id"] == "test-run-005"
        assert "debug/skypilot_task_id" in summary_dict

        # Verify latency was logged
        assert "debug/skypilot_queue_latency_s" in summary_dict
        assert summary_dict["debug/skypilot_queue_latency_s"] == 123.45

    finally:
        # Restore original home
        if original_home is not None:
            os.environ["HOME"] = original_home
        else:
            os.environ.pop("HOME", None)

        # Clean up
        wandb.finish()


def test_log_wandb_existing_active_run(wandb_test_env):
    """Test that log_wandb uses an existing active run."""
    # Create a run manually
    run = wandb.init(
        project="test-project",
        name="existing-run",
        mode="offline",
    )

    # Log some initial data
    wandb.log({"initial/value": 1})

    # Now use log_wandb - it should use the existing run
    success = log_wandb("test/value", 42)
    assert success is True

    # Verify it used the same run
    assert wandb.run is not None
    assert wandb.run.id == run.id
    summary_dict = dict(wandb.run.summary)
    assert "test/value" in summary_dict
    assert summary_dict["test/value"] == 42

    # Clean up
    wandb.finish()


def log_wandb_live_test():
    """Live test that creates a real wandb run visible in the web UI.

    This test requires network access and will create a run at:
    https://wandb.ai/metta-research/metta/runs/test_log_wandb_{timestamp}

    Run directly with: python common/tests/wandb/test_log_wandb.py
    """
    # Create a unique run ID with timestamp
    timestamp = int(time.time())
    run_id = f"test_log_wandb_{timestamp}"

    # Set up environment for live run
    original_env = {
        "WANDB_MODE": os.environ.get("WANDB_MODE"),
        "WANDB_PROJECT": os.environ.get("WANDB_PROJECT"),
        "METTA_RUN_ID": os.environ.get("METTA_RUN_ID"),
        "WANDB_ENTITY": os.environ.get("WANDB_ENTITY"),
    }

    try:
        # Configure for live run
        os.environ.pop("WANDB_MODE", None)  # Remove offline mode
        os.environ["WANDB_PROJECT"] = "metta"
        os.environ["WANDB_ENTITY"] = "metta-research"
        os.environ["METTA_RUN_ID"] = run_id

        # Log various types of data
        print(f"\n🚀 Creating live wandb run: {run_id}")

        # Basic metrics (all at step 0)
        assert log_wandb("live_test/startup", 1.0, step=0) is True
        assert log_wandb("live_test/int", 42, step=0) is True
        assert log_wandb("live_test/float", 3.14159, step=0) is True
        assert log_wandb("live_test/string", "Hello from live test!", step=0) is True
        assert log_wandb("live_test/bool", True, step=0) is True

        # Complex types
        assert log_wandb("live_test/list", [1, 2, 3, 4, 5], step=0) is True
        assert log_wandb("live_test/dict", {"test": "live", "timestamp": timestamp}, step=0) is True

        # Time series data
        for i in range(10):
            assert log_wandb("live_test/progress", i * 10, step=i, also_summary=False) is True
            assert log_wandb("live_test/loss", 1.0 / (i + 1), step=i, also_summary=False) is True

        # Log debug info (no step specified, just summary)
        log_debug_info()

        # Final summary metrics (no step needed, just summary)
        assert log_wandb("live_test/final_score", 0.95) is True
        assert log_wandb("live_test/test_passed", True) is True
        assert log_wandb("live_test/completion_time", time.time() - timestamp) is True

        # Get the run URL
        if wandb.run:
            entity = os.environ.get("WANDB_ENTITY", "metta-research")
            project = os.environ.get("WANDB_PROJECT", "metta")
            run_url = f"https://wandb.ai/{entity}/{project}/runs/{run_id}"
            print("\n✅ Live test completed successfully!")
            print(f"📊 View run at: {run_url}")
            print(f"   Run ID: {run_id}")

            # Also log the URL to the run itself
            log_wandb("live_test/run_url", run_url)

    finally:
        # Clean up
        if wandb.run is not None:
            wandb.finish()

        # Restore original environment
        for key, value in original_env.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


if __name__ == "__main__":
    # When running this script directly, run the live test
    # When importing for pytest, only the test_* functions will be discovered
    log_wandb_live_test()
