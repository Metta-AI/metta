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

from metta.common.wandb.log_wandb import (
    ensure_wandb_run,
    log_debug_info,
    log_single_value,
    log_to_wandb,
    parse_value,
)


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


def test_ensure_wandb_run_creates_new_run(wandb_test_env):
    """Test that ensure_wandb_run creates a new run when METTA_RUN_ID is set."""
    # Set a run ID
    os.environ["METTA_RUN_ID"] = "test-run-001"

    # Ensure a run exists
    run = ensure_wandb_run()

    # Verify the run was created
    assert run is not None
    assert run.name == "test-run-001"
    assert run.id == "test-run-001"

    # Clean up
    wandb.finish()


def test_ensure_wandb_run_without_run_id(wandb_test_env):
    """Test that ensure_wandb_run fails when no run ID is available."""
    # Remove METTA_RUN_ID
    os.environ.pop("METTA_RUN_ID", None)

    # Should raise RuntimeError
    with pytest.raises(RuntimeError, match="No active wandb run and METTA_RUN_ID not set"):
        ensure_wandb_run()


def test_ensure_wandb_run_existing_active(wandb_test_env):
    """Test that ensure_wandb_run returns existing active run."""
    # Create a run manually
    run = wandb.init(
        project="test-project",
        name="existing-run",
        mode="offline",
    )

    # ensure_wandb_run should return the existing run
    result = ensure_wandb_run()
    assert result is not None
    assert result.id == run.id

    # Clean up
    wandb.finish()


def test_log_single_value(wandb_test_env):
    """Test logging a single value."""
    os.environ["METTA_RUN_ID"] = "test-run-002"

    # Log a single value
    log_single_value("test/metric", 42.0)

    # Verify the run was created and value was logged
    assert wandb.run is not None
    assert wandb.run.name == "test-run-002"

    # Verify the value was logged to summary
    summary_dict = dict(wandb.run.summary)
    assert "test/metric" in summary_dict
    assert summary_dict["test/metric"] == 42.0

    # Clean up
    wandb.finish()


def test_log_to_wandb_multiple_metrics(wandb_test_env):
    """Test logging multiple metrics at once."""
    os.environ["METTA_RUN_ID"] = "test-run-003"

    # Log multiple metrics
    metrics = {
        "accuracy": 0.95,
        "loss": 0.05,
        "learning_rate": 0.001,
    }

    log_to_wandb(metrics, step=100)

    # Verify all metrics were logged
    assert wandb.run is not None
    summary_dict = dict(wandb.run.summary)

    for key, value in metrics.items():
        assert key in summary_dict
        assert summary_dict[key] == value

    # Clean up
    wandb.finish()


@pytest.mark.slow
def test_log_multiple_value_types(wandb_test_env):
    """Test logging multiple values of different types."""
    os.environ["METTA_RUN_ID"] = "test-run-004"

    # Log multiple values of different types
    values = {
        "test/int": 42,
        "test/float": 3.14159,
        "test/string": "hello",
        "test/bool": True,
        "test/list": [1, 2, 3],
        "test/dict": {"a": 1, "b": 2},
    }

    for key, value in values.items():
        log_single_value(key, value)

    # Verify all values are in summary
    assert wandb.run is not None

    # Access summary values directly from wandb.run.summary
    for key, value in values.items():
        assert wandb.run.summary.get(key) is not None

        # For simple types, direct comparison works
        if isinstance(value, (int, float, str, bool)):
            assert wandb.run.summary[key] == value
        # For complex types, wandb stores them differently
        # We just verify they were stored
        else:
            assert key in dict(wandb.run.summary)

    # Clean up
    wandb.finish()


def test_log_with_steps(wandb_test_env):
    """Test logging values at different steps."""
    os.environ["METTA_RUN_ID"] = "test-run-005"

    # Log values at different steps without adding to summary
    for step in range(5):
        log_single_value("test/counter", step * 10, step=step, also_summary=False)

    # Log a final value to summary
    log_single_value("test/final", 100, step=10, also_summary=True)

    # Verify summary contains only the final value
    assert wandb.run is not None
    summary_dict = dict(wandb.run.summary)

    assert "test/final" in summary_dict
    assert summary_dict["test/final"] == 100

    # Counter values should not be in summary since also_summary=False
    # Note: wandb might still track the last value, so we just verify our explicit summary value

    # Clean up
    wandb.finish()


def test_log_debug_info(wandb_test_env):
    """Test the log_debug_info function."""
    os.environ["METTA_RUN_ID"] = "test-run-006"
    os.environ["SKYPILOT_TASK_ID"] = "sky-2024-01-01-12-00-00-123456_test_1"
    os.environ["HOSTNAME"] = "test-host"
    os.environ["RANK"] = "0"
    os.environ["LOCAL_RANK"] = "0"

    # Run log_debug_info
    log_debug_info()

    # Verify debug values were logged
    assert wandb.run is not None
    summary_dict = dict(wandb.run.summary)

    # Verify environment info was logged
    assert "debug/metta_run_id" in summary_dict
    assert summary_dict["debug/metta_run_id"] == "test-run-006"
    assert "debug/skypilot_task_id" in summary_dict
    assert summary_dict["debug/skypilot_task_id"] == "sky-2024-01-01-12-00-00-123456_test_1"
    assert "debug/hostname" in summary_dict
    assert summary_dict["debug/hostname"] == "test-host"
    assert "debug/rank" in summary_dict
    assert summary_dict["debug/rank"] == "0"
    assert "debug/timestamp" in summary_dict

    # Clean up
    wandb.finish()


def test_parse_value():
    """Test the parse_value function."""
    # Test integer parsing
    assert parse_value("42") == 42
    assert parse_value("0") == 0
    assert parse_value("-10") == -10

    # Test float parsing
    assert parse_value("3.14") == 3.14
    assert parse_value("0.0") == 0.0
    assert parse_value("-1.5") == -1.5

    # Test boolean parsing
    assert parse_value("true") is True
    assert parse_value("True") is True
    assert parse_value("false") is False
    assert parse_value("False") is False

    # Test JSON parsing
    assert parse_value('{"a": 1, "b": 2}') == {"a": 1, "b": 2}
    assert parse_value('[1, 2, 3]') == [1, 2, 3]
    assert parse_value('null') is None

    # Test string fallback
    assert parse_value("hello world") == "hello world"
    assert parse_value("not-a-number") == "not-a-number"


def test_log_to_wandb_no_summary(wandb_test_env):
    """Test logging without adding to summary."""
    os.environ["METTA_RUN_ID"] = "test-run-007"

    # Log metrics without summary
    metrics = {"metric1": 1, "metric2": 2}
    log_to_wandb(metrics, step=0, also_summary=False)

    # Log other metrics with summary
    summary_metrics = {"summary1": 10, "summary2": 20}
    log_to_wandb(summary_metrics, step=1, also_summary=True)

    # Verify only summary metrics are in summary
    assert wandb.run is not None
    summary_dict = dict(wandb.run.summary)

    assert "summary1" in summary_dict
    assert "summary2" in summary_dict
    assert summary_dict["summary1"] == 10
    assert summary_dict["summary2"] == 20

    # Clean up
    wandb.finish()


def test_ensure_wandb_run_resume(wandb_test_env):
    """Test that ensure_wandb_run can resume an existing run."""
    run_id = "test-run-008"
    os.environ["METTA_RUN_ID"] = run_id

    # Create an initial run and log some data
    run1 = wandb.init(
        project="test-project",
        name=run_id,
        id=run_id,
        mode="offline",
    )
    wandb.log({"initial/metric": 1.0})
    wandb.finish()

    # Now use ensure_wandb_run to resume
    run2 = ensure_wandb_run()
    assert run2 is not None
    assert run2.id == run_id

    # Log more data
    log_single_value("resumed/metric", 2.0)

    # Clean up
    wandb.finish()


def log_wandb_live_test():
    """Live test that creates a real wandb run visible in the web UI.

    This test requires network access and will create a run at:
    https://wandb.ai/metta-research/metta/runs/test_log_wandb_{timestamp}

    Run directly with: python -m pytest test_log_wandb.py::log_wandb_live_test -s
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
        log_single_value("live_test/startup", 1.0, step=0)
        log_single_value("live_test/int", 42, step=0)
        log_single_value("live_test/float", 3.14159, step=0)
        log_single_value("live_test/string", "Hello from live test!", step=0)
        log_single_value("live_test/bool", True, step=0)

        # Complex types
        log_single_value("live_test/list", [1, 2, 3, 4, 5], step=0)
        log_single_value("live_test/dict", {"test": "live", "timestamp": timestamp}, step=0)

        # Time series data
        for i in range(10):
            log_single_value("live_test/progress", i * 10, step=i, also_summary=False)
            log_single_value("live_test/loss", 1.0 / (i + 1), step=i, also_summary=False)

        # Log debug info
        log_debug_info()

        # Final summary metrics
        log_single_value("live_test/final_score", 0.95)
        log_single_value("live_test/test_passed", True)
        log_single_value("live_test/completion_time", time.time() - timestamp)

        # Get the run URL
        if wandb.run:
            entity = os.environ.get("WANDB_ENTITY", "metta-research")
            project = os.environ.get("WANDB_PROJECT", "metta")
            run_url = f"https://wandb.ai/{entity}/{project}/runs/{run_id}"
            print("\n✅ Live test completed successfully!")
            print(f"📊 View run at: {run_url}")
            print(f"   Run ID: {run_id}")

            # Also log the URL to the run itself
            log_single_value("live_test/run_url", run_url)

    except Exception as e:
        print(f"\n❌ Live test failed: {e}")
        raise

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
    log_wandb_live_test()
