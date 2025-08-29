#!/usr/bin/env -S uv run
# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "pytest",
# ]
# ///
"""
Tests for the SkyPilot latency calculation utility.
"""

import datetime
import os
import subprocess
import sys
from pathlib import Path

import pytest

from metta.common.util.skypilot_latency import (
    calculate_queue_latency,
    parse_submission_timestamp,
)


class TestSkyPilotLatency:
    """Tests for the SkyPilot latency helper."""

    @pytest.fixture(autouse=True)
    def preserve_env(self):
        """Preserve original environment variables."""
        original_task_id = os.environ.get("SKYPILOT_TASK_ID")
        yield
        if original_task_id is None:
            os.environ.pop("SKYPILOT_TASK_ID", None)
        else:
            os.environ["SKYPILOT_TASK_ID"] = original_task_id

    def test_parse_submission_timestamp_valid(self):
        """Test parsing valid task IDs."""
        # Test standard format
        task_id = "sky-2024-01-15-14-30-45-123456_cluster_1"
        ts = parse_submission_timestamp(task_id)
        assert ts.year == 2024
        assert ts.month == 1
        assert ts.day == 15
        assert ts.hour == 14
        assert ts.minute == 30
        assert ts.second == 45
        assert ts.microsecond == 123456
        assert ts.tzinfo == datetime.timezone.utc

    def test_parse_submission_timestamp_managed(self):
        """Test parsing managed task IDs."""
        task_id = "sky-managed-2024-12-25-00-00-00-000000_managed-cluster_42"
        ts = parse_submission_timestamp(task_id)
        assert ts.year == 2024
        assert ts.month == 12
        assert ts.day == 25
        assert ts.hour == 0
        assert ts.minute == 0
        assert ts.second == 0
        assert ts.microsecond == 0

    def test_parse_submission_timestamp_invalid_format(self):
        """Test parsing invalid task IDs."""
        invalid_ids = [
            "invalid-format",
            "sky-invalid-timestamp_cluster_1",
            "not-a-task-id",
            "",
            "sky-_cluster_1",
            "sky-2024-13-01-25-61-61-000000_cluster_1",  # Invalid date/time
        ]

        for task_id in invalid_ids:
            with pytest.raises(ValueError):
                parse_submission_timestamp(task_id)

    def test_parse_submission_timestamp_extended_microseconds(self):
        """Test parsing task IDs with more than 6 microsecond digits."""
        # SkyPilot sometimes uses 9 digits for microseconds
        task_id = "sky-2024-01-15-14-30-45-123456789_cluster_1"
        ts = parse_submission_timestamp(task_id)
        # Should truncate to 6 digits
        assert ts.microsecond == 123456

    def test_calculate_queue_latency_success(self):
        """Test successful latency calculation."""
        # Create a timestamp 5 seconds ago
        past_time = datetime.datetime.now(datetime.timezone.utc) - datetime.timedelta(seconds=5)
        task_id = f"sky-{past_time:%Y-%m-%d-%H-%M-%S-%f}_test_123"
        os.environ["SKYPILOT_TASK_ID"] = task_id

        latency = calculate_queue_latency()

        # Should be approximately 5 seconds (allow some tolerance)
        assert 4.5 < latency < 6.0

    def test_calculate_queue_latency_no_env(self):
        """Test when SKYPILOT_TASK_ID is not set."""
        os.environ.pop("SKYPILOT_TASK_ID", None)

        with pytest.raises(RuntimeError, match="SKYPILOT_TASK_ID environment variable not set"):
            calculate_queue_latency()

    def test_calculate_queue_latency_invalid_format(self):
        """Test with invalid task ID format."""
        os.environ["SKYPILOT_TASK_ID"] = "invalid-format"

        with pytest.raises(ValueError, match="Invalid task ID format"):
            calculate_queue_latency()

    @pytest.mark.parametrize("prefix", ["sky-", "sky-managed-"])
    def test_calculate_queue_latency_with_prefixes(self, prefix):
        """Test with different valid prefixes."""
        now = datetime.datetime.now(datetime.timezone.utc)
        ts_str = now.strftime("%Y-%m-%d-%H-%M-%S-%f")
        os.environ["SKYPILOT_TASK_ID"] = f"{prefix}{ts_str}_demo_123"

        latency = calculate_queue_latency()
        # Should be very close to 0 since we just created it
        assert 0 <= latency < 1

    def test_main_function_success(self):
        """Test the main() function with valid task ID."""
        # Create a timestamp 10 seconds ago
        past_time = datetime.datetime.now(datetime.timezone.utc) - datetime.timedelta(seconds=10)
        task_id = f"sky-{past_time:%Y-%m-%d-%H-%M-%S-%f}_test_123"

        env = os.environ.copy()
        env["SKYPILOT_TASK_ID"] = task_id

        # Find the script
        script_path = Path(__file__).parent.parent.parent / "skypilot_latency.py"
        if not script_path.exists():
            # Try alternative path structure
            script_path = Path(__file__).parent.parent.parent.parent / "src" / "metta" / "common" / "util" / "skypilot_latency.py"

        result = subprocess.run(
            [sys.executable, str(script_path)],
            capture_output=True,
            text=True,
            env=env,
        )

        assert result.returncode == 0

        # Check stdout contains just the latency number
        try:
            latency = float(result.stdout.strip())
            assert 9.5 < latency < 11.0  # Should be around 10 seconds
        except ValueError:
            pytest.fail(f"Expected float in stdout, got: {result.stdout}")

        # Check stderr contains the info message
        assert "SkyPilot queue latency:" in result.stderr
        assert task_id in result.stderr

    def test_main_function_no_env(self):
        """Test the main() function without SKYPILOT_TASK_ID."""
        env = os.environ.copy()
        env.pop("SKYPILOT_TASK_ID", None)

        # Find the script
        script_path = Path(__file__).parent.parent.parent / "skypilot_latency.py"
        if not script_path.exists():
            script_path = Path(__file__).parent.parent.parent.parent / "src" / "metta" / "common" / "util" / "skypilot_latency.py"

        result = subprocess.run(
            [sys.executable, str(script_path)],
            capture_output=True,
            text=True,
            env=env,
        )

        assert result.returncode == 1
        assert "Error:" in result.stderr
        assert "SKYPILOT_TASK_ID environment variable not set" in result.stderr

    def test_main_function_invalid_format(self):
        """Test the main() function with invalid task ID."""
        env = os.environ.copy()
        env["SKYPILOT_TASK_ID"] = "not-a-valid-task-id"

        # Find the script
        script_path = Path(__file__).parent.parent.parent / "skypilot_latency.py"
        if not script_path.exists():
            script_path = Path(__file__).parent.parent.parent.parent / "src" / "metta" / "common" / "util" / "skypilot_latency.py"

        result = subprocess.run(
            [sys.executable, str(script_path)],
            capture_output=True,
            text=True,
            env=env,
        )

        assert result.returncode == 1
        assert "Error:" in result.stderr
        assert "Invalid task ID format" in result.stderr

    def test_queue_latency_precision(self):
        """Test that latency calculation maintains reasonable precision."""
        # Create timestamps with microsecond precision
        for microseconds in [0, 123456, 999999]:
            past_time = datetime.datetime.now(datetime.timezone.utc).replace(microsecond=0) - datetime.timedelta(seconds=2, microseconds=microseconds)
            task_id = f"sky-{past_time:%Y-%m-%d-%H-%M-%S-%f}_test_123"
            os.environ["SKYPILOT_TASK_ID"] = task_id

            latency = calculate_queue_latency()

            # Should be approximately 2 seconds plus the microseconds
            expected = 2 + (microseconds / 1_000_000)
            # Allow 0.1 second tolerance for test execution time
            assert abs(latency - expected) < 0.1
