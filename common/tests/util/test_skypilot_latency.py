import datetime
import importlib
import os
import re
import subprocess
import sys
import uuid

import pytest


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

    @pytest.mark.parametrize("prefix", ["sky-", "managed-sky-", "sky-managed-"])
    def test_queue_latency_with_prefixes(self, prefix):
        """Test with different valid prefixes."""
        ts = datetime.datetime.utcnow().strftime("%Y-%m-%d-%H-%M-%S-%f")
        os.environ["SKYPILOT_TASK_ID"] = f"{prefix}{ts}_demo_{uuid.uuid4().hex[:3]}"

        mod = importlib.import_module("metta.common.util.skypilot_latency")
        latency = mod.queue_latency_s()
        assert latency is not None and 0 <= latency < 1

    def test_queue_latency_no_env(self):
        """Test when SKYPILOT_TASK_ID is not set."""
        os.environ.pop("SKYPILOT_TASK_ID", None)
        mod = importlib.import_module("metta.common.util.skypilot_latency")
        assert mod.queue_latency_s() is None

    def test_queue_latency_invalid_format(self):
        """Test with invalid task ID format."""
        os.environ["SKYPILOT_TASK_ID"] = "invalid-format"
        mod = importlib.import_module("metta.common.util.skypilot_latency")
        assert mod.queue_latency_s() is None

    def test_main_function_output(self):
        """Test the main() function's output."""
        ts = datetime.datetime.utcnow() - datetime.timedelta(seconds=5)
        task_id = f"sky-{ts:%Y-%m-%d-%H-%M-%S-%f}_test_123"

        env = os.environ.copy()
        env["SKYPILOT_TASK_ID"] = task_id
        env.pop("WANDB_API_KEY", None)
        env.pop("WANDB_RUN_NAME", None)
        env.pop("METTA_RUN_ID", None)  # Also remove METTA_RUN_ID to avoid wandb logging

        # Fix: Find the script relative to the test file
        # Since we're running from the 'common' directory in CI,
        # and both files are in the same directory structure
        script_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
            "src",
            "metta",
            "common",
            "util",
            "skypilot_latency.py",
        )

        result = subprocess.run(
            [sys.executable, script_path],
            capture_output=True,
            text=True,
            env=env,
        )

        assert result.returncode == 0
        assert "SkyPilot queue latency:" in result.stdout
        assert task_id in result.stdout

        # Check latency is approximately correct
        match = re.search(r"SkyPilot queue latency: ([\d.]+) s", result.stdout)
        assert match is not None
        latency = float(match.group(1))
        assert 4 < latency < 10  # Should be around 5 seconds
