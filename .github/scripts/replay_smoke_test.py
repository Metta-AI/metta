#!/usr/bin/env -S uv run
# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "psutil>=6.0.0",
# ]
# ///
"""
Run replay smoke tests with benchmarking.
"""

import os
import sys

# Add parent directory to path to import utils
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.smoke_test import SmokeTest


class ReplaySmokeTest(SmokeTest):
    """Replay smoke test implementation."""

    def get_command(self) -> list[str]:
        return [
            "uv",
            "run",
            "./tools/replay.py",
            "+hardware=github",
            "wandb=off",
        ]

    def get_timeout(self) -> int:
        return int(os.environ.get("REPLAY_TIMEOUT", "300"))


if __name__ == "__main__":
    test = ReplaySmokeTest()
    sys.exit(test.run())
