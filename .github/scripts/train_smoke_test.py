#!/usr/bin/env -S uv run
# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "psutil>=6.0.0",
# ]
# ///
"""
Run training smoke tests with benchmarking.
"""

import os
import sys

# Add parent directory to path to import utils
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.smoke_test import SmokeTest


class TrainingSmokeTest(SmokeTest):
    """Training smoke test implementation."""

    def get_command(self) -> list[str]:
        return [
            "uv",
            "run",
            "./tools/train.py",
            "+hardware=github",
            "wandb=off",
        ]

    def get_timeout(self) -> int:
        return int(os.environ.get("TRAINING_TIMEOUT", "600"))


if __name__ == "__main__":
    test = TrainingSmokeTest()
    sys.exit(test.run())
