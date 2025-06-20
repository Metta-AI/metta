"""
Run replay smoke tests with benchmarking.
"""

import os
import sys
from typing import List

# Add parent directory to path to import utils
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.smoke_test import SmokeTest


class ReplaySmokeTest(SmokeTest):
    """Replay smoke test implementation."""

    def get_command(self) -> List[str]:
        return [
            "python3",
            "./tools/replay.py",
            "+hardware=github",
            "wandb=off",
        ]

    def get_timeout(self) -> int:
        return int(os.environ.get("REPLAY_TIMEOUT", "300"))


if __name__ == "__main__":
    test = ReplaySmokeTest()
    sys.exit(test.run())
