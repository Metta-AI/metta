#!/usr/bin/env -S uv run
# /// script
# requires-python = ">=3.11"
# dependencies = [
# "psutil>=6.0.0",
# ]
# ///
"""
Run training smoke tests with benchmarking using torchrun.
"""

import os
import sys

# Add parent directory to path to import utils
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.smoke_test import SmokeTest


class TrainingSmokeTest(SmokeTest):
    """Training smoke test implementation."""

    def get_command(self) -> list[str]:
        # Set required environment variables
        os.environ.setdefault("PYTHONUNBUFFERED", "1")
        os.environ.setdefault("PYTHONOPTIMIZE", "1")
        os.environ.setdefault("HYDRA_FULL_ERROR", "1")
        os.environ.setdefault("WANDB_DIR", "./wandb")
        os.environ.setdefault("DATA_DIR", "./train_dir")

        return [
            "uv",
            "run",
            "torchrun",
            f"--nnodes={os.environ.get('NUM_NODES', '1')}",
            f"--nproc-per-node={os.environ.get('NUM_GPUS', '1')}",
            f"--master-addr={os.environ.get('MASTER_ADDR', 'localhost')}",
            f"--master-port={os.environ.get('MASTER_PORT', '12345')}",
            f"--node-rank={os.environ.get('NODE_INDEX', '0')}",
            "tools/train.py",
            "trainer.num_workers=null",
            "+user=ci",
            "wandb=off",
        ]

    def get_timeout(self) -> int:
        return int(os.environ.get("TRAINING_TIMEOUT", "600"))


if __name__ == "__main__":
    test = TrainingSmokeTest()
    sys.exit(test.run())
