#!/usr/bin/env python3
"""
Quick regression test for the trainer/distributed-helper handshake.

Repro steps (no actual GPUs required):
    uv run metta/debug/test_trainer_device_regression.py --device cuda:1

Before the trainer refactor at 067f94b7d6fd12be1d8a9642ee6c9b0d24118175,
the DistributedHelper inherited the exact torch device (`cuda:1`, etc.)
from the caller.  HEAD now rebuilds a helper with `SystemConfig(device=self._device.type)`,
which drops the ordinal and silently redirects every rank to `cuda:0`.

This script mimics both code paths and fails (exit code 1) when the new
logic loses the ordinal so you can quickly validate the hypothesis.
"""

from __future__ import annotations

import argparse
import sys

import torch

from metta.rl.system_config import SystemConfig


def _helper_device_seen_by_trainer(trainer_device: torch.device) -> torch.device:
    """Mimic the new Trainer logic that rebuilds a helper from device.type."""
    helper_cfg = SystemConfig(device=trainer_device.type)
    return torch.device(helper_cfg.device)


def _expected_helper_device(trainer_device: torch.device) -> torch.device:
    """Baseline behaviour where the helper sees the full ordinal."""
    helper_cfg = SystemConfig(device=str(trainer_device))
    return torch.device(helper_cfg.device)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--device",
        default="cuda:1",
        help="Device the trainer believes it is using (default: %(default)s)",
    )
    args = parser.parse_args()

    trainer_device = torch.device(args.device)
    buggy_helper_device = _helper_device_seen_by_trainer(trainer_device)
    expected_helper_device = _expected_helper_device(trainer_device)

    print(f"Trainer requested device : {trainer_device}")
    print(f"Helper sees (new logic)  : {buggy_helper_device}")
    print(f"Helper should see        : {expected_helper_device}")

    if buggy_helper_device != expected_helper_device:
        print(
            "\nMismatch detected: helper lost the CUDA ordinal. "
            "All ranks will contended for the same GPU when the trainer "
            "recreates DistributedHelper with SystemConfig(device=self._device.type)."
        )
        return 1

    print("\nNo mismatch detected. Trainer is preserving the CUDA ordinal.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
