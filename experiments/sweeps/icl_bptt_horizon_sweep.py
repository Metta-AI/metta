#!/usr/bin/env python3
"""
Launch Skypilot jobs to test different BPTT horizons for in-context learning.
Maintains roughly constant memory usage by scaling batch size inversely with horizon.
"""

import subprocess
import time
from typing import List, Tuple

# Base memory budget (current config): batch_size × bptt_horizon
MEMORY_BUDGET = 2064384 * 256  # ~528M tokens

# Configuration: (horizon, batch_size, name_suffix)
# Note: Batch sizes are chosen to keep memory ~constant AND satisfy divisibility constraints:
#   - batch_size % minibatch_size == 0
#   - minibatch_size % bptt_horizon == 0
#   - (batch_size // bptt_horizon) % (minibatch_size // bptt_horizon) == 0
CONFIGS: List[Tuple[int, int, str]] = [
    (128, 4128768, "h128"),  # 2x batch size for 0.5x horizon
    (256, 2064384, "h256"),  # baseline
    (512, 1032192, "h512"),  # 0.5x batch size for 2x horizon
    # For 1024, adjust to nearest multiple that works with default minibatch (16384) → 524288
    (1024, 524288, "h1024"),
]


def _minibatch_size_for_horizon(horizon: int) -> int:
    """Return a valid minibatch_size for the given horizon.

    Defaults to 16384, except where divisibility requires a different value.
    """
    if horizon == 768:
        # 49152 = 768 * 64; also divides 688128 cleanly
        return 49152
    # 16384 works for 128, 256, 512, 1024 (with batch_size=524288)
    return 16384


# Number of replicates per configuration
REPLICATES = 3

# Base experiment name
BASE_NAME = "icl_bptt_sweep"
GROUP_NAME = f"{BASE_NAME}_{int(time.time())}"


def launch_job(
    horizon: int,
    batch_size: int,
    suffix: str,
    replicate: int,
    group: str,
) -> subprocess.CompletedProcess:
    """Launch a single Skypilot job with the given configuration."""

    job_name = f"{BASE_NAME}_{suffix}_rep{replicate}"

    # Build the command
    minibatch_size = _minibatch_size_for_horizon(horizon)

    # Use the project launcher to create and validate SkyPilot tasks
    cmd = [
        "uv",
        "run",
        "./devops/skypilot/launch.py",
        "experiments.recipes.icl_resource_chain.train",
        "--run",
        job_name,
        "--args",
        f"group={group}",
        "--overrides",
        f"trainer.bptt_horizon={horizon}",
        f"trainer.batch_size={batch_size}",
        f"trainer.minibatch_size={minibatch_size}",
    ]

    # Increase heartbeat timeout only for long horizons (to tolerate longer checkpoint uploads)
    if horizon in (512, 1024):
        cmd.extend(["--heartbeat-timeout-seconds", "1600"])  # 26 minutes

    print(f"\nLaunching {job_name}...")
    print(f"  Horizon: {horizon}, Batch size: {batch_size:,}")
    print(f"  Minibatch size: {minibatch_size:,}")
    print(f"  Memory budget: {batch_size * horizon:,} tokens")
    print(f"  Command: {' '.join(cmd)}")

    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode != 0:
        print(f"ERROR launching {job_name}:")
        print(result.stderr)
    else:
        print(f"Successfully launched {job_name}")
        print(result.stdout)

    return result


def main():
    """Launch all jobs in the sweep."""
    print(f"Starting BPTT horizon sweep: {GROUP_NAME}")
    print(f"Configurations: {len(CONFIGS)}")
    print(f"Replicates per config: {REPLICATES}")
    print(f"Total jobs: {len(CONFIGS) * REPLICATES}")
    print("\nConfigurations:")
    for horizon, batch_size, suffix in CONFIGS:
        print(
            f"  - Horizon {horizon:4d}: batch_size={batch_size:,} ({batch_size * horizon:,} tokens)"
        )

    # Confirm before launching
    response = input("\nProceed with launching? (y/N): ")
    if response.lower() != "y":
        print("Aborted.")
        return

    successful_jobs = []
    failed_jobs = []

    # Launch all jobs
    for horizon, batch_size, suffix in CONFIGS:
        for rep in range(1, REPLICATES + 1):
            result = launch_job(horizon, batch_size, suffix, rep, GROUP_NAME)

            job_info = f"{suffix}_rep{rep}"
            if result.returncode == 0:
                successful_jobs.append(job_info)
            else:
                failed_jobs.append(job_info)

            # Small delay between launches to avoid overwhelming the system
            time.sleep(2)

    # Summary
    print("\n" + "=" * 60)
    print("LAUNCH SUMMARY")
    print("=" * 60)
    print(f"Total jobs attempted: {len(CONFIGS) * REPLICATES}")
    print(f"Successful launches: {len(successful_jobs)}")
    print(f"Failed launches: {len(failed_jobs)}")

    if failed_jobs:
        print("\nFailed jobs:")
        for job in failed_jobs:
            print(f"  - {job}")

    print(f"\nWandB group: {GROUP_NAME}")
    print("\nTo monitor jobs:")
    print("  sky status")
    print("\nTo view logs:")
    print("  sky logs <job_name>")
    print("\nTo cancel all jobs:")
    print(f"  sky cancel {BASE_NAME}_*")


if __name__ == "__main__":
    main()
