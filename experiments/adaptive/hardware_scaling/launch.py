#!/usr/bin/env python3
"""Launch FoM sweeps for different hardware configurations.

Example:
  python experiments/adaptive/hardware_scaling/launch.py \
      --experiment-name my_fom \
      --gpu-counts 1 2 4 8 \
      --node-counts 1 2 \
      --total-timesteps 300000000
"""

import argparse
import subprocess
import time
from typing import List, Tuple


def get_hardware_pairs(gpu_counts: List[int], node_counts: List[int]) -> List[Tuple[int, int]]:
    """Generate valid (gpu, node) pairs."""
    return [(g, n) for g in gpu_counts for n in node_counts if g <= 8 * n]


def launch_sweep(
    experiment_name: str,
    gpus: int,
    nodes: int,
    total_timesteps: int,
) -> subprocess.Popen:
    """Launch a single sweep for given hardware configuration."""
    sweep_name = f"{experiment_name}.g{gpus}n{nodes}"

    cmd = [
        "uv",
        "run",
        "./tools/run.py",
        "experiments.adaptive.hardware_scaling.fom_sweep",
        f"sweep_name={sweep_name}",
        f"gpus={gpus}",
        f"nodes={nodes}",
        f"total_timesteps={total_timesteps}",
    ]

    print(f"Launching {sweep_name}: {' '.join(cmd)}")
    return subprocess.Popen(cmd)


def main():
    parser = argparse.ArgumentParser(description="Launch FoM hardware scaling sweeps")

    # Essential parameters
    parser.add_argument(
        "--experiment-name",
        type=str,
        required=True,
        help="Base name for the experiment (e.g., 'my_fom_experiment')",
    )
    parser.add_argument(
        "--gpu-counts",
        type=int,
        nargs="+",
        default=[1, 2, 4, 8],
        help="GPU counts to test",
    )
    parser.add_argument(
        "--node-counts",
        type=int,
        nargs="+",
        default=[1, 2],
        help="Node counts to test",
    )
    parser.add_argument(
        "--total-timesteps",
        type=int,
        default=300_000_000,
        help="Total timesteps per trial",
    )
    parser.add_argument(
        "--sequential",
        action="store_true",
        help="Run sweeps sequentially instead of in parallel",
    )

    args = parser.parse_args()

    # Generate hardware pairs
    pairs = get_hardware_pairs(args.gpu_counts, args.node_counts)
    print(f"Launching {len(pairs)} sweeps for configurations: {pairs}")

    processes = []

    for gpus, nodes in pairs:
        proc = launch_sweep(
            args.experiment_name,
            gpus,
            nodes,
            args.total_timesteps,
        )

        if args.sequential:
            # Wait for this sweep to complete before launching next
            ret = proc.wait()
            print(f"Sweep g{gpus}n{nodes} completed with code {ret}")
        else:
            # Collect process for later monitoring
            processes.append(proc)
            time.sleep(60)  # Brief delay between launches

    # If running in parallel, wait for all to complete
    if not args.sequential and processes:
        print("All sweeps launched. Waiting for completion (Ctrl+C to leave running)...")
        try:
            for p in processes:
                p.wait()
        except KeyboardInterrupt:
            print("Leaving sweeps running in background.")

    print("Done.")


if __name__ == "__main__":
    main()