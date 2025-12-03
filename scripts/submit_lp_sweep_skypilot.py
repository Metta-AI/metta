#!/usr/bin/env -S uv run
"""Submit LP sweep jobs to SkyPilot.

This script enumerates all LP hyperparameter combinations and submits each
as a separate SkyPilot training job that runs in parallel.

Usage:
    # Submit all 18 configs
    ./scripts/submit_lp_sweep_skypilot.py --group lp_sweep_sky

    # Test with dry run
    ./scripts/submit_lp_sweep_skypilot.py --group lp_sweep_sky --dry-run

    # Limit number of jobs
    ./scripts/submit_lp_sweep_skypilot.py --group lp_sweep_sky --max-runs 6

    # Custom resources
    ./scripts/submit_lp_sweep_skypilot.py --group lp_sweep_sky --gpus 4 --max-runtime-hours 40
"""

from __future__ import annotations

import argparse
import itertools
import json
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence


@dataclass(frozen=True)
class SweepDimension:
    """Sweep axis specification."""

    key: str
    values: Sequence[object]


# LP hyperparameter grid
SWEEP_DIMENSIONS: Sequence[SweepDimension] = (
    SweepDimension("ema_timescale", [0.001, 0.005, 0.01]),
    SweepDimension("progress_smoothing", [0.01, 0.05, 0.1]),
    SweepDimension("exploration_bonus", [0.05, 0.1]),
)

# Default configuration
DEFAULT_RECIPE = "recipes.experiment.cvc.proc_maps.train"
DEFAULT_VARIANTS = ["heart_chorus", "lonely_heart"]
DEFAULT_NUM_COGS = 4
DEFAULT_TOTAL_TIMESTEPS = 100_000_000  # 100M for real learning
DEFAULT_EPOCH_INTERVAL = 50


def _format_value(value: object) -> str:
    """Format values for CLI."""
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)):
        return str(value)
    if isinstance(value, str):
        return value
    return json.dumps(value)


def _enumerate_jobs(args: argparse.Namespace) -> list[tuple[str, dict[str, object]]]:
    """Enumerate all sweep configurations with run names."""
    axes = [dim.values for dim in SWEEP_DIMENSIONS]
    jobs = []

    for idx, combo in enumerate(itertools.product(*axes), start=1):
        if args.max_runs is not None and idx > args.max_runs:
            break

        # Build overrides dict
        overrides = {
            "group": args.run_group,
            "trainer.total_timesteps": args.total_timesteps,
            "checkpointer.epoch_interval": args.epoch_interval,
            "num_cogs": args.num_cogs,
            "variants": json.dumps(DEFAULT_VARIANTS),
        }

        # Add sweep dimensions
        for dim, value in zip(SWEEP_DIMENSIONS, combo, strict=False):
            overrides[dim.key] = value

        # Generate run name
        run_name = f"lp_grid_{idx:03d}"
        overrides["run"] = run_name

        jobs.append((run_name, overrides))

    return jobs


def _build_launch_command(
    run_name: str,
    overrides: dict[str, object],
    args: argparse.Namespace,
) -> list[str]:
    """Build SkyPilot launch command."""
    cmd = ["./devops/skypilot/launch.py", args.recipe]

    # Add overrides
    for key, value in overrides.items():
        cmd.append(f"{key}={_format_value(value)}")

    # Add resource flags
    if args.gpus:
        cmd.extend(["--gpus", str(args.gpus)])
    if args.nodes:
        cmd.extend(["--nodes", str(args.nodes)])
    if args.max_runtime_hours:
        cmd.extend(["--max-runtime-hours", str(args.max_runtime_hours)])

    # Add spot flag if requested
    if args.spot:
        cmd.append("--spot")

    # Add git ref if provided
    if args.git_ref:
        cmd.extend(["--git-ref", args.git_ref])

    return cmd


def submit_jobs(args: argparse.Namespace) -> None:
    """Submit all LP sweep jobs to SkyPilot."""
    jobs = _enumerate_jobs(args)

    print(f"{'='*60}")
    print(f"LP Sweep Submission Summary")
    print(f"{'='*60}")
    print(f"Total jobs: {len(jobs)}")
    print(f"Recipe: {args.recipe}")
    print(f"Group: {args.run_group}")
    print(f"Timesteps per job: {args.total_timesteps:,}")
    print(f"Resources: {args.gpus} GPUs, {args.nodes} nodes")
    print(f"Max runtime: {args.max_runtime_hours}h per job")
    print(f"Spot instances: {args.spot}")
    print(f"{'='*60}\n")

    if args.dry_run:
        print("DRY RUN MODE - Commands that would be executed:\n")
        for run_name, overrides in jobs:
            cmd = _build_launch_command(run_name, overrides, args)
            print(f"[{run_name}]")
            print(f"  {' '.join(cmd)}")
            print(f"  LP params: ema={overrides['ema_timescale']}, "
                  f"smoothing={overrides['progress_smoothing']}, "
                  f"exploration={overrides['exploration_bonus']}\n")
        return

    # Confirm before launching
    if not args.yes:
        response = input(f"\nSubmit {len(jobs)} jobs to SkyPilot? (yes/no): ")
        if response.lower() not in ("yes", "y"):
            print("Cancelled.")
            return

    # Submit jobs
    successful = 0
    failed = 0

    for run_name, overrides in jobs:
        cmd = _build_launch_command(run_name, overrides, args)
        print(f"\n{'='*60}")
        print(f"Submitting: {run_name}")
        print(f"LP params: ema={overrides['ema_timescale']}, "
              f"smoothing={overrides['progress_smoothing']}, "
              f"exploration={overrides['exploration_bonus']}")
        print(f"{'='*60}")

        try:
            result = subprocess.run(cmd, check=True, cwd=Path.cwd())
            if result.returncode == 0:
                successful += 1
                print(f"✅ {run_name} submitted successfully")
            else:
                failed += 1
                print(f"❌ {run_name} submission failed")
        except subprocess.CalledProcessError as e:
            failed += 1
            print(f"❌ {run_name} submission failed: {e}")
        except Exception as e:
            failed += 1
            print(f"❌ {run_name} submission error: {e}")

    print(f"\n{'='*60}")
    print(f"Submission Summary")
    print(f"{'='*60}")
    print(f"Successful: {successful}/{len(jobs)}")
    print(f"Failed: {failed}/{len(jobs)}")
    print(f"\nMonitor jobs with:")
    print(f"  uv run sky jobs queue")
    print(f"  uv run sky jobs queue | grep {args.run_group}")
    print(f"\nView logs:")
    print(f"  uv run sky jobs logs <JOB_ID> --follow")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Submit LP sweep to SkyPilot",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # Sweep configuration
    parser.add_argument(
        "--run-group",
        "--group",
        dest="run_group",
        default="lp_sweep_sky",
        help="WandB group name for all runs",
    )
    parser.add_argument(
        "--recipe",
        default=DEFAULT_RECIPE,
        help="Recipe to train",
    )
    parser.add_argument(
        "--max-runs",
        type=int,
        help="Limit number of configurations to submit",
    )

    # Training parameters
    parser.add_argument(
        "--total-timesteps",
        type=int,
        default=DEFAULT_TOTAL_TIMESTEPS,
        help="Total timesteps per run (default: 100M)",
    )
    parser.add_argument(
        "--epoch-interval",
        type=int,
        default=DEFAULT_EPOCH_INTERVAL,
        help="Checkpoint epoch interval",
    )
    parser.add_argument(
        "--num-cogs",
        type=int,
        default=DEFAULT_NUM_COGS,
        help="Number of CoGs",
    )

    # SkyPilot resources
    parser.add_argument(
        "--gpus",
        type=int,
        default=8,
        help="GPUs per job",
    )
    parser.add_argument(
        "--nodes",
        type=int,
        default=1,
        help="Nodes per job",
    )
    parser.add_argument(
        "--max-runtime-hours",
        type=float,
        default=36.0,
        help="Max runtime per job (hours)",
    )
    parser.add_argument(
        "--spot",
        action="store_true",
        help="Use spot instances (cheaper but can be preempted)",
    )
    parser.add_argument(
        "--git-ref",
        help="Git ref to checkout (default: current branch HEAD)",
    )

    # Execution control
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print commands without executing",
    )
    parser.add_argument(
        "-y",
        "--yes",
        action="store_true",
        help="Skip confirmation prompt",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # Change to repo root
    repo_root = Path(__file__).parent.parent
    import os
    os.chdir(repo_root)

    submit_jobs(args)


if __name__ == "__main__":
    main()

