#!/usr/bin/env -S uv run
"""Request evaluation for all wandb runs matching a pattern."""

import argparse
import subprocess
from typing import List

import wandb

from metta.common.util.constants import METTA_WANDB_ENTITY, METTA_WANDB_PROJECT
from metta.setup.utils import info, success, warning


def find_runs_by_pattern(
    pattern: str,
    entity: str = METTA_WANDB_ENTITY,
    project: str = METTA_WANDB_PROJECT,
    limit: int = 1000,
) -> List[str]:
    """Find all wandb runs containing the given pattern in their name."""
    api = wandb.Api()
    runs = api.runs(f"{entity}/{project}")

    matching_runs = []
    count = 0
    for run in runs:
        if count >= limit:
            break
        if pattern in run.name:
            matching_runs.append(run.name)
            count += 1

    return matching_runs


def execute_request_eval(
    run_names: List[str],
    eval_name: str,
    git_hash: str | None = None,
    policy_select_type: str = "all",
    dry_run: bool = False,
    additional_args: List[str] = None,
    batch_size: int = 10,
) -> None:
    """Execute request_eval.py for the given runs in batches."""
    if not run_names:
        warning("No matching runs found")
        return

    # Process runs in batches to avoid URL length limits
    total_runs = len(run_names)
    info(f"Processing {total_runs} runs in batches of {batch_size}")

    for i in range(0, total_runs, batch_size):
        batch = run_names[i : i + batch_size]
        batch_num = (i // batch_size) + 1
        total_batches = (total_runs + batch_size - 1) // batch_size

        info(f"Processing batch {batch_num}/{total_batches} ({len(batch)} runs)")

        # Construct the command for this batch
        cmd = ["./tools/request_eval.py"]

        # Add policy arguments for this batch
        for run_name in batch:
            cmd.extend(["--policy", f"wandb://run/{run_name}"])

        # Add eval argument
        cmd.extend(["--eval", eval_name])

        # Add optional arguments
        if git_hash:
            cmd.extend(["--git-hash", git_hash])

        cmd.extend(["--policy-select-type", policy_select_type])

        if dry_run:
            cmd.append("--dry-run")

        # Add any additional arguments
        if additional_args:
            cmd.extend(additional_args)

        if dry_run:
            info(f"Batch {batch_num} command: {' '.join(cmd)}")
        else:
            try:
                subprocess.run(cmd, check=True)
                success(f"Batch {batch_num}/{total_batches} completed successfully")
            except subprocess.CalledProcessError as e:
                warning(f"Batch {batch_num} failed with exit code {e.returncode}")
                # Continue with next batch instead of exiting
                continue

    if not dry_run:
        success(f"Finished processing all {total_batches} batches")


def main():
    parser = argparse.ArgumentParser(
        description="Request evaluation for all wandb runs matching a pattern",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--pattern",
        type=str,
        required=True,
        help="Pattern to match in run names (e.g., 'tick_timing')",
    )

    parser.add_argument(
        "--eval",
        type=str,
        required=True,
        help="Evaluation suite name",
    )

    parser.add_argument(
        "--git-hash",
        type=str,
        default=None,
        help="Git hash to use for the evaluation",
    )

    parser.add_argument(
        "--policy-select-type",
        type=str,
        default="all",
        choices=["all", "latest", "top"],
        help="Policy selection type",
    )

    parser.add_argument(
        "--wandb-entity",
        type=str,
        default=METTA_WANDB_ENTITY,
        help="W&B entity",
    )

    parser.add_argument(
        "--wandb-project",
        type=str,
        default=METTA_WANDB_PROJECT,
        help="W&B project",
    )

    parser.add_argument(
        "--limit",
        type=int,
        default=1000,
        help="Maximum number of runs to process",
    )

    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be executed without actually running",
    )

    parser.add_argument(
        "--list-only",
        action="store_true",
        help="Only list matching runs without executing request_eval",
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=10,
        help="Number of runs to process in each batch (to avoid URL length limits)",
    )

    args, unknown_args = parser.parse_known_args()

    info(f"Searching for runs containing '{args.pattern}'...")
    matching_runs = find_runs_by_pattern(
        pattern=args.pattern,
        entity=args.wandb_entity,
        project=args.wandb_project,
        limit=args.limit,
    )

    info(f"Found {len(matching_runs)} matching runs")

    if args.list_only:
        for run_name in matching_runs:
            print(run_name)
        return

    if matching_runs:
        for i, run_name in enumerate(matching_runs, 1):
            info(f"{i:3d}. {run_name}")

    execute_request_eval(
        run_names=matching_runs,
        eval_name=args.eval,
        git_hash=args.git_hash,
        policy_select_type=args.policy_select_type,
        dry_run=args.dry_run,
        additional_args=unknown_args,
        batch_size=args.batch_size,
    )


if __name__ == "__main__":
    main()
