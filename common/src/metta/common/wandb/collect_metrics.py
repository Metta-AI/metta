#!/usr/bin/env -S uv run
# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "wandb>=0.16.0",
#     "requests>=2.31.0",
# ]
# ///

import os
import sys

import wandb

from metta.common.util.constants import METTA_WANDB_ENTITY, METTA_WANDB_PROJECT


def get_run_metrics(entity: str, project: str, run_id: str) -> set[str] | None:
    """Fetch all metrics from a wandb run."""
    # Initialize wandb API
    api = wandb.Api()

    try:
        run = api.run(f"{entity}/{project}/{run_id}")
    except Exception as e:
        print(f"Error accessing run: {e}")
        return None

    metrics = set()

    # Get metrics from summary
    summary = run.summary
    if summary:
        for key in summary.keys():
            if key not in ["_timestamp", "_runtime", "_step", "_wandb"]:
                metrics.add(key)

    # Get metrics from history sample
    try:
        history = run.history(samples=1)
        if history and len(history) > 0:
            sample = history[0]
            for key in sample.keys():
                if key not in ["_timestamp", "_runtime", "_step", "_wandb"]:
                    metrics.add(key)
    except Exception as e:
        print(f"Note: Could not fetch history sample: {e}")

    return metrics


def load_existing_metrics(filepath: str) -> set[str]:
    """Load existing metrics from CSV file."""
    metrics = set()

    if os.path.exists(filepath):
        print(f"Loading existing metrics from {filepath}...")
        with open(filepath, "r") as f:
            for line in f:
                line = line.strip()
                if line:
                    metrics.add(line)
        print(f"Loaded {len(metrics)} existing metrics")
    else:
        print(f"No existing file found at {filepath}, starting fresh")

    return metrics


def save_metrics_to_csv(metrics: set[str], filepath: str) -> None:
    """Save metrics to CSV file, one per line, alphabetically sorted."""
    sorted_metrics = sorted(metrics)

    with open(filepath, "w") as f:
        for metric in sorted_metrics:
            f.write(metric + "\n")

    print(f"Saved {len(sorted_metrics)} total metrics to {filepath}")


def main() -> None:
    # Defaults
    entity = METTA_WANDB_ENTITY
    project = METTA_WANDB_PROJECT
    default_run_names = [
        "github.sky.main.f634b79.20250701_203512",
        "daphne.moretime.nav_memory_sequence.navigation_finetuned.06-25",
        # Add more default run IDs here
    ]

    # Command line arguments
    if len(sys.argv) > 1:
        # Single run specified
        run_names = [sys.argv[1]]
        if len(sys.argv) > 2:
            project = sys.argv[2]
        if len(sys.argv) > 3:
            entity = sys.argv[3]
    else:
        # No run specified, use defaults
        run_names = default_run_names
        print(f"No run name specified, processing {len(run_names)} default runs")

    # Output file path
    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_file = os.path.join(script_dir, "wandb_metrics.csv")

    # Load existing metrics
    existing_metrics = load_existing_metrics(output_file)

    # Collect metrics from all runs
    all_new_metrics = set()

    for run_name in run_names:
        print(f"\nFetching metrics for run: {entity}/{project}/{run_name}")
        print("-" * 60)

        new_metrics = get_run_metrics(entity, project, run_name)

        if new_metrics is not None:
            run_unique = new_metrics - existing_metrics - all_new_metrics
            print(f"Found {len(new_metrics)} metrics in this run ({len(run_unique)} new)")
            all_new_metrics.update(new_metrics)
        else:
            print(f"Failed to fetch data for run {run_name}")

    if all_new_metrics:
        # Calculate total new metrics
        unique_new = all_new_metrics - existing_metrics
        print(f"\n{'=' * 60}")
        print(f"Total metrics across all runs: {len(all_new_metrics)}")
        print(f"New unique metrics: {len(unique_new)}")

        # Merge and save
        all_metrics = existing_metrics | all_new_metrics
        save_metrics_to_csv(all_metrics, output_file)

        # Show examples of new metrics
        if unique_new and len(unique_new) <= 10:
            print("\nNew metrics added:")
            for metric in sorted(unique_new)[:10]:
                print(f"  - {metric}")
        elif unique_new:
            print(f"\nShowing first 10 of {len(unique_new)} new metrics:")
            for metric in sorted(unique_new)[:10]:
                print(f"  - {metric}")
    else:
        print("\nNo metrics fetched from any runs.")


if __name__ == "__main__":
    main()
