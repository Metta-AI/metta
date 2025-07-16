#!/usr/bin/env -S uv run
"""
Explore available runs and metrics in WandB.

This script helps you discover what runs and metrics are available
before running the full analysis.
"""

import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")

from metta_metrics_analysis.wandb_data_collector import WandBDataCollector


def main():
    """Explore available runs and metrics."""

    # Initialize collector
    print("=== Initializing WandB Data Collector ===")
    collector = WandBDataCollector(
        entity="metta-research",
        project="metta",
    )

    # List recent runs
    print("\n=== Recent Runs ===")
    # Filter for runs with "jacke" in the name
    runs_df = collector.list_runs(filters={"display_name": {"$regex": "jacke"}}, limit=20)

    if len(runs_df) == 0:
        print("No runs found!")
        return

    print(f"\nFound {len(runs_df)} recent runs:")
    print(runs_df[["run_name", "group", "state", "duration", "step"]].to_string())

    # Get metrics from runs that have data
    print("\n=== Available Metrics ===")
    found_metrics = False
    for idx, row in runs_df.iterrows():
        run_id = row["run_id"]
        run_name = row["run_name"]
        step_count = row["step"]

        # Skip runs with no steps
        if step_count == 0:
            continue

        print(f"\nChecking metrics for run: {run_name} ({run_id}) - {step_count} steps")

        metrics = collector.get_available_metrics(run_id)
        if metrics:
            found_metrics = True
            print(f"Found {len(metrics)} metrics:")
            # Group metrics by prefix
            metric_groups = {}
            for metric in sorted(metrics):
                prefix = metric.split("/")[0] if "/" in metric else "other"
                if prefix not in metric_groups:
                    metric_groups[prefix] = []
                metric_groups[prefix].append(metric)

            for prefix, group_metrics in sorted(metric_groups.items()):
                print(f"\n{prefix}/ ({len(group_metrics)} metrics):")
                for metric in group_metrics[:5]:  # Show first 5
                    print(f"  - {metric}")
                if len(group_metrics) > 5:
                    print(f"  ... and {len(group_metrics) - 5} more")
            break  # Found metrics, stop searching

    if not found_metrics:
        print("\nNo metrics found in recent runs. The runs may be empty or still initializing.")

    # Suggest next steps
    print("\n=== Next Steps ===")
    print("\n1. Choose metrics from the list above for your analysis")
    print("2. Use run groups or tags to filter specific experiments")
    print("3. Run the basic_analysis.py script with your chosen metrics")
    print("\nExample filter options:")
    print("  - run_filter={'group': 'your_sweep_name'}")
    print("  - run_filter={'tags': {'$in': ['your_tag']}}")
    print("  - run_filter={'state': 'finished'}")


if __name__ == "__main__":
    main()
