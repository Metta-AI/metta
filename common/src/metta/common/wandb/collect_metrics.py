#!/usr/bin/env -S uv run
# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "wandb>=0.16.0",
#     "requests>=2.31.0",
#     "tabulate>=0.9.0",
# ]
# ///

import os
import sys
from collections import defaultdict

import wandb
from tabulate import tabulate


def get_run_sections(entity: str, project: str, run_id: str):
    """
    Fetch all metric sections from a wandb run.

    Args:
        entity: WandB entity (username or team name)
        project: WandB project name
        run_id: WandB run ID

    Returns:
        dict: Dictionary mapping section names to list of metrics
    """
    # Initialize wandb API
    api = wandb.Api()

    # Get the run
    try:
        run = api.run(f"{entity}/{project}/{run_id}")
    except Exception as e:
        print(f"Error accessing run: {e}")
        return None

    # Get all logged metrics from summary (more reliable than history without pandas)
    summary = run.summary

    # Organize metrics by section
    sections = defaultdict(list)

    # Process summary metrics
    if summary:
        for key in summary.keys():
            if key not in ["_timestamp", "_runtime", "_step", "_wandb"]:
                # Split by '/' to get section hierarchy
                parts = key.split("/")
                if len(parts) > 1:
                    section = parts[0]
                    sections[section].append(key)
                else:
                    sections["_root"].append(key)

    # Also check history keys (without pandas)
    try:
        # Get a sample from history to find all logged keys
        history = run.history(samples=1)
        if history and len(history) > 0:
            # history is a list of dicts when pandas is not available
            sample = history[0]
            for key in sample.keys():
                if key not in ["_timestamp", "_runtime", "_step", "_wandb"]:
                    # Split by '/' to get section hierarchy
                    parts = key.split("/")
                    if len(parts) > 1:
                        section = parts[0]
                        if key not in sections[section]:
                            sections[section].append(key)
                    else:
                        if key not in sections["_root"]:
                            sections["_root"].append(key)
    except Exception as e:
        print(f"Note: Could not fetch history sample: {e}")

    return dict(sections)


def organize_metrics_hierarchically(all_metrics):
    """Organize metrics into a hierarchical structure."""
    hierarchy = {}

    for metric in all_metrics:
        parts = metric.split("/")
        current = hierarchy

        for i, part in enumerate(parts):
            if i == len(parts) - 1:
                # This is the leaf metric
                if "_metrics" not in current:
                    current["_metrics"] = []
                current["_metrics"].append(metric)
            else:
                # This is a section
                if part not in current:
                    current[part] = {}
                current = current[part]

    return hierarchy


def count_metrics_in_hierarchy(node):
    """Count total metrics in a hierarchical node."""
    count = 0
    if "_metrics" in node:
        count += len(node["_metrics"])

    for key, value in node.items():
        if key != "_metrics" and isinstance(value, dict):
            count += count_metrics_in_hierarchy(value)

    return count


def print_hierarchy(node, prefix="", is_last=True, show_full_paths=False):
    """Print the hierarchical structure in a tree format."""
    items = [(k, v) for k, v in sorted(node.items()) if k != "_metrics"]

    for i, (key, value) in enumerate(items):
        is_last_item = i == len(items) - 1

        # Print the branch
        print(prefix + ("└── " if is_last_item else "├── ") + key, end="")

        # Count metrics in this subtree
        if isinstance(value, dict):
            metric_count = count_metrics_in_hierarchy(value)
            print(f" ({metric_count} metrics)")

            # Recursively print children
            extension = "    " if is_last_item else "│   "
            print_hierarchy(value, prefix + extension, is_last_item, show_full_paths)
        else:
            print()

    # Print leaf metrics if requested
    if show_full_paths and "_metrics" in node:
        for i, metric in enumerate(sorted(node["_metrics"])):
            is_last_metric = i == len(node["_metrics"]) - 1
            print(prefix + ("└── " if is_last_metric and not items else "├── ") + f"[{metric}]")


def print_sections_table(sections: dict):
    """Print sections and metrics in a nice table format."""
    if not sections:
        print("No sections found!")
        return

    # Collect all metrics
    all_metrics = []
    for metrics in sections.values():
        all_metrics.extend(metrics)

    # Organize hierarchically
    hierarchy = organize_metrics_hierarchically(all_metrics)

    print("\nMetric Hierarchy:")
    print("=" * 60)
    print_hierarchy(hierarchy)

    # Print summary by top-level section
    print("\nSummary by Top-Level Section:")
    print("=" * 60)
    table_data = []

    for section, metrics in sorted(sections.items()):
        if section != "_root":
            table_data.append([section, len(metrics)])

    if sections.get("_root"):
        table_data.append(["[root level]", len(sections["_root"])])

    print(tabulate(table_data, headers=["Section", "Metric Count"], tablefmt="grid"))

    # Print overall summary
    total_metrics = sum(len(metrics) for metrics in sections.values())
    print(f"\nTotal top-level sections: {len(sections)}")
    print(f"Total metrics: {total_metrics}")

    # Find metrics with most nesting
    max_depth = 0
    deepest_metrics = []
    for metric in all_metrics:
        depth = metric.count("/")
        if depth > max_depth:
            max_depth = depth
            deepest_metrics = [metric]
        elif depth == max_depth:
            deepest_metrics.append(metric)

    print(f"Maximum nesting depth: {max_depth}")
    if deepest_metrics and len(deepest_metrics) <= 5:
        print("Deepest metrics:")
        for metric in deepest_metrics[:5]:
            print(f"  - {metric}")


def main():
    # You can modify these defaults or pass them as arguments
    entity = "metta-research"  # Based on your wandb interface
    project = "metta"  # Based on your wandb interface

    # Example run ID from your data
    run_id = "github.sky.main.93e1165.20250630_183832"

    # Allow command line override
    if len(sys.argv) > 1:
        run_id = sys.argv[1]
    if len(sys.argv) > 2:
        project = sys.argv[2]
    if len(sys.argv) > 3:
        entity = sys.argv[3]

    print(f"Fetching sections for run: {entity}/{project}/{run_id}")
    print("=" * 60)

    sections = get_run_sections(entity, project, run_id)

    if sections:
        print_sections_table(sections)

        # Save full metric list in the same directory as the script
        script_dir = os.path.dirname(os.path.abspath(__file__))
        output_file = os.path.join(script_dir, "wandb_metrics.txt")

        print(f"\nSaving full metric list to '{output_file}'...")
        all_metrics = []
        for metrics in sections.values():
            all_metrics.extend(metrics)

        with open(output_file, "w") as f:
            f.write(f"WandB Metrics for {entity}/{project}/{run_id}\n")
            f.write("=" * 60 + "\n\n")
            for metric in sorted(all_metrics):
                f.write(metric + "\n")
            f.write(f"\nTotal metrics: {len(all_metrics)}\n")

        print(f"Saved {len(all_metrics)} metrics to {output_file}")
    else:
        print("Failed to fetch run data.")


if __name__ == "__main__":
    main()
