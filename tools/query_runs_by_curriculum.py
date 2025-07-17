#!/usr/bin/env -S uv run
# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "wandb>=0.20.1",
# ]
# ///

"""
Query wandb for runs by curriculum configuration and output run names.
Dynamically discovers curriculum paths instead of using hardcoded mappings.
"""

import argparse
import json
import sys
from collections import defaultdict
from typing import Dict, List, Optional

import wandb


def get_curriculum_short_name(curriculum_path: str) -> str:
    """
    Generate a short name for a curriculum path.

    Args:
        curriculum_path: Full curriculum path

    Returns:
        Short name for the curriculum
    """
    # Remove common prefixes
    path = curriculum_path.replace("/env/mettagrid/curriculum/", "")
    path = path.replace("/env/mettagrid/arena/", "arena_")
    path = path.replace("env/mettagrid/curriculum/", "")
    path = path.replace("env/mettagrid/arena/", "arena_")

    # Replace slashes with underscores
    path = path.replace("/", "_")

    return path


def query_runs_by_curriculum(
    entity: str = "metta-research",
    project: str = "metta",
    user_filter: Optional[str] = None,
    limit: int = 200,
    state_filter: Optional[str] = None,
) -> Dict[str, List[str]]:
    """
    Query wandb for runs grouped by curriculum type.

    Returns:
        Dictionary mapping curriculum short names to lists of run names
    """
    print(f"Querying runs from {entity}/{project}...", file=sys.stderr)

    try:
        api = wandb.Api()

        # Build filters
        filters = {}
        if state_filter:
            filters["state"] = state_filter
        if user_filter:
            filters["display_name"] = {"$regex": f".*{user_filter}.*"}

        # Get runs
        runs = api.runs(f"{entity}/{project}", filters=filters, per_page=limit)

        curriculum_runs = defaultdict(list)
        processed_count = 0

        for run in runs:
            processed_count += 1
            if processed_count % 20 == 0:
                print(f"Processed {processed_count} runs...", file=sys.stderr)

            # Check if run has curriculum config
            config = run.config
            curriculum_path = None

            if config:
                # Check various paths where curriculum might be stored
                if "trainer" in config and isinstance(config["trainer"], dict) and "curriculum" in config["trainer"]:
                    curriculum_path = config["trainer"]["curriculum"]
                elif "curriculum" in config:
                    curriculum_path = config["curriculum"]
                elif (
                    "train_job" in config
                    and isinstance(config["train_job"], dict)
                    and "curriculum" in config["train_job"]
                ):
                    curriculum_path = config["train_job"]["curriculum"]

            if curriculum_path:
                curriculum_short_name = get_curriculum_short_name(curriculum_path)
                curriculum_runs[curriculum_short_name].append(run.name)

    except Exception as e:
        print(f"Error querying wandb: {e}", file=sys.stderr)
        return {}

    # Convert defaultdict to regular dict and sort runs within each curriculum
    result = {}
    for curriculum_name, runs in curriculum_runs.items():
        result[curriculum_name] = sorted(runs)

    return result


def main():
    parser = argparse.ArgumentParser(
        description="Query wandb for runs by curriculum configuration",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # List all runs by curriculum
  %(prog)s --user jacke

  # Get runs for specific curriculum only
  %(prog)s --user jacke --curriculum simple

  # Output as shell commands
  %(prog)s --user jacke --format shell

  # Output as JSON
  %(prog)s --user jacke --format json

  # Show curriculum paths and their short names
  %(prog)s --user jacke --show-paths
        """,
    )

    parser.add_argument("--user", help="Filter runs by username (e.g., jacke)")
    parser.add_argument("--curriculum", help="Only show runs with this curriculum type (use short name)")
    parser.add_argument("--entity", default="metta-research", help="wandb entity")
    parser.add_argument("--project", default="metta", help="wandb project")
    parser.add_argument("--limit", type=int, default=200, help="Maximum runs to fetch")
    parser.add_argument("--state", help="Filter by run state (finished, crashed, failed, etc.)")
    parser.add_argument("--format", choices=["table", "shell", "json"], default="table", help="Output format")
    parser.add_argument("--max-per-curriculum", type=int, help="Max runs to show per curriculum")
    parser.add_argument("--show-paths", action="store_true", help="Show full curriculum paths")

    args = parser.parse_args()

    # Query runs
    curriculum_runs = query_runs_by_curriculum(
        entity=args.entity, project=args.project, user_filter=args.user, limit=args.limit, state_filter=args.state
    )

    # If showing paths, we need to get the full paths
    if args.show_paths:
        print("Curriculum short names and their full paths:")
        print("=" * 60)

        # We need to query again to get the full paths
        api = wandb.Api()
        filters = {"state": args.state}
        if args.user:
            filters["display_name"] = {"$regex": f".*{args.user}.*"}

        runs = api.runs(f"{args.entity}/{args.project}", filters=filters, per_page=args.limit)

        curriculum_paths = {}
        for run in runs:
            config = run.config
            if config:
                curriculum_path = None
                if "trainer" in config and isinstance(config["trainer"], dict) and "curriculum" in config["trainer"]:
                    curriculum_path = config["trainer"]["curriculum"]
                elif "curriculum" in config:
                    curriculum_path = config["curriculum"]
                elif (
                    "train_job" in config
                    and isinstance(config["train_job"], dict)
                    and "curriculum" in config["train_job"]
                ):
                    curriculum_path = config["train_job"]["curriculum"]

                if curriculum_path:
                    curriculum_short_name = get_curriculum_short_name(curriculum_path)
                    curriculum_paths[curriculum_short_name] = curriculum_path

        for short_name, full_path in sorted(curriculum_paths.items()):
            print(f"{short_name:<30} -> {full_path}")

        return

    # Filter by curriculum if specified
    if args.curriculum:
        if args.curriculum not in curriculum_runs:
            print(f"No runs found for curriculum: {args.curriculum}", file=sys.stderr)
            available = ", ".join(sorted(curriculum_runs.keys()))
            print(f"Available curricula: {available}", file=sys.stderr)
            return
        curriculum_runs = {args.curriculum: curriculum_runs[args.curriculum]}

    # Limit runs per curriculum if specified
    if args.max_per_curriculum:
        for curriculum_name in curriculum_runs:
            curriculum_runs[curriculum_name] = curriculum_runs[curriculum_name][: args.max_per_curriculum]

    # Output results
    if args.format == "json":
        print(json.dumps(curriculum_runs, indent=2))

    elif args.format == "shell":
        print("#!/bin/bash")
        print("# Navigation evaluation commands grouped by curriculum")
        print()

        for curriculum_name, runs in curriculum_runs.items():
            print(f"# {curriculum_name.upper()} CURRICULUM")
            for run_name in runs:
                eval_run_name = f"{curriculum_name}_eval_{run_name}"
                cmd = f"""./tools/sim.py \\
    sim=navigation \\
    run={eval_run_name} \\
    policy_uri=wandb://run/{run_name} \\
    sim_job.stats_db_uri=wandb://stats/navigation_db \\
    device=cpu"""
                print(cmd)
                print()
            print()

    else:  # table format
        if not curriculum_runs:
            print("No runs found with curriculum configurations.")
            return

        print("Found runs by curriculum:")
        print("=" * 60)

        for curriculum_name, runs in sorted(curriculum_runs.items()):
            print(f"\n{curriculum_name.upper()} ({len(runs)} runs):")
            print("-" * 40)
            for run_name in runs:
                print(f"  wandb://run/{run_name}")

        print("\nTo run evaluations, use:")
        print("./tools/auto_eval_by_curriculum.py --user", args.user or "USERNAME")
        print("\nTo see curriculum paths:")
        print("./tools/query_runs_by_curriculum.py --user", args.user or "USERNAME", "--show-paths")


if __name__ == "__main__":
    main()
