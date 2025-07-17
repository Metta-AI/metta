#!/usr/bin/env -S uv run
# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "wandb>=0.20.1",
# ]
# ///

"""
Debug script to see what curriculum paths are actually stored in wandb runs.
"""

import argparse
import sys
from collections import Counter
from typing import Optional

import wandb


def debug_curriculum_paths(
    entity: str = "metta-research",
    project: str = "metta",
    user_filter: Optional[str] = None,
    limit: int = 200,
    state_filter: Optional[str] = None,
) -> None:
    """Debug what curriculum paths are actually found in runs."""

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

        curriculum_paths = []
        all_config_keys = set()
        trainer_config_keys = set()
        processed_count = 0

        for run in runs:
            processed_count += 1
            if processed_count % 20 == 0:
                print(f"Processed {processed_count} runs...", file=sys.stderr)

            config = run.config

            # Collect all config keys for debugging
            if config:
                all_config_keys.update(config.keys())
                if "trainer" in config and isinstance(config["trainer"], dict):
                    trainer_config_keys.update(config["trainer"].keys())

            # Check various paths where curriculum might be stored
            curriculum_path = None
            curriculum_source = None

            if config:
                # Check trainer.curriculum (most common)
                if "trainer" in config and isinstance(config["trainer"], dict) and "curriculum" in config["trainer"]:
                    curriculum_path = config["trainer"]["curriculum"]
                    curriculum_source = "trainer.curriculum"
                # Check direct curriculum key
                elif "curriculum" in config:
                    curriculum_path = config["curriculum"]
                    curriculum_source = "curriculum"
                # Check other possible locations
                elif (
                    "train_job" in config
                    and isinstance(config["train_job"], dict)
                    and "curriculum" in config["train_job"]
                ):
                    curriculum_path = config["train_job"]["curriculum"]
                    curriculum_source = "train_job.curriculum"

            if curriculum_path:
                curriculum_paths.append(
                    {
                        "run_name": run.name,
                        "curriculum_path": curriculum_path,
                        "curriculum_source": curriculum_source,
                        "state": run.state,
                        "created_at": run.created_at,
                        "user": run.user.name if run.user else "unknown",
                    }
                )

    except Exception as e:
        print(f"Error querying wandb: {e}", file=sys.stderr)
        return

    print(f"\nProcessed {processed_count} runs total", file=sys.stderr)
    print(f"Found {len(curriculum_paths)} runs with curriculum configs", file=sys.stderr)
    print("=" * 80)

    # Show curriculum path frequency
    path_counter = Counter(item["curriculum_path"] for item in curriculum_paths)
    print("\nCURRICULUM PATH FREQUENCY:")
    print("-" * 50)
    for path, count in path_counter.most_common():
        print(f"{count:3d}: {path}")

    # Show config source frequency
    source_counter = Counter(item["curriculum_source"] for item in curriculum_paths)
    print("\nCURRICULUM SOURCE LOCATION:")
    print("-" * 50)
    for source, count in source_counter.most_common():
        print(f"{count:3d}: {source}")

    # Show all runs with curriculum paths
    print("\nALL RUNS WITH CURRICULUM CONFIGS:")
    print("-" * 50)
    for item in sorted(curriculum_paths, key=lambda x: x["created_at"], reverse=True):
        print(f"{item['run_name']} ({item['user']}) -> {item['curriculum_path']}")

    # Debug: Show all config keys found
    print("\nDEBUG: ALL CONFIG KEYS FOUND:")
    print("-" * 50)
    for key in sorted(all_config_keys):
        print(f"  {key}")

    print("\nDEBUG: TRAINER CONFIG KEYS FOUND:")
    print("-" * 50)
    for key in sorted(trainer_config_keys):
        print(f"  trainer.{key}")


def main():
    parser = argparse.ArgumentParser(description="Debug curriculum paths in wandb runs")
    parser.add_argument("--user", help="Filter runs by username")
    parser.add_argument("--entity", default="metta-research", help="wandb entity")
    parser.add_argument("--project", default="metta", help="wandb project")
    parser.add_argument("--limit", type=int, default=200, help="Maximum runs to fetch")
    parser.add_argument("--state", help="Filter by run state (finished, crashed, failed, etc.)")

    args = parser.parse_args()

    debug_curriculum_paths(
        entity=args.entity, project=args.project, user_filter=args.user, limit=args.limit, state_filter=args.state
    )


if __name__ == "__main__":
    main()
