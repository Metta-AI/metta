#!/usr/bin/env -S uv run
# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "wandb>=0.20.1",
# ]
# ///
"""
Validate that a W&B run exists for a given policy name.
Exit codes:
  0 - Run found
  1 - Run not found or error
"""

import os
import sys

import wandb


def main() -> None:
    """Check if a W&B run exists for the given policy."""
    # Get environment variables
    api_key = os.environ.get("WANDB_API_KEY")
    policy = os.environ.get("POLICY")
    entity = os.environ.get("WANDB_ENTITY", "metta-research")
    project = os.environ.get("WANDB_PROJECT", "metta")

    # Validate inputs
    if not api_key:
        print("Error: WANDB_API_KEY environment variable not set")
        sys.exit(1)

    if not policy:
        print("Error: POLICY environment variable not set")
        sys.exit(1)

    print(f"Checking for run '{policy}' in {entity}/{project}...")

    try:
        # Initialize W&B API
        api = wandb.Api(api_key=api_key)

        # Search for runs with matching display name
        runs = api.runs(path=f"{entity}/{project}", filters={"display_name": policy})

        # Check if any runs were found
        run_list = list(runs)

        if run_list:
            print(f"✓ Run found: {policy}")
            print(f"  Run ID: {run_list[0].id}")
            print(f"  State: {run_list[0].state}")
            print(f"  Created at: {run_list[0].created_at}")
            sys.exit(0)
        else:
            print(f"✗ Run not found: {policy}")
            sys.exit(1)

    except Exception as e:
        print(f"Error accessing W&B API: {e}")
        print(f"Entity: {entity}")
        print(f"Project: {project}")
        sys.exit(1)


if __name__ == "__main__":
    main()
