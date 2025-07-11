#!/usr/bin/env -S uv run
"""
Runs a single eval task inside a Docker container.

This script:
1. Receives eval task details via environment variables
2. Checks out the correct git hash
3. Runs sim.py with the appropriate parameters
4. Reports success/failure back
"""

import json
import os
import subprocess
import sys
import traceback

import requests


def checkout_git_hash(git_hash: str) -> None:
    """Checkout the specified git hash."""
    print(f"Checking out git hash: {git_hash}")
    result = subprocess.run(
        ["git", "checkout", git_hash],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(f"Failed to checkout git hash {git_hash}: {result.stderr}")
    print(f"Successfully checked out git hash: {git_hash}")


def run_sim_task(
    policy_id: str,
    sim_suite: str,
    eval_task_id: str,
    env_overrides: dict,
) -> None:
    """Run the simulation task using sim.py."""
    # Determine if policy_id is already a full URI or just an ID
    if policy_id.startswith("wandb://") or policy_id.startswith("file://"):
        policy_uri = policy_id
    else:
        # Assume it's a wandb artifact ID
        policy_uri = f"wandb://metta-research/metta/{policy_id}:latest"

    cmd = [
        "python",
        "tools/sim.py",
        f"policy_uri={policy_uri}",
        f"sim={sim_suite}",
        f"eval_task_id={eval_task_id}",
        f"run=eval_task_{eval_task_id[:8]}",
    ]

    # Add environment overrides
    for key, value in env_overrides.items():
        cmd.append(f"{key}={value}")

    print(f"Running command: {' '.join(cmd)}")

    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode != 0:
        raise RuntimeError(f"sim.py failed with exit code {result.returncode}:\n{result.stderr}")

    print("Simulation completed successfully")


def update_task_status(
    backend_url: str,
    eval_task_id: str,
    assignee: str,
    status: str,
    error_reason: str | None = None,
) -> None:
    """Update the task status in the backend."""
    url = f"{backend_url}/tasks/claimed/update"

    if error_reason:
        status_update = {"status": status, "error_reason": error_reason}
    else:
        status_update = status

    data = {"assignee": assignee, "statuses": {eval_task_id: status_update}}

    response = requests.post(url, json=data)
    response.raise_for_status()
    print(f"Updated task status to: {status}")


def main() -> None:
    # Get task details from environment
    backend_url = os.environ["BACKEND_URL"]
    eval_task_id = os.environ["EVAL_TASK_ID"]
    policy_id = os.environ["POLICY_ID"]
    sim_suite = os.environ["SIM_SUITE"]
    git_hash = os.environ["GIT_HASH"]
    assignee = os.environ["ASSIGNEE"]
    env_overrides = json.loads(os.environ.get("ENV_OVERRIDES", "{}"))

    print(f"Starting eval task runner for task {eval_task_id}")
    print(f"Policy: {policy_id}, Sim Suite: {sim_suite}, Git Hash: {git_hash}")

    try:
        # Checkout the correct git hash
        checkout_git_hash(git_hash)

        # Run the simulation
        run_sim_task(policy_id, sim_suite, eval_task_id, env_overrides)

        # Update status to done
        update_task_status(backend_url, eval_task_id, assignee, "done")

    except Exception as e:
        error_msg = f"{type(e).__name__}: {str(e)}\n{traceback.format_exc()}"
        print(f"Task failed: {error_msg}", file=sys.stderr)

        # Update status to error
        try:
            update_task_status(backend_url, eval_task_id, assignee, "error", error_msg)
        except Exception as update_error:
            print(f"Failed to update task status: {update_error}", file=sys.stderr)
            sys.exit(1)

        sys.exit(1)


if __name__ == "__main__":
    main()
