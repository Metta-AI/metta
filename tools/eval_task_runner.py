#!/usr/bin/env -S uv run
"""
Runs eval tasks inside a Docker container.

This script has two modes:
1. Single task mode (legacy): Runs one task from environment variables
2. Worker mode: Continuously polls for tasks matching its git hash

In worker mode:
- Checks out the specified git hash once at startup
- Polls the backend for tasks assigned to this worker
- Processes tasks one at a time
- Reports success/failure back
"""

import argparse
import json
import os
import subprocess
import sys
import time
import traceback

import httpx
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


class EvalWorker:
    """Worker that continuously processes tasks for a specific git hash."""

    def __init__(self, backend_url: str, git_hash: str, assignee: str):
        self.backend_url = backend_url
        self.git_hash = git_hash
        self.assignee = assignee
        self.client = httpx.Client(timeout=30.0)

    def get_claimed_tasks(self) -> list:
        """Get tasks claimed by this worker."""
        try:
            response = self.client.get(f"{self.backend_url}/tasks/claimed", params={"assignee": self.assignee})
            if response.status_code == 200:
                return response.json()
            return []
        except Exception as e:
            print(f"Failed to get claimed tasks: {e}")
            return []

    def claim_available_task(self) -> dict | None:
        """Try to claim an available task matching our git hash."""
        try:
            # Get available tasks
            response = self.client.get(f"{self.backend_url}/tasks/available", params={"limit": 10})
            if response.status_code != 200:
                return None

            tasks = response.json()["tasks"]

            # Filter for our git hash
            matching_tasks = [t for t in tasks if t["attributes"]["git_hash"] == self.git_hash]

            if not matching_tasks:
                return None

            # Try to claim the first one
            task = matching_tasks[0]
            response = self.client.post(
                f"{self.backend_url}/tasks/claim", json={"eval_task_ids": [task["id"]], "assignee": self.assignee}
            )

            if response.status_code == 200 and response.json():
                return task

            return None

        except Exception as e:
            print(f"Failed to claim task: {e}")
            return None

    def run(self) -> None:
        """Main worker loop."""
        print(f"Starting eval worker for git hash {self.git_hash}")
        print(f"Backend URL: {self.backend_url}")
        print(f"Assignee: {self.assignee}")

        # Checkout git hash once at startup
        try:
            checkout_git_hash(self.git_hash)
        except Exception as e:
            print(f"Failed to checkout git hash: {e}", file=sys.stderr)
            sys.exit(1)

        while True:
            try:
                # Check if we have a claimed task
                claimed_tasks = self.get_claimed_tasks()

                if claimed_tasks:
                    # Process first claimed task
                    task = claimed_tasks[0]
                    print(f"Processing task {task['id']}")

                    try:
                        # Run the simulation
                        run_sim_task(
                            task["policy_id"],
                            task["sim_suite"],
                            task["id"],
                            task["attributes"].get("env_overrides", {}),
                        )

                        # Update status to done
                        update_task_status(self.backend_url, task["id"], self.assignee, "done")

                    except Exception as e:
                        error_msg = f"{type(e).__name__}: {str(e)}\n{traceback.format_exc()}"
                        print(f"Task failed: {error_msg}", file=sys.stderr)

                        # Update status to error
                        update_task_status(self.backend_url, task["id"], self.assignee, "error", error_msg)
                else:
                    # Try to claim a new task
                    task = self.claim_available_task()
                    if task:
                        print(f"Claimed task {task['id']}")
                        continue  # Process it in next iteration

                # Sleep before next poll
                time.sleep(5)

            except KeyboardInterrupt:
                print("Worker interrupted")
                break
            except Exception as e:
                print(f"Error in worker loop: {e}", file=sys.stderr)
                time.sleep(10)  # Back off on errors


def main() -> None:
    parser = argparse.ArgumentParser(description="Eval task runner")
    parser.add_argument("--worker-mode", action="store_true", help="Run in worker mode")
    args = parser.parse_args()

    if args.worker_mode:
        # Worker mode - continuous processing
        backend_url = os.environ["BACKEND_URL"]
        git_hash = os.environ["GIT_HASH"]
        assignee = os.environ["WORKER_ASSIGNEE"]

        worker = EvalWorker(backend_url, git_hash, assignee)
        worker.run()
    else:
        # Legacy single task mode
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
