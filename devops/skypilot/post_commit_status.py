#!/usr/bin/env python3
"""
Post GitHub commit status from SkyPilot job.

This script is called from the SkyPilot run script and reads from environment variables:
    COMMIT_SHA: Git commit SHA
    CMD_EXIT: Command exit code (0 = success, non-zero = failure)
    SKYPILOT_TASK_ID: SkyPilot task ID for the URL
    GITHUB_TOKEN: GitHub API token
"""

import os
import sys

# Add metta to Python path
sys.path.insert(0, "/workspace/metta")

from metta.common.util.github import post_commit_status


def main():
    # Get environment variables
    commit_sha = os.environ.get("COMMIT_SHA")
    cmd_exit = int(os.environ.get("CMD_EXIT", "0"))
    skypilot_task_id = os.environ.get("SKYPILOT_TASK_ID")

    if not commit_sha:
        print("Error: COMMIT_SHA environment variable not set", file=sys.stderr)
        return 1

    # Determine state and description based on exit code
    if cmd_exit == 0:
        state = "success"
        description = "Training completed successfully"
    else:
        state = "failure"
        description = f"Training failed with exit code {cmd_exit}"

    # Build target URL if we have a task ID
    target_url = None
    if skypilot_task_id:
        target_url = f"https://console.skypilot.co/jobs/{skypilot_task_id}"

    try:
        _result = post_commit_status(commit_sha=commit_sha, state=state, description=description, target_url=target_url)
        print(f"Successfully posted {state} status for commit {commit_sha[:8]}")
        return 0
    except ValueError as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1
    except Exception as e:
        print(f"Failed to post status: {e}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
