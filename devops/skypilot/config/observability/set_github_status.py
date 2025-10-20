#!/usr/bin/env python3
"""
Post GitHub commit status from SkyPilot job.

Usage:
    ./set_github_status.py <state> <description>

Example:
    ./set_github_status.py pending "Queued on SkyPilot..."
    ./set_github_status.py success "Training completed successfully"

Positional arguments:
    state         explicit state: success/failure/error/pending
    description   status description

Environment variables:
    GITHUB_PAT                   (required) Personal Access Token with repo
    GITHUB_REPOSITORY            (required) e.g. "Metta-AI/metta"
    METTA_GIT_REF                (required) git SHA to update
    CMD_EXIT                     (optional) exit code to include
    METTA_RUN_ID                 (optional) used to build a link to wandb
    SKYPILOT_TASK_ID             (optional) used to suggest a log to review
    SKYPILOT_JOB_ID              (optional) SkyPilot job ID (read from file if not set)
    GITHUB_STATUS_CONTEXT        (optional) status context, default "Skypilot/E2E"
    IS_MASTER                    (optional) only run if true, default false
    ENABLE_GITHUB_STATUS         (optional) only run if true, default false
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

# Allow importing metta.common from repo if present (no install needed)
REPO_ROOT = Path(__file__).resolve().parents[2]
CANDIDATES = [REPO_ROOT / "common" / "src", Path("/workspace/metta/common/src")]
for p in CANDIDATES:
    if p.exists():
        sys.path.insert(0, str(p))
        break

import gitta as git  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Post GitHub commit status from SkyPilot job",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("state", help="Status state: success/failure/error/pending")
    parser.add_argument("description", help="Status description")

    args = parser.parse_args()

    # Check if we should run (from shell script logic)
    is_master = os.getenv("IS_MASTER", "false").lower() == "true"
    enable_github_status = os.getenv("ENABLE_GITHUB_STATUS", "false").lower() == "true"

    if not is_master or not enable_github_status:
        print(
            "[SKIP] GitHub status update skipped (IS_MASTER={}, ENABLE_GITHUB_STATUS={})".format(
                is_master, enable_github_status
            )
        )
        return 0

    # Read SkyPilot job ID from file if not in environment
    job_id = os.getenv("SKYPILOT_JOB_ID", "").strip()
    if not job_id and Path("/tmp/.sky_tmp/sky_job_id").exists():
        try:
            job_id = Path("/tmp/.sky_tmp/sky_job_id").read_text().strip()
            os.environ["SKYPILOT_JOB_ID"] = job_id
        except Exception as e:
            print(f"[WARN] Could not read SkyPilot job ID: {e}")

    # Get required environment variables
    commit_sha = os.getenv("METTA_GIT_REF", "").strip()
    if not commit_sha:
        print("[ERROR] METTA_GIT_REF is required")
        return 1

    repo = os.getenv("GITHUB_REPOSITORY", "").strip()
    if not repo:
        print("[ERROR] GITHUB_REPOSITORY is required (e.g. Metta-AI/metta)")
        return 1

    token = os.getenv("GITHUB_PAT", "").strip()
    if not token:
        print("[ERROR] GITHUB_PAT is required")
        return 1

    context = os.getenv("GITHUB_STATUS_CONTEXT", "Skypilot/E2E").strip()
    if not context:
        print("[ERROR] post_commit_status requires a valid context string!")
        return 1

    # Use provided arguments
    state = args.state
    desc = args.description

    # Add exit code to description if provided
    try:
        cmd_exit = int(os.getenv("CMD_EXIT", "0"))
        if cmd_exit != 0 and state in ["failure", "error"]:
            desc += f" (exit code {cmd_exit})"
    except ValueError:
        pass

    # Add job ID to description if available
    if job_id:
        print(f"[INFO] Setting GitHub status for job {job_id}")
        desc += f" - [ jl {job_id} ]"
    else:
        print("[INFO] No SkyPilot job ID found")

    # The target_url is a URL that GitHub will associate with the commit status
    target_url = None
    wandb_run_id = os.getenv("METTA_RUN_ID") or None
    if wandb_run_id:
        target_url = f"https://wandb.ai/metta-research/metta/runs/{wandb_run_id}"
        print(f"[INFO] Target URL: {target_url}")

    print(f"[RUN] Setting GitHub status: {state} - {desc}")

    # Light retry for transient errors
    for attempt in range(1, 5):
        try:
            git.post_commit_status(
                commit_sha=commit_sha,
                state=state,
                repo=repo,
                context=context,
                description=desc,
                target_url=target_url,
                token=token,
            )
            print(f"[OK] {repo}@{commit_sha[:8]} -> {state} ({context})")
            return 0
        except Exception as e:
            if attempt == 4:
                print(f"[ERROR] Failed to post status after retries: {e}")
                return 2

            sleep_s = 2**attempt
            print(f"[WARN] Post failed (attempt {attempt}), retrying in {sleep_s}s: {e}")
            time.sleep(sleep_s)

    return 2


if __name__ == "__main__":
    sys.exit(main())
