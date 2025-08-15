#!/usr/bin/env python3
"""
Post GitHub commit status from SkyPilot job.
Env:
  GITHUB_PAT                   (required) Personal Access Token with repo
  GITHUB_REPOSITORY            (required) e.g. "Metta-AI/metta"
  METTA_GIT_REF                (required) git SHA to update
  GITHUB_STATUS_STATE          (required) explicit state: success/failure/error/pending
  CMD_EXIT                     (optional) exit code to include
  METTA_RUN_ID                 (optional) used to build a link to wandb
  SKYPILOT_TASK_ID             (optional) used to suggest a log to review
  GITHUB_STATUS_CONTEXT        (optional) status context, default "Skypilot/E2E"
  GITHUB_STATUS_DESCRIPTION    (optional) custom description, default "Training completed successfully" etc
"""

from __future__ import annotations

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

from metta.common.util.github import post_commit_status  # noqa: E402


def main() -> int:
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

    state = os.getenv("GITHUB_STATUS_STATE", "").strip()
    if not state:
        print("[ERROR] GITHUB_STATUS_STATE is required")
        return 1

    context = os.getenv("GITHUB_STATUS_CONTEXT", "Skypilot/E2E").strip()
    if not context:
        print("[ERROR] post_commit_status requires a valid context string!")
        return 1

    failure_message = "Training failed!"
    try:
        cmd_exit = int(os.getenv("CMD_EXIT", "0"))
        failure_message += f" (exit code {cmd_exit})"
    except ValueError:
        cmd_exit = None

    desc = os.getenv(
        "GITHUB_STATUS_DESCRIPTION",
        "Training completed successfully" if state == "success" else failure_message,
    )

    task_id = os.getenv("SKYPILOT_TASK_ID", "").strip()
    if task_id:
        desc += f" - [ jl {task_id} ]"

    # The target_url is a URL that GitHub will associate with the commit status. When users view the commit status
    # on GitHub (for example, in pull requests or on the commit page), they can click on the status check and be
    # directed to this URL. We want to link to the expected wandb report based on the run name
    target_url = None
    wandb_run_id = os.getenv("METTA_RUN_ID") or None
    if wandb_run_id:
        target_url = f"https://wandb.ai/metta-research/metta/runs/{wandb_run_id}"
        print(f"[INFO] Target URL: {target_url}")

    # Light retry for transient errors
    for attempt in range(1, 5):
        try:
            post_commit_status(
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
