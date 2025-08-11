#!/usr/bin/env python3
"""
Post GitHub commit status from SkyPilot job.
Env:
  METTA_GIT_REF     (required) git SHA to update
  CMD_EXIT          (optional) exit code; 0 => success, else failure
  STATE             (optional) explicit state: success/failure/error/pending
  SKYPILOT_TASK_ID  (optional) used to build a console link
  TARGET_URL        (optional) preferred link to show in GitHub
  GITHUB_PAT        (required) Personal Access Token with repo
  GITHUB_REPOSITORY (required) e.g. "Metta-AI/metta"
  STATUS_CONTEXT    (optional) status context, default "Skypilot/E2E"
  DESCRIPTION       (optional) custom description
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
        print("Error: METTA_GIT_REF is required", file=sys.stderr)
        return 1

    repo = os.getenv("GITHUB_REPOSITORY", "").strip()
    if not repo:
        print("Error: GITHUB_REPOSITORY is required (e.g. Metta-AI/metta)", file=sys.stderr)
        return 1

    token = os.getenv("GITHUB_PAT", "").strip()
    if not token:
        print("Error: GITHUB_PAT is required", file=sys.stderr)
        return 1

    try:
        cmd_exit = int(os.getenv("CMD_EXIT", "0"))
    except ValueError:
        cmd_exit = 1

    # Check for explicit STATE override (e.g., for pending status)
    state = os.getenv("STATE", "").strip()
    if not state:
        # If no explicit state, determine from exit code
        state = "success" if cmd_exit == 0 else "failure"

    context = os.getenv("STATUS_CONTEXT", "Skypilot/E2E")
    desc = os.getenv(
        "DESCRIPTION",
        "Training completed successfully" if state == "success" else f"Training failed (exit {cmd_exit})",
    )

    # Prefer explicit TARGET_URL; else try SkyPilot console; else None
    target_url = os.getenv("TARGET_URL") or None
    if not target_url:
        task_id = os.getenv("SKYPILOT_TASK_ID")
        if task_id:
            target_url = f"https://console.skypilot.co/jobs/{task_id}"

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
                print(f"[ERR] Failed to post status after retries: {e}", file=sys.stderr)
                return 2
            sleep_s = 2**attempt
            print(f"[WARN] Post failed (attempt {attempt}), retrying in {sleep_s}s: {e}", file=sys.stderr)
            time.sleep(sleep_s)

    return 2


if __name__ == "__main__":
    sys.exit(main())
