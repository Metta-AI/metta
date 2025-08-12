#!/usr/bin/env python3
"""
Post GitHub commit status from SkyPilot job.
Env:
  METTA_GIT_REF     (required) git SHA to update
  SKYPILOT_TASK_ID  (optional) used to build a console link
  GITHUB_PAT        (required) Personal Access Token with repo
  GITHUB_REPOSITORY (required) e.g. "Metta-AI/metta"
  CMD_EXIT          (optional) exit code; 0 => success, else failure

  Configuration for the status report:

  METTA_RUN_ID                 (optional) used to build a link to wandb
  GITHUB_STATUS_STATE          (optional) explicit state: success/failure/error/pending (overrides CMD_EXIT)
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

from metta.common.util.git import GitError, get_matched_pr  # noqa: E402
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

    print(f"[DEBUG] Repository: {repo}")
    print(f"[DEBUG] Commit SHA: {commit_sha}")
    print(f"[DEBUG] SHA length: {len(commit_sha)} chars")

    # Use get_matched_pr to verify commit exists on GitHub
    # This will return None if commit gets 404 (doesn't exist in repo)
    try:
        pr_info = get_matched_pr(commit_sha)
        if pr_info:
            pr_num, pr_title = pr_info
            print(f"[INFO] Commit {commit_sha[:8]} found, associated with PR #{pr_num}: {pr_title}")
        else:
            print(f"[INFO] Commit {commit_sha[:8]} found (no associated PR)")
    except GitError as e:
        # GitError is raised for network issues or if the commit doesn't exist
        error_msg = str(e)
        if "404" in error_msg or "not found" in error_msg.lower():
            print(f"ERROR: Commit {commit_sha} not found in {repo}!", file=sys.stderr)
            print("\nPossible causes:", file=sys.stderr)
            print("- Commit exists in a fork, not the main repository", file=sys.stderr)
            print("- Commit hasn't been pushed to GitHub", file=sys.stderr)
            print("- GITHUB_REPOSITORY points to wrong repo", file=sys.stderr)
            return 1
        else:
            # Network or other error - log but continue
            print(f"[WARN] Could not check PR status: {e}")

    try:
        cmd_exit = int(os.getenv("CMD_EXIT", "0"))
    except ValueError:
        cmd_exit = 1

    # Check for explicit GITHUB_STATUS_STATE override (e.g., for pending status)
    state = os.getenv("GITHUB_STATUS_STATE", "").strip()
    if not state:
        # If no explicit state, determine from exit code
        state = "success" if cmd_exit == 0 else "failure"

    context = os.getenv("GITHUB_STATUS_CONTEXT", "Skypilot/E2E")
    desc = os.getenv(
        "GITHUB_STATUS_DESCRIPTION",
        "Training completed successfully" if state == "success" else f"Training failed (exit {cmd_exit})",
    )

    print(f"[DEBUG] State: '{state}'")
    print(f"[DEBUG] Context: '{context}'")
    print(f"[DEBUG] Description: '{desc}'")
    print(f"[DEBUG] Description length: {len(desc)} chars")

    # The target_url is a URL that GitHub will associate with the commit status. When users view the commit status
    # on GitHub (for example, in pull requests or on the commit page), they can click on the status check and be
    # directed to this URL. We want to link to the expected wandb report based on the run name
    target_url = None
    wandb_run_id = os.getenv("METTA_RUN_ID") or None
    if wandb_run_id:
        target_url = f"https://wandb.ai/metta-research/metta/runs/{wandb_run_id}"
        print(f"[DEBUG] Target URL: {target_url}")

    # Light retry for transient errors
    for attempt in range(1, 5):
        try:
            print("[DEBUG] Calling post_commit_status with:")
            print(f"  commit_sha: {commit_sha}")
            print(f"  state: {state}")
            print(f"  repo: {repo}")
            print(f"  context: {context}")
            print(f"  description: {desc}")
            print(f"  target_url: {target_url}")

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
            error_msg = str(e)

            # Don't retry client errors - they won't succeed
            if "422" in error_msg or "401" in error_msg or "403" in error_msg:
                print(f"[ERR] Failed to post status: {e}", file=sys.stderr)
                if "422" in error_msg:
                    print("[ERR] Error 422 usually means:", file=sys.stderr)
                    print("  - Invalid state (must be: error, failure, pending, success)", file=sys.stderr)
                    print("  - Description too long (>140 chars)", file=sys.stderr)
                    print("  - Context too long (>255 chars)", file=sys.stderr)
                    print("  - Invalid characters in fields", file=sys.stderr)
                    print("  - Commit doesn't exist in repository", file=sys.stderr)
                return 1

            if attempt == 4:
                print(f"[ERR] Failed to post status after retries: {e}", file=sys.stderr)
                return 2

            sleep_s = 2**attempt
            print(f"[WARN] Post failed (attempt {attempt}), retrying in {sleep_s}s: {e}", file=sys.stderr)
            time.sleep(sleep_s)

    return 2


if __name__ == "__main__":
    sys.exit(main())
