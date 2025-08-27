"""
GitHub status posting functionality.
"""

import logging
import sys
import time
from pathlib import Path
from typing import Optional

# Allow importing metta.common from repo if present (no install needed)
REPO_ROOT = Path(__file__).resolve().parents[2]
CANDIDATES = [REPO_ROOT / "common" / "src", Path("/workspace/metta/common/src")]
for p in CANDIDATES:
    if p.exists():
        sys.path.insert(0, str(p))
        break

try:
    from metta.common.util.github import post_commit_status
except ImportError:
    post_commit_status = None

logger = logging.getLogger(__name__)


def set_github_status(
    state: str,
    description: str,
    commit_sha: str,
    repo: str,
    token: str,
    context: str = "Skypilot/E2E",
    exit_code: Optional[int] = None,
    job_id: Optional[str] = None,
    wandb_run_id: Optional[str] = None,
    max_retries: int = 4,
) -> bool:
    """
    Post GitHub commit status.

    Args:
        state: Status state (success/failure/error/pending)
        description: Status description
        commit_sha: Git SHA to update
        repo: Repository (e.g. "Metta-AI/metta")
        token: GitHub Personal Access Token
        context: Status context (default "Skypilot/E2E")
        exit_code: Optional exit code to include in description
        job_id: Optional SkyPilot job ID
        wandb_run_id: Optional WandB run ID for target URL
        max_retries: Maximum number of retry attempts

    Returns:
        True if status was posted successfully, False otherwise
    """
    if not all([state, description, commit_sha, repo, token]):
        logger.warning("GitHub status requires state, description, commit_sha, repo, and token")
        return False

    if post_commit_status is None:
        logger.warning("GitHub status posting not available (metta.common not found)")
        return False

    # Build description
    desc = description

    # Add exit code to description if provided
    if exit_code is not None and exit_code != 0 and state in ["failure", "error"]:
        desc += f" (exit code {exit_code})"

    # Add job ID to description if available
    if job_id:
        logger.info(f"Setting GitHub status for job {job_id}")
        desc += f" - [ jl {job_id} ]"

    # Build target URL
    target_url = None
    if wandb_run_id:
        target_url = f"https://wandb.ai/metta-research/metta/runs/{wandb_run_id}"
        logger.info(f"Target URL: {target_url}")

    logger.info(f"Setting GitHub status: {state} - {desc}")

    # Light retry for transient errors
    for attempt in range(1, max_retries + 1):
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
            logger.info(f"{repo}@{commit_sha[:8]} -> {state} ({context})")
            return True
        except Exception as e:
            if attempt == max_retries:
                logger.error(f"Failed to post status after {max_retries} retries: {e}")
                return False

            sleep_s = 2**attempt
            logger.warning(f"Post failed (attempt {attempt}), retrying in {sleep_s}s: {e}")
            time.sleep(sleep_s)

    return False
