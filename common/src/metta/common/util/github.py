import logging
import os
import time
from typing import Any, Dict, Optional

import httpx

from metta.common.util.constants import METTA_GITHUB_ORGANIZATION, METTA_GITHUB_REPO


def post_commit_status(
    commit_sha: str,
    state: str,
    repo: Optional[str] = None,
    context: str = "CI/Skypilot",
    description: Optional[str] = None,
    target_url: Optional[str] = None,
    token: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Post a status update for a commit to GitHub.

    Args:
        commit_sha: The SHA of the commit
        state: The state of the status (error, failure, pending, success)
        repo: Repository in format "owner/repo". If not provided, uses default from constants
        context: A string label to differentiate this status from others
        description: A short description of the status
        target_url: The target URL to associate with this status
        token: GitHub token. If not provided, uses GITHUB_TOKEN env var

    Returns:
        The created status object

    Raises:
        ValueError: If no token is available
        httpx.HTTPError: If the API request fails
    """
    # Use default repo if not provided
    if repo is None:
        repo = f"{METTA_GITHUB_ORGANIZATION}/{METTA_GITHUB_REPO}"

    # Get token
    github_token = token or os.environ.get("GITHUB_TOKEN")
    if not github_token:
        raise ValueError("GitHub token not provided and GITHUB_TOKEN environment variable not set")

    # Build request
    url = f"https://api.github.com/repos/{repo}/statuses/{commit_sha}"
    headers = {"Authorization": f"token {github_token}", "Accept": "application/vnd.github.v3+json"}

    data = {
        "state": state,
        "context": context,
    }

    if description:
        data["description"] = description
    if target_url:
        data["target_url"] = target_url

    # Make request
    try:
        response = httpx.post(url, headers=headers, json=data, timeout=10.0)
        response.raise_for_status()
        return response.json()
    except httpx.HTTPError as e:
        logging.error(f"Failed to post GitHub status: {e}")
        raise


def set_skypilot_test_status(
    state: str,
    description: str,
    commit_sha: str,
    token: Optional[str] = None,
    context: str = "Skypilot/E2E",
    exit_code: Optional[int] = None,
    job_id: Optional[str] = None,
    wandb_run_id: Optional[str] = None,
    max_retries: int = 4,
) -> bool:
    """
    Post GitHub commit status for Skypilot tests with retry logic and enhanced descriptions.

    Args:
        state: Status state (success/failure/error/pending)
        description: Status description
        commit_sha: Git SHA to update
        token: GitHub token. If not provided, uses GITHUB_TOKEN env var
        context: Status context (default "Skypilot/E2E")
        exit_code: Optional exit code to include in description
        job_id: Optional SkyPilot job ID
        wandb_run_id: Optional WandB run ID for target URL
        max_retries: Maximum number of retry attempts

    Returns:
        True if status was posted successfully, False otherwise
    """
    logger = logging.getLogger(__name__)

    if not all([state, description, commit_sha]):
        logger.warning("GitHub status requires state, description, and commit_sha")
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
                repo=None,  # Always use default repo
                context=context,
                description=desc,
                target_url=target_url,
                token=token,
            )

            # Format repo for logging
            display_repo = f"{METTA_GITHUB_ORGANIZATION}/{METTA_GITHUB_REPO}"
            logger.info(f"{display_repo}@{commit_sha[:8]} -> {state} ({context})")
            return True
        except Exception as e:
            if attempt == max_retries:
                logger.error(f"Failed to post status after {max_retries} retries: {e}")
                return False

            sleep_s = 2**attempt
            logger.warning(f"Post failed (attempt {attempt}), retrying in {sleep_s}s: {e}")
            time.sleep(sleep_s)

    return False
