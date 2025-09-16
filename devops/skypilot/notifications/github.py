#!/usr/bin/env python3
"""GitHub status notification platform."""

from devops.skypilot.notifications.notification import get_notification_info
from devops.skypilot.utils.job_config import JobConfig
from gitta import post_commit_status
from metta.common.util.constants import (
    METTA_GITHUB_ORGANIZATION,
    METTA_GITHUB_REPO,
    METTA_WANDB_ENTITY,
    METTA_WANDB_PROJECT,
)
from metta.common.util.log_config import getRankAwareLogger
from metta.common.util.retry import retry_function

logger = getRankAwareLogger(__name__)


def send_github_status(termination_reason: str, job_config: JobConfig) -> bool:
    """Send GitHub status update."""
    if not job_config.enable_github_status or not job_config.github_pat:
        logger.debug("GitHub status updates disabled")
        return False

    info = get_notification_info(termination_reason, job_config)

    # Map state to GitHub format
    github_state = "success" if info["state"] == "success" else "failure"

    # Validate SHA
    commit_sha = job_config.metta_git_ref
    if not commit_sha or len(commit_sha) < 40:
        logger.error(f"Invalid GitHub SHA: '{commit_sha}' (expected 40 characters)")
        return False

    # Build description
    description = info["description"]
    if job_config.skypilot_job_id:
        description += f" - [ jl {job_config.skypilot_job_id} ]"

    # Build target URL
    target_url = None
    if job_config.metta_run_id:
        target_url = f"https://wandb.ai/{METTA_WANDB_ENTITY}/{METTA_WANDB_PROJECT}/runs/{job_config.metta_run_id}"

    repo = job_config.github_repository or f"{METTA_GITHUB_ORGANIZATION}/{METTA_GITHUB_REPO}"

    logger.info(f"Posting GitHub status: {repo}@{commit_sha[:8]} → {github_state}")

    try:
        retry_function(
            lambda: post_commit_status(
                commit_sha=commit_sha,
                state=github_state,
                repo=repo,
                context=job_config.github_status_context,
                description=description,
                target_url=target_url,
                token=job_config.github_pat,
            ),
            max_retries=3,
            initial_delay=2.0,
            max_delay=30.0,
        )
        logger.info("✅ Successfully set GitHub status")
        return True
    except Exception as e:
        logger.error(f"GitHub status update failed: {e}")
        return False
