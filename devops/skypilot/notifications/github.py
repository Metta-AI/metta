#!/usr/bin/env python3
from typing import Literal

from devops.skypilot.utils.job_config import JobConfig
from gitta import post_commit_status
from metta.common.util.log_config import getRankAwareLogger
from metta.common.util.retry import retry_function

logger = getRankAwareLogger(__name__)


def set_github_status(
    state: Literal["success", "failure", "error", "pending"], description: str, job_config: JobConfig
) -> bool:
    if not job_config.enable_github_status:
        logger.debug("GitHub status updates disabled")
        return False

    # Extract and validate required fields
    commit_sha = job_config.metta_git_ref or ""
    token = job_config.github_pat
    context = job_config.github_status_context

    if not all([state, description, commit_sha, token]):
        logger.warning(
            f"Skipping GitHub status - missing params: "
            f"state={bool(state)}, desc={bool(description)}, "
            f"sha={bool(commit_sha)}, token={'set' if token else 'missing'}"
        )
        return False

    # Validate full SHA format
    if len(commit_sha) < 40:
        logger.error(f"Invalid GitHub SHA: '{commit_sha}' (expected 40 characters)")
        return False

    # Build description
    desc = description
    if job_config.skypilot_job_id:
        desc += f" - [ jl {job_config.skypilot_job_id} ]"

    # Build target URL
    target_url = None
    if job_config.metta_run_id:
        target_url = f"https://wandb.ai/metta-research/metta/runs/{job_config.metta_run_id}"

    repo = job_config.github_repository

    if not repo:
        logger.error("GitHub repository not configured")
        return False

    # Log detailed payload info before posting
    logger.info(
        f"Posting GitHub status:\n"
        f"  repo        = {repo}\n"
        f"  sha         = {commit_sha}\n"
        f"  state       = {state}\n"
        f"  context     = {context}\n"
        f"  description = {desc}\n"
        f"  target_url  = {target_url}"
    )

    try:
        retry_function(
            lambda: post_commit_status(
                commit_sha=commit_sha,
                state=state,
                repo=repo,
                context=context,
                description=desc,
                target_url=target_url,
                token=token,
            ),
            max_retries=3,
            initial_delay=2.0,
            max_delay=30.0,
        )
        logger.info(f"✅ Successfully set GitHub status: {repo}@{commit_sha[:8]} → {state}")
        return True

    except Exception as e:
        logger.error(f"GitHub status update failed: {e}")
        return False
