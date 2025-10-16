"""Commit validation for remote task execution with SkyPilot awareness.

Wraps gitta.validate_commit_state with SkyPilot-specific logic to ensure
remote tasks can reproduce local code state.
"""

from __future__ import annotations

import logging
import os
from typing import Optional

import gitta

logger = logging.getLogger(__name__)


def get_validated_commit_hash(
    target_repo: Optional[str] = None,
    skip_git_check: bool = False,
    skip_cmd: str = "skipping git check",
) -> Optional[str]:
    """Get current commit hash with SkyPilot-aware validation for remote execution.

    This wraps gitta.validate_commit_state with additional SkyPilot-specific
    behavior: when running on SkyPilot (SKYPILOT_TASK_ID is set), validation
    is automatically relaxed.

    Args:
        target_repo: Optional repository slug (e.g., "owner/repo") to validate against
        skip_git_check: Skip validation checks (useful in CI/remote environments)
        skip_cmd: Message to show in error about how to skip validation

    Returns:
        Current commit hash if validation passes, None if not in a git repo

    Raises:
        GitError: If validation fails (uncommitted changes, unpushed commits, etc.)
    """
    # Return None if not in a git repo
    try:
        gitta.get_current_commit()
    except (gitta.GitError, ValueError):
        logger.warning("Not in a git repository, using git_hash=None")
        return None

    # Check repository match before validation
    if target_repo and not gitta.is_repo_match(target_repo):
        logger.warning("Not in repository %s, using git_hash=None", target_repo)
        return None

    # Skip validation if requested or running on SkyPilot
    on_skypilot = bool(os.getenv("SKYPILOT_TASK_ID"))
    if skip_git_check or on_skypilot:
        if on_skypilot:
            logger.info("Running on SkyPilot, skipping git validation")
        return gitta.get_current_commit()

    # Perform full validation
    try:
        return gitta.validate_commit_state(
            require_clean=True,
            require_pushed=True,
            target_repo=target_repo,
            allow_untracked=False,
        )
    except gitta.GitError as e:
        # Add context about skip_cmd to the error
        raise gitta.GitError(f"{e}\nYou can skip this check with {skip_cmd}") from e
