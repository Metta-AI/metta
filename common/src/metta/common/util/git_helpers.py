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


def get_task_commit_hash(
    target_repo: Optional[str] = None,
    skip_git_check: bool = False,
) -> Optional[str]:
    """Get current commit hash for task execution with SkyPilot-aware validation.

    Returns a commit hash suitable for remote task execution. Validation is
    automatically skipped when running on SkyPilot (SKYPILOT_TASK_ID is set).

    Args:
        target_repo: Optional repository slug (e.g., "owner/repo") to validate against
        skip_git_check: Skip validation checks (useful in CI/remote environments)

    Returns:
        Current commit hash if validation passes, None if not in a git repo

    Raises:
        GitError: If validation fails (uncommitted changes, unpushed commits, etc.)
    """
    # Check if we're in a git repo
    try:
        commit_hash = gitta.get_current_commit()
    except (gitta.GitError, ValueError):
        logger.warning("Not in a git repository, using git_hash=None")
        return None

    # Check if we're in the right repo
    if target_repo and not gitta.is_repo_match(target_repo):
        logger.warning("Not in repository %s, using git_hash=None", target_repo)
        return None

    # Skip validation if requested or on SkyPilot
    if skip_git_check:
        return commit_hash

    if os.getenv("SKYPILOT_TASK_ID"):
        logger.info("Running on SkyPilot, skipping git validation")
        return commit_hash

    # Perform full validation
    return gitta.validate_commit_state(
        require_clean=True,
        require_pushed=True,
        target_repo=target_repo,
        allow_untracked=False,
    )
