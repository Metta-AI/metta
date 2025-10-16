"""Utilities for obtaining git metadata needed by remote runs."""

from __future__ import annotations

import logging
import os
from typing import Optional

import gitta

logger = logging.getLogger(__name__)


def get_git_hash_for_remote_task(
    target_repo: Optional[str] = None,
    skip_git_check: bool = False,
    skip_cmd: str = "skipping git check",
) -> Optional[str]:
    """Return the commit hash to attach to remote tasks."""

    try:
        current_commit = gitta.get_current_commit()
    except (gitta.GitError, ValueError):
        logger.warning("Not in a git repository, using git_hash=None")
        return None

    if target_repo and not gitta.is_repo_match(target_repo):
        logger.warning("Origin not set to %s, using git_hash=None", target_repo)
        return None

    on_skypilot = bool(os.getenv("SKYPILOT_TASK_ID"))
    has_changes, status_output = gitta.has_unstaged_changes()
    if has_changes:
        logger.warning("Working tree has unstaged changes.\n%s", status_output)
        if not skip_git_check:
            if on_skypilot:
                logger.warning("Running on skypilot: proceeding despite unstaged changes")
            else:
                raise gitta.GitError(
                    "You have uncommitted changes to tracked files that won't be reflected in the remote task.\n"
                    f"You can push your changes or specify to skip this check with {skip_cmd}"
                )
    elif status_output:
        logger.info("Proceeding with unstaged changes.\n%s", status_output)

    if not gitta.is_commit_pushed(current_commit) and not on_skypilot:
        short_commit = current_commit[:8]
        if not skip_git_check:
            raise gitta.GitError(
                f"Commit {short_commit} hasn't been pushed.\n"
                f"You can push your changes or specify to skip this check with {skip_cmd}"
            )
        logger.warning("Proceeding with unpushed commit %s due to %s", short_commit, skip_cmd)

    return current_commit
