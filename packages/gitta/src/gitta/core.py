"""Core git command runner and exceptions."""

from __future__ import annotations

import logging
import os
import shlex
import subprocess
import time
from pathlib import Path
from typing import Iterable, Mapping, Optional

logger = logging.getLogger(__name__)

# Default environment variables to ensure non-interactive git behavior
DEFAULT_ENV = {
    "GIT_PAGER": "cat",  # Don't use pager
    "GIT_TERMINAL_PROMPT": "0",  # Don't prompt for credentials
    "GIT_ASKPASS": "",  # Don't use askpass helper
    "SSH_ASKPASS": "",  # Don't use SSH askpass
    "LC_ALL": "C",  # Ensure consistent output
}


class GitError(Exception):
    """Base exception for git operations."""

    pass


class GitNotInstalledError(GitError):
    """Raised when git is not installed."""

    pass


class DubiousOwnershipError(GitError):
    """Raised when git detects dubious ownership of repository."""

    pass


class NotAGitRepoError(GitError):
    """Raised when operation is performed outside a git repository."""

    pass


def run_git_cmd(
    args: Iterable[str],
    cwd: Optional[Path] = None,
    timeout: Optional[float] = None,
    env_overrides: Optional[Mapping[str, str]] = None,
    check: bool = True,
) -> str:
    """
    Run a git command with consistent environment and error handling.

    Args:
        args: Git command arguments (without 'git' prefix)
        cwd: Working directory for the command
        timeout: Command timeout in seconds (default: 30s)
        env_overrides: Additional environment variables to set
        check: If False, return empty string on error instead of raising

    Returns:
        Command output as string (stripped)

    Raises:
        GitNotInstalledError: If git is not installed
        DubiousOwnershipError: If git detects dubious ownership
        NotAGitRepoError: If not in a git repository
        GitError: For other git command failures
    """
    cmd = ["git", *args]

    # Set up environment
    env = os.environ.copy()
    env.update(DEFAULT_ENV)
    if env_overrides:
        env.update(env_overrides)

    # Default timeout
    if timeout is None:
        timeout = 30.0

    cmd_str = " ".join(shlex.quote(str(a)) for a in cmd)
    logger.debug(f"Running: {cmd_str}")

    t0 = time.time()
    try:
        result = subprocess.run(
            cmd,
            cwd=str(cwd) if cwd else None,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            env=env,
            check=False,
            timeout=timeout,
        )
    except FileNotFoundError as e:
        raise GitNotInstalledError(
            "Git is not installed or not in PATH. Please install git: https://git-scm.com/downloads"
        ) from e
    except subprocess.TimeoutExpired as e:
        raise GitError(f"Git command timed out after {timeout}s: {' '.join(cmd)}") from e

    duration = time.time() - t0

    if logger.isEnabledFor(logging.DEBUG):
        logger.debug(f"Command completed in {duration:.3f}s with exit code {result.returncode}")

    # Handle errors
    if result.returncode != 0:
        stderr = result.stderr.decode("utf-8", errors="replace").strip()

        # Check for specific error conditions
        if "not a git repository" in stderr.lower():
            raise NotAGitRepoError(f"Not in a git repository: {stderr}")

        if "dubious ownership" in stderr:
            repo_path = cwd or Path.cwd()
            raise DubiousOwnershipError(
                f"{stderr}\n\n"
                f"To fix this, run:\n"
                f"  git config --global --add safe.directory {repo_path}\n"
                f"Or set environment variable:\n"
                f"  GITTA_AUTO_ADD_SAFE_DIRECTORY=1"
            )

        # Handle non-critical errors if check=False
        if not check:
            return ""

        # Generic error
        cmd_str = " ".join(shlex.quote(str(a)) for a in args)
        raise GitError(f"git {cmd_str} failed ({result.returncode}): {stderr}")

    return result.stdout.decode("utf-8", errors="surrogateescape").strip()


def run_git_with_cwd(args: list[str], cwd: str | Path | None = None) -> str:
    """Run a git command with optional working directory and return its output."""
    return run_git_cmd(args, cwd=Path(cwd) if cwd else None)


def run_git(*args: str) -> str:
    """Run a git command and return its output."""
    return run_git_cmd(list(args))


def run_git_in_dir(cwd: str | Path, *args: str) -> str:
    """Run a git command in a specific directory and return its output."""
    return run_git_cmd(list(args), cwd=Path(cwd))
