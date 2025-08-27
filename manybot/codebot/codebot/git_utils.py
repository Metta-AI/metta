"""
Minimal git utilities for codebot.

Standalone implementation of git operations needed by codeclip.
"""

import subprocess
from pathlib import Path
from typing import Optional


class GitError(Exception):
    """Custom exception for git-related errors."""

    pass


def run_git_in_dir(cwd: Path, *args: str) -> str:
    """Run a git command in a specific directory and return its output."""
    try:
        result = subprocess.run(["git", *args], capture_output=True, text=True, check=True, cwd=str(cwd))
        return result.stdout.strip()
    except subprocess.CalledProcessError as e:
        raise GitError(f"Git command failed ({e.returncode}): {e.stderr.strip()}") from e
    except FileNotFoundError as e:
        raise GitError("Git is not installed!") from e


def find_root(start: Path) -> Optional[Path]:
    """Return the repository root that contains start, or None if not in a repo."""
    current = start if start.is_dir() else start.parent
    while current != current.parent:
        if (current / ".git").is_dir():
            return current
        current = current.parent
    return None


def fetch(repo_root: Path) -> None:
    """Best effort git fetch. No exception if it fails."""
    try:
        run_git_in_dir(repo_root, "fetch")
    except GitError:
        # Network issues are non-fatal for fetch
        pass


def ref_exists(repo_root: Path, ref: str) -> bool:
    """True if ref resolves in this repo."""
    try:
        run_git_in_dir(repo_root, "rev-parse", "--verify", "--quiet", ref)
        return True
    except GitError:
        return False


def diff(repo_root: Path, base_ref: str) -> str:
    """Unified diff of working tree vs base_ref. Empty string if no changes or failure."""
    try:
        return run_git_in_dir(repo_root, "diff", base_ref)
    except GitError:
        return ""
