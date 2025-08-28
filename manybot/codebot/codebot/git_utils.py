"""
Minimal git utilities for codebot.

Standalone implementation of git operations needed by codeclip.
"""

import subprocess
from pathlib import Path
from typing import List, Optional


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


def add_all(repo_root: Path) -> None:
    """Add all changes to the staging area."""
    run_git_in_dir(repo_root, "add", ".")


def commit(repo_root: Path, message: str, author_name: Optional[str] = None, author_email: Optional[str] = None) -> str:
    """Create a commit with the given message. Returns commit hash."""
    args = ["commit", "-m", message]

    # Set author if provided
    if author_name and author_email:
        args.extend(["--author", f"{author_name} <{author_email}>"])

    run_git_in_dir(repo_root, *args)

    # Extract commit hash from output
    commit_hash = run_git_in_dir(repo_root, "rev-parse", "HEAD")
    return commit_hash


def get_commit_message(repo_root: Path, commit_hash: str) -> str:
    """Get the commit message for a given commit hash."""
    try:
        return run_git_in_dir(repo_root, "log", "--format=%B", "-n", "1", commit_hash)
    except GitError:
        return ""


def get_last_commits(repo_root: Path, count: int = 20) -> List[dict]:
    """Get the last N commits with hash and message."""
    try:
        # Format: hash|subject
        output = run_git_in_dir(repo_root, "log", f"--max-count={count}", "--format=%H|%s")
        commits = []

        for line in output.split("\n"):
            if "|" in line:
                hash_part, subject = line.split("|", 1)
                commits.append({"hash": hash_part.strip(), "subject": subject.strip()})

        return commits
    except GitError:
        return []


def reset_soft(repo_root: Path, commit_hash: str) -> None:
    """Reset to a commit using --soft (keeps changes in staging area)."""
    run_git_in_dir(repo_root, "reset", "--soft", commit_hash)


def get_current_branch(repo_root: Path) -> str:
    """Get the current branch name."""
    try:
        return run_git_in_dir(repo_root, "branch", "--show-current")
    except GitError:
        return "main"  # Fallback to main


def has_uncommitted_changes(repo_root: Path) -> bool:
    """Check if there are uncommitted changes in the working directory."""
    try:
        # Check if there are staged changes
        staged = run_git_in_dir(repo_root, "diff", "--staged", "--name-only")
        # Check if there are unstaged changes
        unstaged = run_git_in_dir(repo_root, "diff", "--name-only")

        return bool(staged.strip() or unstaged.strip())
    except GitError:
        return False
