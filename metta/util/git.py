import subprocess
from typing import Optional


class GitError(Exception):
    """Custom exception for git-related errors."""


def run_git(*args: str, allow_codes: Optional[list[int]] = None) -> str:
    """
    Run a git command and return its output.

    Args:
        *args: Git command arguments.
        allow_codes: Optional list of non-zero exit codes to allow without raising.

    Returns:
        str: Output of the command.

    Raises:
        GitError: For any disallowed non-zero return code.
    """
    try:
        result = subprocess.run(["git", *args], capture_output=True, text=True, check=True)
        return result.stdout.strip()
    except subprocess.CalledProcessError as e:
        if allow_codes and e.returncode in allow_codes:
            return e.stdout.strip()
        if e.returncode == 129:
            raise GitError(f"Malformed git command or bad object: {e.stderr.strip()}") from e
        raise GitError(f"Git command failed ({e.returncode}): {e.stderr.strip()}") from e


def get_current_branch() -> str:
    """Get the current git branch name."""
    try:
        return run_git("symbolic-ref", "--short", "HEAD")
    except GitError as e:
        if "not a git repository" in str(e):
            raise ValueError("Not in a git repository") from e
        elif "HEAD is not a symbolic ref" in str(e):
            return get_current_commit()
        raise


def get_current_commit() -> str:
    """Get the current git commit hash."""
    return run_git("rev-parse", "HEAD")


def commit_exists(commit_hash: str) -> bool:
    """
    Check if the commit can be checked out. This verifies the commit object
    exists locally and is a full commit (not a tag/blob/partial).
    """
    try:
        obj_type = run_git("cat-file", "-t", commit_hash)
        return obj_type.strip() == "commit"
    except GitError:
        return False


def is_commit_contained(branch: str, commit_hash: str) -> bool:
    """
    Check if the commit is contained in the given remote branch.

    Args:
        branch: Remote branch name.
        commit_hash: Commit hash to check.

    Returns:
        bool: True if commit is contained, False otherwise.
    """
    try:
        branches = run_git("branch", "-r", "--contains", commit_hash)
        return branch in branches.splitlines()
    except GitError:
        return False
