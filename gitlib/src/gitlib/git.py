"""Core git functions and utilities."""

import os
import subprocess
from functools import wraps
from pathlib import Path
from time import time
from typing import Optional

# Constants that were in metta.common.util.constants
METTA_GITHUB_ORGANIZATION = "Metta-AI"
METTA_GITHUB_REPO = "metta"

# Simple memoization decorator


def memoize(max_age=60):
    """Simple memoization decorator with time-based expiry."""

    def decorator(func):
        cache = {}
        cache_time = {}

        @wraps(func)
        async def wrapper(*args, **kwargs):
            key = str(args) + str(kwargs)
            current_time = time()

            if key in cache and current_time - cache_time[key] < max_age:
                return cache[key]

            result = await func(*args, **kwargs)
            cache[key] = result
            cache_time[key] = current_time
            return result

        return wrapper

    return decorator


class GitError(Exception):
    """Custom exception for git-related errors."""


METTA_API_REPO = f"{METTA_GITHUB_ORGANIZATION}/{METTA_GITHUB_REPO}"
METTA_API_REPO_URL = f"git@github.com:{METTA_API_REPO}.git"


def run_git_with_cwd(cwd: str, *args) -> str:
    """Run git command with a specific working directory."""
    try:
        result = subprocess.run(["git"] + list(args), cwd=cwd, capture_output=True, text=True, check=True)
        return result.stdout.strip()
    except subprocess.CalledProcessError as e:
        raise GitError(f"Git command failed: {e.stderr.strip()}") from e


def run_git(*args, cwd: Optional[str] = None, allow_error: bool = False) -> str:
    """Run git command and return output, raise GitError on failure unless allow_error=True."""
    try:
        result = subprocess.run(["git"] + list(args), cwd=cwd, capture_output=True, text=True, check=True)
        return result.stdout.strip()
    except subprocess.CalledProcessError as e:
        if allow_error:
            return ""
        raise GitError(f"Git command failed: {e.stderr.strip()}") from e


def run_git_in_dir(repo_root: Path, *args, allow_error: bool = False) -> str:
    """Run git command in a specific directory."""
    return run_git(*args, cwd=str(repo_root), allow_error=allow_error)


def run_gh(*args, cwd: Optional[str] = None, check: bool = True) -> str:
    """Run gh CLI command and return output, raise exception on failure if check=True."""
    try:
        result = subprocess.run(["gh"] + list(args), cwd=cwd, capture_output=True, text=True, check=check)
        return result.stdout.strip()
    except subprocess.CalledProcessError as e:
        if not check:
            return ""
        raise RuntimeError(f"gh command failed: {e.stderr.strip()}") from e


def get_current_branch() -> str:
    """Get current branch name for the metta repo."""
    try:
        return run_git_with_cwd(os.getcwd(), "rev-parse", "--abbrev-ref", "HEAD")
    except GitError:
        return "main"


def get_current_commit() -> str:
    """Get current commit hash for the metta repo."""
    try:
        return run_git_with_cwd(os.getcwd(), "rev-parse", "HEAD")
    except GitError:
        return ""


def get_branch_commit(branch: str) -> str:
    """Get the commit hash for a specific branch."""
    try:
        return run_git_with_cwd(os.getcwd(), "rev-parse", branch)
    except GitError:
        return ""


def get_commit_message(commit: str) -> str:
    """Get the commit message for a specific commit."""
    try:
        return run_git_with_cwd(os.getcwd(), "log", "-1", "--pretty=%B", commit)
    except GitError:
        return ""


def has_unstaged_changes() -> bool:
    """Check if there are unstaged changes in the repo."""
    try:
        status = run_git_with_cwd(os.getcwd(), "status", "--porcelain")
        return bool(status)
    except GitError:
        return False


def is_commit_pushed(commit: str) -> bool:
    """Check if a commit has been pushed to the remote."""
    try:
        # Check if commit exists in remote branches
        remote_branches = run_git_with_cwd(os.getcwd(), "branch", "-r", "--contains", commit)
        return bool(remote_branches)
    except GitError:
        return False


def validate_git_ref(ref: str) -> bool:
    """Validate that a git reference exists."""
    try:
        run_git_with_cwd(os.getcwd(), "rev-parse", "--verify", ref)
        return True
    except GitError:
        return False


def get_matched_pr(target_branch: str = "origin/main") -> Optional[str]:
    """Get PR number that matches the current branch."""
    current_branch = get_current_branch()
    if not current_branch or current_branch == "main":
        return None

    try:
        # Use gh CLI to find PRs for this branch
        pr_list = run_gh("pr", "list", "--head", current_branch, "--json", "number", "--limit", "1")
        if pr_list:
            import json

            prs = json.loads(pr_list)
            if prs:
                return str(prs[0]["number"])
    except Exception:
        pass

    return None


def get_remote_url(remote: str = "origin") -> str:
    """Get the URL of a git remote."""
    try:
        return run_git_with_cwd(os.getcwd(), "remote", "get-url", remote)
    except GitError:
        return ""


def is_metta_ai_repo() -> bool:
    """Check if current repo is the Metta AI repository."""
    remote_url = get_remote_url()
    return METTA_API_REPO in remote_url or METTA_API_REPO_URL in remote_url


def get_git_hash_for_remote_task(task_name: str) -> Optional[str]:
    """Get git hash for a remote task."""
    # This would require implementation based on specific remote task tracking
    # For now, return None as placeholder
    return None


def get_latest_commit(repo_root: Path, branch: str = "HEAD") -> str:
    """Get the latest commit hash for a repository."""
    try:
        return run_git_in_dir(repo_root, "rev-parse", branch)
    except GitError:
        return ""


def get_file_list(repo_root: Path) -> list[str]:
    """Get list of files in a git repository."""
    try:
        files = run_git_in_dir(repo_root, "ls-files")
        return files.split("\n") if files else []
    except GitError:
        return []


def get_commit_count(repo_root: Path) -> int:
    """Get the number of commits in a repository."""
    try:
        count = run_git_in_dir(repo_root, "rev-list", "--count", "HEAD")
        return int(count) if count else 0
    except (GitError, ValueError):
        return 0


def add_remote(repo_root: Path, name: str, url: str) -> None:
    """Add a git remote to a repository."""
    try:
        run_git_in_dir(repo_root, "remote", "add", name, url)
    except GitError as e:
        # Check if remote already exists
        if "already exists" in str(e):
            # Update the URL instead
            run_git_in_dir(repo_root, "remote", "set-url", name, url)
        else:
            raise


def find_root(path: Optional[Path] = None) -> Optional[Path]:
    """Find the root of a git repository.

    Args:
        path: Starting path to search from. If None, uses current directory.

    Returns:
        Path to repository root, or None if not in a git repository.
    """
    if path is None:
        path = Path.cwd()
    elif not isinstance(path, Path):
        path = Path(path)

    # Walk up the directory tree looking for .git
    current = path if path.is_dir() else path.parent

    while current != current.parent:
        if (current / ".git").exists():
            return current
        current = current.parent

    # Check root directory
    if (current / ".git").exists():
        return current

    return None


def fetch(repo_root: Path, remote: str = "origin") -> None:
    """Fetch from a git remote.

    Args:
        repo_root: Root directory of the repository
        remote: Name of the remote (default: origin)
    """
    try:
        run_git_in_dir(repo_root, "fetch", remote, allow_error=False)
    except GitError:
        # Best effort - ignore fetch errors
        pass


def ref_exists(repo_root: Path, ref: str) -> bool:
    """Check if a git reference exists.

    Args:
        repo_root: Root directory of the repository
        ref: Git reference to check (branch, tag, commit)

    Returns:
        True if the reference exists, False otherwise
    """
    try:
        run_git_in_dir(repo_root, "rev-parse", "--verify", ref, allow_error=False)
        return True
    except GitError:
        return False


def diff(repo_root: Path, base_ref: str) -> str:
    """Get git diff against a base reference.

    Args:
        repo_root: Root directory of the repository
        base_ref: Base reference to diff against

    Returns:
        Diff output as string, empty string on error
    """
    try:
        return run_git_in_dir(repo_root, "diff", base_ref)
    except GitError:
        return ""
