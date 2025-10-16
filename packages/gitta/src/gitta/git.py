"""Higher-level git operations built on top of the helpers in gitta.core."""

from __future__ import annotations

import logging
import re
from functools import lru_cache
from pathlib import Path
from typing import Dict, Optional

from .core import GitError, NotAGitRepoError, run_git, run_git_in_dir

logger = logging.getLogger(__name__)


def get_current_branch() -> str:
    """Get the current git branch name."""
    try:
        return run_git("symbolic-ref", "--short", "HEAD")
    except GitError as e:
        if "HEAD is not a symbolic ref" in str(e):
            return get_current_commit()
        raise


def get_current_commit() -> str:
    """Get the current git commit hash."""
    return run_git("rev-parse", "HEAD")


def get_branch_commit(branch: str) -> str:
    """Get the commit hash for a given branch or remote ref."""
    # only fetch when branch looks like a remote-tracking ref:
    if branch.startswith("origin/"):
        try:
            run_git("fetch", "--quiet")
        except GitError:
            # network issues are non-fatal
            pass

    return run_git("rev-parse", "--verify", branch).strip()


@lru_cache(maxsize=256)
def _get_commit_message_cached(commit_hash: str) -> str:
    """Internal cached helper - only call with resolved commit hashes."""
    return run_git("log", "-1", "--pretty=%B", commit_hash)


def get_commit_message(commit_hash: str) -> str:
    """Get the commit message for a given commit."""
    if len(commit_hash) == 40 and all(c in "0123456789abcdef" for c in commit_hash.lower()):
        return _get_commit_message_cached(commit_hash)

    resolved_hash = run_git("rev-parse", commit_hash)
    return _get_commit_message_cached(resolved_hash)


def has_unstaged_changes(allow_untracked: bool = False) -> tuple[bool, str]:
    """Returns if there are any unstaged changes in the local git checkout.

    If allow_untracked is True, then unstaged changes in the form of new untracked files will be ignored.

    Interpretation of porcelain codes:
    - Lines starting with '??' indicate untracked files
    - Any other status lines indicate changes to tracked files
    """
    status_output = run_git("status", "--porcelain")
    if not allow_untracked:
        return bool(status_output), status_output

    is_dirty_tracked = False
    for line in status_output.splitlines():
        if not line.strip():
            continue
        if line.startswith("??"):
            # untracked file, ignore
            continue
        # any other status code refers to tracked file state change
        is_dirty_tracked = True
        break
    return is_dirty_tracked, status_output


def is_commit_pushed(commit_hash: str) -> bool:
    """
    Check if `commit_hash` has been pushed to the remote tracking branch.

    Fast path:
    - If the current branch has an upstream (e.g. origin/main), we do:
      git merge-base --is-ancestor <commit_hash> <upstream>
      which is a constant-time check.

    Fallback:
    - If no upstream is set, we fall back to the old:
      git branch -r --contains <commit_hash>

    Raises:
        GitError: If the commit hash is invalid or doesn't exist
    """
    # First validate the commit exists
    try:
        run_git("rev-parse", "--verify", commit_hash)
    except GitError as e:
        raise GitError(f"Invalid commit hash: {commit_hash}") from e

    try:
        # Figure out the upstream ref for the current branch (e.g. "origin/main")
        branch = get_current_branch()
        upstream = run_git("rev-parse", "--abbrev-ref", f"{branch}@{{u}}")
    except GitError:
        # No upstream configured - fallback to scanning all remotes
        remote_branches = run_git("branch", "-r", "--contains", commit_hash)
        return bool(remote_branches.strip())

    # Fast constant-time check: is commit_hash an ancestor of upstream?
    try:
        # merge-base --is-ancestor returns exit code 0 if true
        run_git("merge-base", "--is-ancestor", commit_hash, upstream)
        return True
    except GitError:
        return False


def validate_git_ref(ref: str) -> str | None:
    """Validate a git reference exists (locally or in remote)."""
    try:
        commit_hash = run_git("rev-parse", "--verify", ref)
    except GitError:
        return None
    return commit_hash


def resolve_git_ref(ref: str) -> str:
    """Resolve a git reference to its full commit hash."""
    return run_git("rev-parse", ref).strip()


def https_remote_url(url: str) -> str:
    """Canonicalize a git remote URL to a consistent format.

    Converts both SSH and HTTPS GitHub URLs to a canonical HTTPS format
    without the .git suffix for easier comparison.

    Examples:
        git@github.com:Owner/repo.git -> https://github.com/Owner/repo
        https://github.com/Owner/repo.git -> https://github.com/Owner/repo
        ssh://git@github.com/Owner/repo -> https://github.com/Owner/repo
    """
    url = url.strip()

    # Handle SSH format: git@github.com:Owner/repo[.git]
    ssh_match = re.match(r"^git@github\.com:([^/]+)/([^/]+?)(?:\.git)?$", url)
    if ssh_match:
        owner, repo = ssh_match.groups()
        return f"https://github.com/{owner}/{repo}"

    # Handle SSH URL format: ssh://git@github.com/Owner/repo[.git]
    ssh_url_match = re.match(r"^ssh://git@github\.com/([^/]+)/([^/]+?)(?:\.git)?$", url)
    if ssh_url_match:
        owner, repo = ssh_url_match.groups()
        return f"https://github.com/{owner}/{repo}"

    # Handle HTTPS format: https://github.com/Owner/repo[.git]
    if url.startswith("https://github.com/"):
        if url.endswith(".git"):
            url = url[:-4]
        return url

    # Return as-is for non-GitHub URLs
    return url


# Backwards compatibility alias
canonical_remote_url = https_remote_url


def get_remote_url(remote: str = "origin") -> str | None:
    """Get the URL of a remote repository.

    Args:
        remote: Name of the remote (default: "origin")

    Returns:
        The remote URL or None if the remote doesn't exist
    """
    try:
        return run_git("remote", "get-url", remote)
    except GitError:
        return None


def get_all_remotes() -> Dict[str, str]:
    """Get all configured remotes and their URLs.

    Returns:
        Dictionary mapping remote names to their fetch URLs
    """
    try:
        output = run_git("remote", "-v")
        remotes = {}
        for line in output.splitlines():
            if line:
                parts = line.split()
                if len(parts) >= 2 and "(fetch)" in line:
                    remotes[parts[0]] = parts[1]
        return remotes
    except GitError:
        return {}


def is_repo_match(target_repo: str) -> bool:
    """Check if any remote is set to the specified repository.

    This checks all configured remotes, not just 'origin', and handles
    various URL formats (SSH, HTTPS, with/without .git suffix).

    Args:
        target_repo: Repository in format "owner/repo"
    """
    target_url = https_remote_url(f"https://github.com/{target_repo}")

    remotes = get_all_remotes()
    for remote_url in remotes.values():
        if https_remote_url(remote_url) == target_url:
            return True

    return False


def get_file_list(repo_path: Path | None = None, ref: str = "HEAD") -> list[str]:
    """Get list of all files in repository."""
    try:
        if repo_path:
            # First check if ref exists
            run_git_in_dir(repo_path, "rev-parse", "--verify", ref)
            # If ref exists, list files
            output = run_git_in_dir(repo_path, "ls-tree", "-r", "--name-only", ref)
        else:
            # First check if ref exists
            run_git("rev-parse", "--verify", ref)
            # If ref exists, list files
            output = run_git("ls-tree", "-r", "--name-only", ref)
        return output.split("\n") if output else []
    except GitError as e:
        # If ref doesn't exist (empty repo), return empty list
        if "Not a valid object name" in str(e) or "bad revision" in str(e) or "Needed a single revision" in str(e):
            return []
        raise


def get_commit_count(repo_path: Path | None = None) -> int:
    """Get total number of commits."""
    try:
        if repo_path:
            # First check if HEAD exists
            run_git_in_dir(repo_path, "rev-parse", "--verify", "HEAD")
            # If HEAD exists, count commits
            result = run_git_in_dir(repo_path, "rev-list", "--count", "HEAD")
        else:
            # First check if HEAD exists
            run_git("rev-parse", "--verify", "HEAD")
            # If HEAD exists, count commits
            result = run_git("rev-list", "--count", "HEAD")
        return int(result)
    except GitError as e:
        # If HEAD doesn't exist (empty repo), return 0
        if "Not a valid object name" in str(e) or "bad revision" in str(e) or "Needed a single revision" in str(e):
            return 0
        raise


def add_remote(name: str, url: str, repo_path: Path | None = None):
    """Add a remote repository."""
    # Try to remove first in case it exists
    try:
        if repo_path:
            run_git_in_dir(repo_path, "remote", "remove", name)
        else:
            run_git("remote", "remove", name)
    except GitError:
        pass  # Ignore if it doesn't exist

    if repo_path:
        run_git_in_dir(repo_path, "remote", "add", name, url)
    else:
        run_git("remote", "add", name, url)


def find_root(start: Path) -> Optional[Path]:
    """Return the repository root that contains start, or None if not in a repo.

    This uses git's own logic to find the repository root, which correctly
    handles worktrees, submodules, and other edge cases.

    Args:
        start: Starting path (file or directory) to search from

    Returns:
        Path to repository root or None if not in a repo
    """
    try:
        from .core import run_git_cmd

        # Ensure we have a directory for cwd
        if start.is_file():
            cwd = start.parent
        else:
            cwd = start

        # Use git's own logic to find the repository root
        root = run_git_cmd(["rev-parse", "--show-toplevel"], cwd=cwd)
        return Path(root)
    except (GitError, NotAGitRepoError):
        return None


def fetch(repo_root: Path) -> None:
    """Best effort git fetch. No exception if it fails."""
    try:
        run_git_in_dir(repo_root, "fetch")
    except GitError:
        # Network issues are non-fatal for fetch
        pass


# Manual cache for ref_exists - only store successful lookups
_ref_exists_cache: Dict[tuple[Path, str], bool] = {}


def ref_exists(repo_root: Path, ref: str) -> bool:
    """True if ref resolves in this repo. Only caches positive results."""
    cache_key = (repo_root, ref)
    if cache_key in _ref_exists_cache:
        return True

    try:
        run_git_in_dir(repo_root, "rev-parse", "--verify", "--quiet", ref)
        _ref_exists_cache[cache_key] = True
        return True
    except GitError:
        return False


def diff(repo_root: Path, base_ref: str) -> str:
    """Unified diff of working tree vs base_ref. Empty string if no changes or failure."""
    try:
        return run_git_in_dir(repo_root, "diff", base_ref)
    except GitError:
        return ""


def git_log_since(ref: str, max_count: int = 20, oneline: bool = True) -> str:
    """Return git log output from `ref` (exclusive) to HEAD.

    Falls back to the most recent `max_count` commits if `ref` cannot be resolved.
    """

    try:
        args = ["log", f"{ref}..HEAD"]
        if oneline:
            args.append("--oneline")
        return run_git(*args)
    except GitError:
        try:
            fallback_args = ["log"]
            if oneline:
                fallback_args.append("--oneline")
            fallback_args.append(f"-{max_count}")
            return run_git(*fallback_args)
        except GitError:
            return "Unable to retrieve git log"


def validate_commit_state(
    require_clean: bool = True,
    require_pushed: bool = True,
    target_repo: Optional[str] = None,
    allow_untracked: bool = False,
) -> str:
    """Validate git working tree state and return current commit hash.

    This is useful before remote execution to ensure the remote environment
    can reproduce the local code state.

    Args:
        require_clean: Raise error if there are uncommitted changes
        require_pushed: Raise error if current commit isn't pushed
        target_repo: If provided, validate we're in this repo (e.g., "owner/repo")
        allow_untracked: If True, allow untracked files (only check tracked changes)

    Returns:
        Current commit hash

    Raises:
        GitError: If validation fails or not in a git repository
    """
    current_commit = get_current_commit()

    # Validate we're in the right repository
    if target_repo and not is_repo_match(target_repo):
        raise GitError(f"Not in repository {target_repo}")

    # Check for uncommitted changes
    if require_clean:
        has_changes, status_output = has_unstaged_changes(allow_untracked=allow_untracked)
        if has_changes:
            raise GitError(
                f"Working tree has uncommitted changes to tracked files:\n{status_output}\n"
                "Commit or stash your changes before proceeding."
            )

    # Check if commit is pushed
    if require_pushed and not is_commit_pushed(current_commit):
        short_commit = current_commit[:8]
        raise GitError(f"Commit {short_commit} hasn't been pushed to remote.\nPush your changes before proceeding.")

    return current_commit
