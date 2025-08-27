"""Git utilities library for Metta projects."""

from __future__ import annotations

import logging
import os
import re
import shlex
import subprocess
import tempfile
import time
from functools import wraps
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Optional

import httpx

logger = logging.getLogger(__name__)

# ============================================================================
# Command Runner - Core Infrastructure
# ============================================================================

# Default environment variables to ensure non-interactive git behavior
_DEFAULT_ENV = {
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
    return_bytes: bool = False,
    env_overrides: Optional[Mapping[str, str]] = None,
    check: bool = True,
) -> str | bytes:
    """
    Run a git command with consistent environment and error handling.

    Args:
        args: Git command arguments (without 'git' prefix)
        cwd: Working directory for the command
        timeout: Command timeout in seconds (default: 30s)
        return_bytes: Return raw bytes instead of decoded string
        env_overrides: Additional environment variables to set
        check: If False, return empty string on error instead of raising

    Returns:
        Command output as string (stripped) or bytes

    Raises:
        GitNotInstalledError: If git is not installed
        DubiousOwnershipError: If git detects dubious ownership
        NotAGitRepoError: If not in a git repository
        GitError: For other git command failures
    """
    cmd = ["git", *args]

    # Set up environment
    env = os.environ.copy()
    env.update(_DEFAULT_ENV)
    if env_overrides:
        env.update(env_overrides)

    # Default timeout
    if timeout is None:
        timeout = 30.0

    # Log command at DEBUG level
    if logger.isEnabledFor(logging.DEBUG):
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
            return b"" if return_bytes else ""

        # Generic error
        cmd_str = " ".join(shlex.quote(str(a)) for a in args)
        raise GitError(f"git {cmd_str} failed ({result.returncode}): {stderr}")

    # Return output
    if return_bytes:
        return result.stdout
    else:
        return result.stdout.decode("utf-8", errors="surrogateescape").strip()


# ============================================================================
# Public API - No need to maintain __all__ since everything is in one file
# Functions with _ prefix are internal helpers
# ============================================================================

# GitHub constants
METTA_GITHUB_ORGANIZATION = "Metta-AI"
METTA_GITHUB_REPO = "metta"
METTA_API_REPO = f"{METTA_GITHUB_ORGANIZATION}/{METTA_GITHUB_REPO}"
METTA_API_REPO_URL = f"https://github.com/{METTA_API_REPO}.git"


def _memoize(max_age=60):
    """Simple memoization decorator with time-based expiry. (Internal helper)"""

    def decorator(func):
        cache = {}
        cache_time = {}

        @wraps(func)
        async def wrapper(*args, **kwargs):
            key = str(args) + str(kwargs)
            current_time = time.time()

            if key in cache and current_time - cache_time[key] < max_age:
                return cache[key]

            result = await func(*args, **kwargs)
            cache[key] = result
            cache_time[key] = current_time
            return result

        return wrapper

    return decorator


def run_git_with_cwd(args: list[str], cwd: str | Path | None = None) -> str:
    """Run a git command with optional working directory and return its output."""
    return run_git_cmd(args, cwd=Path(cwd) if cwd else None)


def run_git(*args: str) -> str:
    """Run a git command and return its output."""
    return run_git_cmd(list(args))


def run_git_in_dir(cwd: str | Path, *args: str) -> str:
    """Run a git command in a specific directory and return its output."""
    return run_git_cmd(list(args), cwd=Path(cwd))


def run_gh(*args: str) -> str:
    """Run a GitHub CLI command and return its output."""
    try:
        result = subprocess.run(["gh", *args], capture_output=True, text=True, check=True)
        return result.stdout.strip()
    except subprocess.CalledProcessError as e:
        raise GitError(f"GitHub CLI command failed ({e.returncode}): {e.stderr.strip()}") from e
    except FileNotFoundError as e:
        raise GitError("GitHub CLI (gh) is not installed!") from e


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


def get_commit_message(commit_hash: str) -> str:
    """Get the commit message for a given commit."""
    return run_git("log", "-1", "--pretty=%B", commit_hash)


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
        # No upstream configured ─ fallback to scanning all remotes
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


def get_matched_pr(commit_hash: str) -> tuple[int, str] | None:
    """
    Return (PR number, title) if `commit_hash` is the HEAD of an open PR, else None.
    """
    url = f"https://api.github.com/repos/{METTA_GITHUB_ORGANIZATION}/{METTA_GITHUB_REPO}/commits/{commit_hash}/pulls"
    headers = {"Accept": "application/vnd.github.groot-preview+json"}
    try:
        resp = httpx.get(url, headers=headers, timeout=5.0)
        resp.raise_for_status()
    except httpx.HTTPStatusError as e:
        if e.response.status_code == 404:
            # Commit not in repo or no PRs → treat as "no match"
            return None
        raise GitError(f"GitHub API error ({e.response.status_code}): {e.response.text}") from e
    except httpx.RequestError as e:
        # Network / timeout / DNS failure
        raise GitError(f"Network error while querying GitHub: {e}") from e
    pulls = resp.json()
    if not pulls:
        return None
    pr = pulls[0]
    return int(pr["number"]), pr["title"]


def canonical_remote_url(url: str) -> str:
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


def is_metta_ai_repo() -> bool:
    """Check if any remote is set to the metta-ai/metta repository.

    This checks all configured remotes, not just 'origin', and handles
    various URL formats (SSH, HTTPS, with/without .git suffix).
    """
    target_url = canonical_remote_url(f"https://github.com/{METTA_API_REPO}")

    remotes = get_all_remotes()
    for remote_url in remotes.values():
        if canonical_remote_url(remote_url) == target_url:
            return True

    return False


def get_git_hash_for_remote_task(
    skip_git_check: bool = False,
    skip_cmd: str = "skipping git check",
    logger: logging.Logger | None = None,
) -> str | None:
    """
    Get git hash for remote task execution.

    Returns:
        - None if no local git repo or no origin synced to metta-ai/metta
        - Git hash if commit is synced with remote and no dirty changes
        - Raises GitError if dirty changes exist and skip_git_check is False

    Args:
        skip_git_check: If True, skip the dirty changes check
        skip_cmd: The command to show in error messages for skipping the check
    """
    try:
        current_commit = get_current_commit()
    except (GitError, ValueError):
        if logger:
            logger.warning("Not in a git repository, using git_hash=None")
        return None

    if not is_metta_ai_repo():
        if logger:
            logger.warning("Origin not set to metta-ai/metta, using git_hash=None")
        return None

    on_skypilot = bool(os.getenv("SKYPILOT_TASK_ID"))
    has_changes, status_output = has_unstaged_changes()
    if has_changes:
        if logger:
            logger.warning("Working tree has unstaged changes.\n" + status_output)
        if not skip_git_check:
            if on_skypilot:
                # Skypilot jobs can create local files as part of their setup. It's assumed that these changes do not
                # need to be checked in because they wouldn't have an effect on policy evaluator's execution
                if logger:
                    logger.warning("Running on skypilot: proceeding despite unstaged changes")
            else:
                raise GitError(
                    "You have uncommitted changes to tracked files that won't be reflected in the remote task.\n"
                    f"You can push your changes or specify to skip this check with {skip_cmd}"
                )
    else:
        # Only untracked files present (or clean). Log for visibility if untracked exist.
        if status_output and logger:
            logger.info("Proceeding with unstaged changes.\n" + status_output)

    if not is_commit_pushed(current_commit) and not on_skypilot:
        short_commit = current_commit[:8]
        if not skip_git_check:
            raise GitError(
                f"Commit {short_commit} hasn't been pushed.\n"
                f"You can push your changes or specify to skip this check with {skip_cmd}"
            )
        elif logger:
            logger.warning(f"Proceeding with unpushed commit {short_commit} due to {skip_cmd}")

    return current_commit


@_memoize(max_age=60 * 5)
async def get_latest_commit(branch: str = "main") -> str:
    async with httpx.AsyncClient() as client:
        response = await client.get(
            f"https://api.github.com/repos/{METTA_API_REPO}/commits/{branch}",
            headers={"Accept": "application/vnd.github.v3+json"},
        )
        response.raise_for_status()
        commit_data = response.json()
        return commit_data["sha"]


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


# ============================================================================
# GitHub API functionality
# ============================================================================


def post_commit_status(
    commit_sha: str,
    state: str,
    repo: Optional[str] = None,
    context: str = "CI/Skypilot",
    description: Optional[str] = None,
    target_url: Optional[str] = None,
    token: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Post a status update for a commit to GitHub.

    Args:
        commit_sha: The SHA of the commit
        state: The state of the status (error, failure, pending, success)
        repo: Repository in format "owner/repo". If not provided, uses default from constants
        context: A string label to differentiate this status from others
        description: A short description of the status
        target_url: The target URL to associate with this status
        token: GitHub token. If not provided, uses GITHUB_TOKEN env var

    Returns:
        The created status object

    Raises:
        ValueError: If no token is available
        httpx.HTTPError: If the API request fails
    """
    # Use default repo if not provided
    if repo is None:
        repo = f"{METTA_GITHUB_ORGANIZATION}/{METTA_GITHUB_REPO}"

    # Get token
    github_token = token or os.environ.get("GITHUB_TOKEN")
    if not github_token:
        raise ValueError("GitHub token not provided and GITHUB_TOKEN environment variable not set")

    # Build request
    url = f"https://api.github.com/repos/{repo}/statuses/{commit_sha}"
    headers = {"Authorization": f"token {github_token}", "Accept": "application/vnd.github.v3+json"}

    data = {
        "state": state,
        "context": context,
    }

    if description:
        data["description"] = description
    if target_url:
        data["target_url"] = target_url

    # Make request
    try:
        response = httpx.post(url, headers=headers, json=data, timeout=10.0)
        response.raise_for_status()
        return response.json()
    except httpx.HTTPError as e:
        logging.error(f"Failed to post GitHub status: {e}")
        raise


# ============================================================================
# Git filter-repo functionality
# ============================================================================


def filter_repo(source_path: Path, paths: list[str]) -> Path:
    """Filter repository to only include specified paths.

    Args:
        source_path: Path to source repository
        paths: List of paths to keep (e.g., ["mettagrid/", "mettascope/"])

    Returns:
        Path to the filtered repository
    """

    if not (source_path / ".git").exists():
        raise ValueError(f"Not a git repository: {source_path}")

    # Create temporary directory
    target_dir = Path(tempfile.mkdtemp(prefix="filtered-repo-"))
    filtered_path = target_dir / "filtered"

    print("Cloning for filtering...")

    # Clone locally
    source_url = f"file://{source_path.absolute()}"
    try:
        run_git("clone", "--no-local", source_url, str(filtered_path))
    except GitError as e:
        raise RuntimeError(f"Failed to clone: {e}") from e

    # Check if git-filter-repo is available
    try:
        subprocess.run(["git", "filter-repo", "--version"], capture_output=True, text=True, check=True)
    except (subprocess.CalledProcessError, FileNotFoundError) as e:
        # Use the same installation method as metta CLI tool
        raise RuntimeError(
            "\ngit-filter-repo not found. Please install it:\n\n"
            "  metta install filter-repo\n\n"
            "Or install manually:\n"
            "  curl -O https://raw.githubusercontent.com/newren/git-filter-repo/main/git-filter-repo\n"
            "  chmod +x git-filter-repo\n"
            "  sudo mv git-filter-repo /usr/local/bin/\n"
        ) from e

    # Filter repository
    filter_cmd = ["git", "filter-repo", "--force"]
    for path in paths:
        filter_cmd.extend(["--path", path])

    print(f"Filtering to: {', '.join(paths)}")

    result = subprocess.run(filter_cmd, cwd=filtered_path, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"git-filter-repo failed: {result.stderr.strip()}")

    # Verify result
    files = get_file_list(filtered_path)

    if not files:
        raise RuntimeError("Filtered repository is empty!")

    commit_count = get_commit_count(filtered_path)
    print(f"✅ Filtered: {len(files)} files, {commit_count} commits")

    return filtered_path
