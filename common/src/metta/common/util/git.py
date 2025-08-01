import logging
import subprocess
from pathlib import Path

import httpx

from metta.common.util.constants import METTA_GITHUB_ORGANIZATION, METTA_GITHUB_REPO
from metta.common.util.memoization import memoize


class GitError(Exception):
    """Custom exception for git-related errors."""


METTA_API_REPO = f"{METTA_GITHUB_ORGANIZATION}/{METTA_GITHUB_REPO}"
METTA_API_REPO_URL = f"https://github.com/{METTA_API_REPO}.git"


def run_git_with_cwd(args: list[str], cwd: str | Path | None = None) -> str:
    """Run a git command with optional working directory and return its output."""
    try:
        result = subprocess.run(
            ["git", *args], capture_output=True, text=True, check=True, cwd=str(cwd) if cwd else None
        )
        return result.stdout.strip()
    except subprocess.CalledProcessError as e:
        raise GitError(f"Git command failed ({e.returncode}): {e.stderr.strip()}") from e
    except FileNotFoundError as e:
        raise GitError("Git is not installed!") from e


def run_git(*args: str) -> str:
    """Run a git command and return its output."""
    return run_git_with_cwd(list(args))


def run_git_in_dir(cwd: str | Path, *args: str) -> str:
    """Run a git command in a specific directory and return its output."""
    return run_git_with_cwd(list(args), cwd)


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
        if "not a git repository" in str(e):
            raise ValueError("Not in a git repository") from e
        elif "HEAD is not a symbolic ref" in str(e):
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


def has_unstaged_changes() -> bool:
    """Check if there are any unstaged changes."""
    status_output = run_git("status", "--porcelain")
    return bool(status_output)


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
    """
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
            # Commit not in repo or no PRs → treat as “no match”
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


def get_remote_url() -> str | None:
    """Get the URL of the origin remote repository."""
    try:
        return run_git("remote", "get-url", "origin")
    except GitError:
        return None


def is_metta_ai_repo() -> bool:
    """Check if the origin remote is set to metta-ai/metta repository."""
    remote_url = get_remote_url()
    if not remote_url:
        return False
    return remote_url in (f"git@github.com:{METTA_API_REPO}.git", f"https://github.com/{METTA_API_REPO}.git")


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

    if has_unstaged_changes():
        if not skip_git_check:
            raise GitError(
                f"You have uncommitted changes that won't be reflected in the remote task.\n"
                f"You can push your changes or specify to skip this check with {skip_cmd}"
            )
        elif logger:
            logger.info("Proceeding with uncommitted changes")

    if not is_commit_pushed(current_commit):
        short_commit = current_commit[:8]
        if not skip_git_check:
            raise GitError(
                f"Commit {short_commit} hasn't been pushed.\n"
                f"You can push your changes or specify to skip this check with {skip_cmd}"
            )
        elif logger:
            logger.warning(f"Proceeding with unpushed commit {short_commit} due to {skip_cmd}")

    return current_commit


@memoize(max_age=60 * 5)
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
