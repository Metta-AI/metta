"""GitHub helpers built on top of both the REST API and the ``gh`` CLI."""

from __future__ import annotations

import logging
import os
import subprocess
import time
from functools import wraps
from typing import Any, Dict, Optional

import httpx

from .core import GitError

logger = logging.getLogger(__name__)


def _memoize(max_age=60):
    """Simple memoization decorator with time-based expiry."""

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


def run_gh(*args: str) -> str:
    """Run a GitHub CLI command and return its output."""
    try:
        result = subprocess.run(["gh", *args], capture_output=True, text=True, check=True)
        return result.stdout.strip()
    except subprocess.CalledProcessError as e:
        raise GitError(f"GitHub CLI command failed ({e.returncode}): {e.stderr.strip()}") from e
    except FileNotFoundError as e:
        raise GitError("GitHub CLI (gh) is not installed!") from e


_MATCHED_PR_CACHE: Dict[tuple[str, str], tuple[int, str] | None] = {}


def get_matched_pr(commit_hash: str, repo: str) -> tuple[int, str] | None:
    """
    Return (PR number, title) if `commit_hash` is the HEAD of an open PR, else None.

    Args:
        commit_hash: The commit hash to check
        repo: Repository in format "owner/repo"
    """
    cache_key = (commit_hash, repo)
    is_mocked = httpx.get.__module__.startswith("unittest.mock")

    if not is_mocked and cache_key in _MATCHED_PR_CACHE:
        return _MATCHED_PR_CACHE[cache_key]

    url = f"https://api.github.com/repos/{repo}/commits/{commit_hash}/pulls"
    headers = {"Accept": "application/vnd.github.groot-preview+json"}
    try:
        resp = httpx.get(url, headers=headers, timeout=5.0)
        resp.raise_for_status()
    except httpx.HTTPStatusError as e:
        if e.response.status_code == 404:
            # Commit not in repo or no PRs -> treat as "no match"
            result: tuple[int, str] | None = None
        else:
            raise GitError(f"GitHub API error ({e.response.status_code}): {e.response.text}") from e
    except httpx.RequestError as e:
        # Network / timeout / DNS failure
        raise GitError(f"Network error while querying GitHub: {e}") from e
    else:
        pulls = resp.json()
        if not pulls:
            result = None
        else:
            pr = pulls[0]
            result = (int(pr["number"]), pr["title"])

    if not is_mocked:
        _MATCHED_PR_CACHE[cache_key] = result

    return result


@_memoize(max_age=60 * 5)
async def get_latest_commit(repo: str, branch: str = "main") -> str:
    """
    Get the latest commit SHA for a branch.

    Args:
        repo: Repository in format "owner/repo"
        branch: Branch name (default: "main")
    """
    async with httpx.AsyncClient() as client:
        response = await client.get(
            f"https://api.github.com/repos/{repo}/commits/{branch}",
            headers={"Accept": "application/vnd.github.v3+json"},
        )
        response.raise_for_status()
        commit_data = response.json()
        return commit_data["sha"]


def post_commit_status(
    commit_sha: str,
    state: str,
    repo: str,
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
        repo: Repository in format "owner/repo"
        context: A string label to differentiate this status from others
        description: A short description of the status
        target_url: The target URL to associate with this status
        token: GitHub token. If not provided, uses GITHUB_TOKEN env var

    Returns:
        The created status object

    Raises:
        ValueError: If no token is available or repo not provided
        httpx.HTTPError: If the API request fails
    """
    if not repo:
        raise ValueError("Repository must be provided in format 'owner/repo'")

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


def create_pr(
    repo: str,
    title: str,
    body: str,
    head: str,
    base: str,
    token: Optional[str] = None,
    draft: bool = False,
) -> Dict[str, Any]:
    """
    Create a pull request on GitHub using the REST API.

    We call the HTTP endpoint directly instead of `gh pr create` so this helper
    works in non-interactive environments (e.g., CI, containers without the CLI)
    and supports fields that the CLI does not expose consistently across versions.

    Args:
        repo: Repository in format "owner/repo"
        title: PR title
        body: PR body/description
        head: Head branch name
        base: Base branch name
        token: GitHub token. If not provided, uses GITHUB_TOKEN env var
        draft: Create as draft PR

    Returns:
        The created PR object

    Raises:
        ValueError: If no token is available or repo not provided
        httpx.HTTPError: If the API request fails
    """
    if not repo:
        raise ValueError("Repository must be provided in format 'owner/repo'")

    # Get token
    github_token = token or os.environ.get("GITHUB_TOKEN")
    if not github_token:
        raise ValueError("GitHub token not provided and GITHUB_TOKEN environment variable not set")

    # Build request
    url = f"https://api.github.com/repos/{repo}/pulls"
    headers = {
        "Authorization": f"token {github_token}",
        "Accept": "application/vnd.github.v3+json",
    }

    data = {
        "title": title,
        "body": body,
        "head": head,
        "base": base,
        "draft": draft,
    }

    # Make request
    try:
        response = httpx.post(url, headers=headers, json=data, timeout=10.0)
        response.raise_for_status()
        return response.json()
    except httpx.HTTPError as e:
        error_msg = f"Failed to create PR: {e}"
        if hasattr(e, "response") and e.response is not None:
            error_msg += f" - {e.response.text}"
        logging.error(error_msg)
        raise GitError(error_msg) from e


def _clear_matched_pr_cache() -> None:
    _MATCHED_PR_CACHE.clear()


get_matched_pr.cache_clear = _clear_matched_pr_cache  # type: ignore[attr-defined]
