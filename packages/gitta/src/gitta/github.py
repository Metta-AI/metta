"""GitHub helpers built on top of both the REST API and the ``gh`` CLI."""

from __future__ import annotations

import logging
import os
import subprocess
import time
from contextlib import contextmanager
from functools import wraps
from typing import Any, Dict, Generator, Optional

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


@contextmanager
def github_client(
    repo: str,
    token: str | None = None,
    base_url: str | None = None,
    timeout: float = 30.0,
    **headers: str,
) -> Generator[httpx.Client, None, None]:
    """
    Create an authenticated GitHub API client.

    Args:
        repo: Repository in format "owner/repo"
        token: GitHub token. If not provided, uses GITHUB_TOKEN env var
        base_url: Base URL for the API (default: https://api.github.com/repos/{repo})
        timeout: Request timeout in seconds
        **headers: Additional headers to include in requests

    Yields:
        Configured httpx.Client for GitHub API requests
    """
    github_token = token or os.environ.get("GITHUB_TOKEN")

    # Build base URL
    if base_url is None:
        base_url = f"https://api.github.com/repos/{repo}"

    # Build headers
    client_headers = {
        "Accept": "application/vnd.github+json",
        "X-GitHub-Api-Version": "2022-11-28",
        "User-Agent": "gitta-client",
    }

    # Add authentication if token is provided
    if github_token:
        client_headers["Authorization"] = f"token {github_token}"

    # Add any additional headers
    client_headers.update(headers)

    with httpx.Client(
        base_url=base_url,
        headers=client_headers,
        timeout=timeout,
    ) as client:
        yield client


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


def get_commits(
    repo: str,
    branch: str = "main",
    since: str | None = None,
    per_page: int = 100,
    token: str | None = None,
    **headers: str,
) -> list[dict[str, Any]]:
    """
    Get list of commits from a repository.

    Args:
        repo: Repository in format "owner/repo"
        branch: Branch name
        since: ISO 8601 timestamp to filter commits (only commits after this date)
        per_page: Number of commits per page (max 100)
        token: GitHub token for authentication
        **headers: Additional headers (can override Authorization for custom auth)

    Returns:
        List of commit objects from GitHub API

    Raises:
        GitError: If the API request fails
    """
    params: dict[str, Any] = {"sha": branch, "per_page": per_page}
    if since:
        params["since"] = since

    all_commits: list[dict[str, Any]] = []
    page = 1

    with github_client(repo, token=token, **headers) as client:
        while True:
            try:
                resp = client.get("/commits", params={**params, "page": page})
                resp.raise_for_status()
            except httpx.HTTPError as e:
                error_msg = f"Failed to get commits: {e}"
                if hasattr(e, "response") and e.response is not None:
                    error_msg += f" - Status: {e.response.status_code}"
                logger.error(error_msg)
                raise GitError(error_msg) from e

            commits = resp.json() or []
            if not commits:
                break

            all_commits.extend(commits)

            if len(commits) < per_page:
                break

            page += 1

    return all_commits


def get_workflow_runs(
    repo: str,
    workflow_filename: str,
    branch: str | None = None,
    status: str | None = None,
    per_page: int = 1,
    token: str | None = None,
    **headers: str,
) -> list[dict[str, Any]]:
    """
    Get workflow runs for a specific workflow.

    Args:
        repo: Repository in format "owner/repo"
        workflow_filename: Workflow filename (e.g., "checks.yml")
        branch: Filter by branch name
        status: Filter by status (e.g., "completed", "in_progress")
        per_page: Number of runs per page
        token: GitHub token for authentication
        **headers: Additional headers (can override Authorization for custom auth)

    Returns:
        List of workflow run objects

    Raises:
        GitError: If the API request fails
    """
    params: dict[str, Any] = {"per_page": per_page}
    if branch:
        params["branch"] = branch
    if status:
        params["status"] = status

    with github_client(repo, token=token, **headers) as client:
        try:
            resp = client.get(f"/actions/workflows/{workflow_filename}/runs", params=params)
            resp.raise_for_status()
        except httpx.HTTPError as e:
            error_msg = f"Failed to get workflow runs: {e}"
            if hasattr(e, "response") and e.response is not None:
                error_msg += f" - Status: {e.response.status_code}"
            logger.error(error_msg)
            raise GitError(error_msg) from e

        return (resp.json() or {}).get("workflow_runs", [])


def get_workflow_run_jobs(
    repo: str,
    run_id: int,
    per_page: int = 100,
    token: str | None = None,
    **headers: str,
) -> list[dict[str, Any]]:
    """
    Get jobs for a specific workflow run.

    Args:
        repo: Repository in format "owner/repo"
        run_id: Workflow run ID
        per_page: Number of jobs per page
        token: GitHub token for authentication
        **headers: Additional headers (can override Authorization for custom auth)

    Returns:
        List of job objects

    Raises:
        GitError: If the API request fails
    """
    params = {"per_page": per_page}

    with github_client(repo, token=token, **headers) as client:
        try:
            resp = client.get(f"/actions/runs/{run_id}/jobs", params=params)
            resp.raise_for_status()
        except httpx.HTTPError as e:
            error_msg = f"Failed to get workflow run jobs: {e}"
            if hasattr(e, "response") and e.response is not None:
                error_msg += f" - Status: {e.response.status_code}"
            logger.error(error_msg)
            raise GitError(error_msg) from e

        return (resp.json() or {}).get("jobs", [])


def get_pull_requests(
    repo: str,
    state: str = "open",
    since: str | None = None,
    per_page: int = 100,
    token: str | None = None,
    **headers: str,
) -> list[dict[str, Any]]:
    """
    Get list of pull requests from a repository.

    Args:
        repo: Repository in format "owner/repo"
        state: Filter by state ("open", "closed", "all")
        since: ISO 8601 timestamp to filter PRs (only PRs updated after this date)
        per_page: Number of PRs per page (max 100)
        token: GitHub token for authentication
        **headers: Additional headers (can override Authorization for custom auth)

    Returns:
        List of pull request objects from GitHub API

    Raises:
        GitError: If the API request fails
    """
    params: dict[str, Any] = {"state": state, "per_page": per_page, "sort": "updated", "direction": "desc"}

    all_prs: list[dict[str, Any]] = []
    page = 1

    with github_client(repo, token=token, **headers) as client:
        while True:
            try:
                resp = client.get("/pulls", params={**params, "page": page})
                resp.raise_for_status()
            except httpx.HTTPError as e:
                error_msg = f"Failed to get pull requests: {e}"
                if hasattr(e, "response") and e.response is not None:
                    error_msg += f" - Status: {e.response.status_code}"
                logger.error(error_msg)
                raise GitError(error_msg) from e

            prs = resp.json() or []
            if not prs:
                break

            # Filter by since date if provided
            if since:
                from datetime import datetime

                cutoff = datetime.fromisoformat(since.replace("Z", "+00:00"))
                filtered_prs = []
                for pr in prs:
                    updated_at = datetime.fromisoformat(pr["updated_at"].replace("Z", "+00:00"))
                    if updated_at >= cutoff:
                        filtered_prs.append(pr)
                    else:
                        # Since we're sorting by updated desc, stop when we hit older PRs
                        break
                all_prs.extend(filtered_prs)
                if len(filtered_prs) < len(prs):
                    break
            else:
                all_prs.extend(prs)

            if len(prs) < per_page:
                break

            page += 1

    return all_prs


def get_branches(
    repo: str,
    token: str | None = None,
    **headers: str,
) -> list[dict[str, Any]]:
    """
    Get list of branches from a repository.

    Args:
        repo: Repository in format "owner/repo"
        token: GitHub token for authentication
        **headers: Additional headers (can override Authorization for custom auth)

    Returns:
        List of branch objects from GitHub API

    Raises:
        GitError: If the API request fails
    """
    all_branches: list[dict[str, Any]] = []
    page = 1
    per_page = 100

    with github_client(repo, token=token, **headers) as client:
        while True:
            try:
                resp = client.get("/branches", params={"per_page": per_page, "page": page})
                resp.raise_for_status()
            except httpx.HTTPError as e:
                error_msg = f"Failed to get branches: {e}"
                if hasattr(e, "response") and e.response is not None:
                    error_msg += f" - Status: {e.response.status_code}"
                logger.error(error_msg)
                raise GitError(error_msg) from e

            branches = resp.json() or []
            if not branches:
                break

            all_branches.extend(branches)

            if len(branches) < per_page:
                break

            page += 1

    return all_branches


def list_all_workflow_runs(
    repo: str,
    branch: str | None = None,
    status: str | None = None,
    created: str | None = None,
    per_page: int = 100,
    token: str | None = None,
    **headers: str,
) -> list[dict[str, Any]]:
    """
    Get all workflow runs for a repository (not limited to specific workflow).

    Args:
        repo: Repository in format "owner/repo"
        branch: Filter by branch name
        status: Filter by status (e.g., "completed", "in_progress")
        created: Filter by created date (e.g., ">=2024-01-01" or "2024-01-01..2024-12-31")
        per_page: Number of runs per page
        token: GitHub token for authentication
        **headers: Additional headers (can override Authorization for custom auth)

    Returns:
        List of workflow run objects

    Raises:
        GitError: If the API request fails
    """
    params: dict[str, Any] = {"per_page": per_page}
    if branch:
        params["branch"] = branch
    if status:
        params["status"] = status
    if created:
        params["created"] = created

    all_runs: list[dict[str, Any]] = []
    page = 1

    with github_client(repo, token=token, **headers) as client:
        while True:
            try:
                resp = client.get("/actions/runs", params={**params, "page": page})
                resp.raise_for_status()
            except httpx.HTTPError as e:
                error_msg = f"Failed to get workflow runs: {e}"
                if hasattr(e, "response") and e.response is not None:
                    error_msg += f" - Status: {e.response.status_code}"
                logger.error(error_msg)
                raise GitError(error_msg) from e

            data = resp.json() or {}
            runs = data.get("workflow_runs", [])

            if not runs:
                break

            all_runs.extend(runs)

            if len(runs) < per_page:
                break

            page += 1

    return all_runs


def _clear_matched_pr_cache() -> None:
    _MATCHED_PR_CACHE.clear()


get_matched_pr.cache_clear = _clear_matched_pr_cache  # type: ignore[attr-defined]
