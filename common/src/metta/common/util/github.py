import logging
import os
from typing import Any, Dict, Optional

import httpx

from metta.common.util.constants import METTA_GITHUB_ORGANIZATION, METTA_GITHUB_REPO


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
