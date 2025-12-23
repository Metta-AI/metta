"""GitHub API integration for updating PR descriptions."""

import logging
import re
from typing import Optional

import httpx

from github_webhook.config import settings

logger = logging.getLogger(__name__)


async def update_pr_description_with_asana_task(
    repo_full_name: str,
    pr_number: int,
    asana_task_url: str,
    github_token: Optional[str] = None,
) -> bool:
    """
    Update PR description with Asana task link.

    Args:
        repo_full_name: Full repository name (e.g., "Metta-AI/metta")
        pr_number: Pull request number
        asana_task_url: Asana task permalink URL
        github_token: GitHub token (if None, uses GITHUB_TOKEN env var)

    Returns:
        True if update succeeded, False otherwise
    """
    if not github_token:
        github_token = settings.GITHUB_TOKEN

    if not github_token:
        logger.warning("GITHUB_TOKEN not configured - cannot update PR description")
        return False

    url = f"https://api.github.com/repos/{repo_full_name}/pulls/{pr_number}"
    headers = {
        "Authorization": f"Bearer {github_token}",
        "Accept": "application/vnd.github.v3+json",
    }

    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.get(url, headers=headers)
            if response.status_code != 200:
                logger.error(f"GitHub API Error (GET): {response.status_code} - {response.text}")
                return False

            current_body = response.json().get("body") or ""

            # Remove any existing Asana Task links (matches [Asana Task](url) format)
            # Note: If PR description format changes significantly in the future,
            # consider using a more robust approach (e.g., fenced block with sentinel comment)
            asana_link_pattern = r"\[Asana Task\]\([^)]+\)"
            body_without_asana = re.sub(asana_link_pattern, "", current_body).strip()

            new_body = f"{body_without_asana}\n\n[Asana Task]({asana_task_url})"
            payload = {"body": new_body}

            response = await client.patch(url, json=payload, headers=headers)
            if response.status_code != 200:
                logger.error(f"GitHub API Error (PATCH): {response.status_code} - {response.text}")
                return False

            logger.info(f"Updated PR #{pr_number} description with Asana task: {asana_task_url}")
            return True

    except Exception as e:
        logger.error(f"Failed to update PR description: {e}", exc_info=True)
        return False
