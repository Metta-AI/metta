#!/usr/bin/env python3
"""Manually update PR description with Asana task link."""

import asyncio
import sys

from github_webhook.asana_integration import find_task_by_github_url
from github_webhook.github_integration import update_pr_description_with_asana_task


async def main():
    """Update PR description with Asana task link."""
    if len(sys.argv) < 3:
        print("Usage: python update_pr_description.py <repo_full_name> <pr_number>")
        print("Example: python update_pr_description.py Akkikens/metta-fork 3")
        sys.exit(1)

    repo_full_name = sys.argv[1]
    pr_number = int(sys.argv[2])
    pr_url = f"https://github.com/{repo_full_name}/pull/{pr_number}"

    print(f"Looking for Asana task for PR: {pr_url}")

    # Find the task
    task = await find_task_by_github_url(pr_url)
    if not task:
        print(f"❌ No Asana task found for PR {pr_number}")
        sys.exit(1)

    task_url = task.get("permalink_url")
    if not task_url:
        print("❌ Task found but no permalink URL")
        sys.exit(1)

    print(f"✅ Found Asana task: {task_url}")

    # Update PR description
    success = await update_pr_description_with_asana_task(
        repo_full_name=repo_full_name,
        pr_number=pr_number,
        asana_task_url=task_url,
    )

    if success:
        print(f"✅ Updated PR #{pr_number} description with Asana task link")
    else:
        print("❌ Failed to update PR description. Check GITHUB_TOKEN is configured.")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())

