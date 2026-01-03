#!/usr/bin/env python3
"""Manually trigger opened event for a PR."""

import asyncio
import json
import sys

from github_webhook.pr_handler import handle_pull_request_event

async def main():
    """Trigger opened event."""
    if len(sys.argv) < 3:
        print("Usage: python trigger_opened_event.py <repo_full_name> <pr_number>")
        print("Example: python trigger_opened_event.py Akkikens/metta-fork 3")
        sys.exit(1)

    repo_full_name = sys.argv[1]
    pr_number = int(sys.argv[2])
    
    # Fetch PR data from GitHub API (simplified - you'd need GITHUB_TOKEN)
    # For now, create a minimal payload
    payload = {
        "action": "opened",
        "number": pr_number,
        "pull_request": {
            "number": pr_number,
            "title": f"PR #{pr_number}",
            "html_url": f"https://github.com/{repo_full_name}/pull/{pr_number}",
            "user": {"login": "Akkikens"},
            "assignee": None,
        },
        "repository": {
            "full_name": repo_full_name,
        },
    }
    
    print(f"Triggering 'opened' event for PR #{pr_number}...")
    result = await handle_pull_request_event(payload, delivery_id="manual-trigger")
    print(f"Result: {json.dumps(result, indent=2)}")

if __name__ == "__main__":
    asyncio.run(main())
