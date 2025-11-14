#!/usr/bin/env -S uv run
"""Show PRs that touch a specific path."""

import json
import subprocess
import sys
from datetime import datetime

def main():
    if len(sys.argv) < 2:
        print("Usage: tools/pr_feed.py <path> [--status=open|closed|merged|all] [--num_prs=50]")
        print("Example: tools/pr_feed.py metta/jobs")
        sys.exit(1)

    path = sys.argv[1]
    status = "open"
    num_prs = 50

    # Parse optional arguments
    for arg in sys.argv[2:]:
        if arg.startswith("--status="):
            status = arg.split("=")[1]
        elif arg.startswith("--num_prs="):
            num_prs = int(arg.split("=")[1])

    # Validate status
    valid_statuses = {"open": "OPEN", "closed": "CLOSED", "merged": "MERGED", "all": "OPEN, CLOSED, MERGED"}
    if status not in valid_statuses:
        print(f"Invalid status: {status}. Choose from: open, closed, merged, all")
        sys.exit(1)

    states = valid_statuses[status]
    status_display = status.upper()

    print(f"🔍 Searching for {status_display} PRs touching: {path}\n")

    # Run GraphQL query
    query = f"""
    query {{
      repository(owner: "Metta-AI", name: "metta") {{
        pullRequests(first: {num_prs}, states: [{states}], orderBy: {{field: UPDATED_AT, direction: DESC}}) {{
          nodes {{
            number
            title
            url
            author {{ login }}
            updatedAt
            files(first: 100) {{
              nodes {{ path }}
            }}
          }}
        }}
      }}
    }}
    """

    try:
        result = subprocess.run(
            ["gh", "api", "graphql", "-f", f"query={query}"],
            capture_output=True,
            text=True,
            check=True,
        )

        data = json.loads(result.stdout)
        prs = data["data"]["repository"]["pullRequests"]["nodes"]

        # Filter PRs that touch the specified path
        matching_prs = []
        for pr in prs:
            if any(file["path"].startswith(path) for file in pr["files"]["nodes"]):
                matching_prs.append(pr)

        if not matching_prs:
            print(f"No {status_display} PRs found touching {path}")
            return

        # Display results
        for pr in matching_prs:
            updated = datetime.fromisoformat(pr['updatedAt'].replace('Z', '+00:00'))
            updated_str = updated.strftime('%Y-%m-%d')

            print(f"PR #{pr['number']}: {pr['title']}")
            print(f"  Author: @{pr['author']['login']} • Updated: {updated_str}")
            print(f"  {pr['url']}\n")

        print(f"✅ Found {len(matching_prs)} {status_display} PR(s)")

    except subprocess.CalledProcessError as e:
        print(f"Failed to fetch PRs: {e.stderr}")
        sys.exit(1)
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
