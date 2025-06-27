#!/usr/bin/env -S uv run
# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "requests>=2.31.0",
# ]
# ///

import json
import logging
import os
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional

import requests


class GitHubPRDigestCreator:
    """Creates a digest of merged PRs for a given date range."""

    def __init__(self, token: str, repository: str):
        self.token = token
        self.repository = repository
        self.session = requests.Session()
        self.session.headers.update(
            {
                "Authorization": f"token {token}",
                "Accept": "application/vnd.github.v3+json",
                "User-Agent": "PR-Digest-Creator/1.0",
            }
        )

    def get_merged_prs(self, since: str, until: str) -> List[Dict]:
        """Fetch merged PRs within the date range."""
        url = f"https://api.github.com/repos/{self.repository}/pulls"

        params = {"state": "closed", "sort": "updated", "direction": "desc", "per_page": 100}

        all_prs = []
        page = 1

        while True:
            params["page"] = page
            logging.info(f"Fetching page {page} of PRs...")

            response = self.session.get(url, params=params)
            response.raise_for_status()

            prs = response.json()

            if not prs:
                break

            for pr in prs:
                # Only include merged PRs
                if not pr.get("merged_at"):
                    continue

                merged_at = datetime.fromisoformat(pr["merged_at"].replace("Z", "+00:00"))
                since_dt = datetime.fromisoformat(since.replace("Z", "+00:00"))
                until_dt = datetime.fromisoformat(until.replace("Z", "+00:00"))

                if since_dt <= merged_at <= until_dt:
                    all_prs.append(pr)
                elif merged_at < since_dt:
                    # PRs are sorted by updated date, so we can stop when we go too far back
                    logging.info(f"Reached PRs older than {since}, stopping...")
                    return all_prs

            page += 1

            # Safety check to avoid infinite loops
            if page > 50:
                logging.warning("Reached maximum page limit (50), stopping...")
                break

        return all_prs

    def get_pr_details(self, pr_number: int) -> Optional[Dict]:
        """Get detailed information for a specific PR including diff."""
        try:
            # Get basic PR info
            pr_url = f"https://api.github.com/repos/{self.repository}/pulls/{pr_number}"
            pr_response = self.session.get(pr_url)
            pr_response.raise_for_status()
            pr_data = pr_response.json()

            # Get PR diff
            diff_url = f"https://api.github.com/repos/{self.repository}/pulls/{pr_number}.diff"
            diff_response = self.session.get(diff_url)
            diff_response.raise_for_status()

            # Extract labels
            labels = [label["name"] for label in pr_data.get("labels", [])]

            return {
                "number": pr_data["number"],
                "title": pr_data["title"],
                "body": pr_data.get("body", ""),
                "author": pr_data["user"]["login"],
                "merged_at": pr_data["merged_at"],
                "html_url": pr_data["html_url"],
                "labels": labels,
                "diff": diff_response.text,
                "additions": pr_data.get("additions", 0),
                "deletions": pr_data.get("deletions", 0),
                "changed_files": pr_data.get("changed_files", 0),
            }

        except Exception as e:
            logging.error(f"Error fetching details for PR #{pr_number}: {e}")
            return None

    def create_digest(self, since: str, until: str) -> List[Dict]:
        """Create a complete digest of PRs with full details."""
        logging.info(f"Creating PR digest for {self.repository} from {since} to {until}")

        # Get list of merged PRs
        merged_prs = self.get_merged_prs(since, until)
        logging.info(f"Found {len(merged_prs)} merged PRs in date range")

        # Get detailed information for each PR
        detailed_prs = []
        for i, pr in enumerate(merged_prs):
            logging.info(f"Fetching details for PR {i + 1}/{len(merged_prs)}: #{pr['number']}")

            details = self.get_pr_details(pr["number"])
            if details:
                detailed_prs.append(details)

        logging.info(f"Successfully collected details for {len(detailed_prs)} PRs")
        return detailed_prs


def parse_date_range(date_range_str: str) -> tuple[str, str]:
    """Parse date range string into since/until dates."""
    if not date_range_str:
        # Default to last 7 days
        until = datetime.now()
        since = until - timedelta(days=7)
        return since.isoformat() + "Z", until.isoformat() + "Z"

    if " to " in date_range_str:
        since_str, until_str = date_range_str.split(" to ")
        since = datetime.fromisoformat(since_str.strip())
        until = datetime.fromisoformat(until_str.strip())
    else:
        # Single date - treat as "since this date"
        since = datetime.fromisoformat(date_range_str.strip())
        until = datetime.now()

    return since.isoformat() + "Z", until.isoformat() + "Z"


def main():
    """Main entry point."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

    # Get configuration from environment
    github_token = os.getenv("GITHUB_TOKEN")
    github_repository = os.getenv("GITHUB_REPOSITORY")
    date_range = os.getenv("DATE_RANGE", "")
    output_file = os.getenv("PR_DIGEST_FILE", "pr_digest_output.json")

    if not github_token:
        print("Error: GITHUB_TOKEN not provided")
        sys.exit(1)

    if not github_repository:
        print("Error: GITHUB_REPOSITORY not provided")
        sys.exit(1)

    # Parse date range
    since, until = parse_date_range(date_range)
    logging.info(f"Date range: {since} to {until}")

    # Create digest
    creator = GitHubPRDigestCreator(github_token, github_repository)

    try:
        digest = creator.create_digest(since, until)

        # Save digest to file
        output_path = Path(output_file)
        with open(output_path, "w") as f:
            json.dump(digest, f, indent=2)

        print(f"✅ Created PR digest with {len(digest)} PRs")
        print(f"✅ Saved to {output_path}")

        # Set output for GitHub Actions
        if os.getenv("GITHUB_ACTIONS"):
            with open(os.environ["GITHUB_OUTPUT"], "a") as f:
                f.write(f"pr_count={len(digest)}\n")
                f.write(f"digest_file={output_path}\n")

    except Exception as e:
        logging.error(f"Failed to create PR digest: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
