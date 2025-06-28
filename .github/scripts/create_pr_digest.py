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
from typing import Dict, List, Optional, Set

import requests


class PRDigestCreator:
    """Creates a digest of merged PRs, checking cache before fetching details."""

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
        self.summaries_dir = Path("pr-summaries")

    def get_cached_pr_numbers(self) -> Set[int]:
        """Get set of PR numbers that already have summaries."""
        cached_prs = set()

        # Check for individual PR summary files
        if self.summaries_dir.exists():
            for pr_file in self.summaries_dir.glob("pr_*.txt"):
                try:
                    # Extract PR number from filename (pr_1234.txt)
                    pr_number = int(pr_file.stem.split("_")[1])
                    cached_prs.add(pr_number)
                except (ValueError, IndexError):
                    logging.warning(f"Couldn't parse PR number from {pr_file}")

            logging.info(f"Found {len(cached_prs)} PR summary files in {self.summaries_dir}/")
        else:
            logging.info(f"PR summaries directory {self.summaries_dir}/ does not exist yet")

        return cached_prs

    def get_merged_prs(self, since: str, until: str) -> List[Dict]:
        """Fetch merged PRs within the date range (minimal data)."""
        url = f"https://api.github.com/repos/{self.repository}/pulls"

        params = {"state": "closed", "sort": "updated", "direction": "desc", "per_page": 100}

        all_prs = []
        page = 1
        consecutive_old_prs = 0
        old_pr_threshold = 20  # Stop after seeing 20 consecutive old PRs

        while True:
            params["page"] = page
            logging.info(f"Fetching page {page} of PRs...")

            response = self.session.get(url, params=params)
            response.raise_for_status()

            prs = response.json()

            if not prs:
                break

            page_found_recent = False
            for pr in prs:
                # Only include merged PRs
                if not pr.get("merged_at"):
                    continue

                merged_at = datetime.fromisoformat(pr["merged_at"].replace("Z", "+00:00"))
                since_dt = datetime.fromisoformat(since.replace("Z", "+00:00"))
                until_dt = datetime.fromisoformat(until.replace("Z", "+00:00"))

                if since_dt <= merged_at <= until_dt:
                    # Store minimal data for cache checking
                    all_prs.append(
                        {
                            "number": pr["number"],
                            "title": pr["title"],
                            "merged_at": pr["merged_at"],
                            "html_url": pr["html_url"],
                            "author": pr["user"]["login"],
                        }
                    )
                    consecutive_old_prs = 0  # Reset counter when we find a recent PR
                    page_found_recent = True
                elif merged_at < since_dt:
                    consecutive_old_prs += 1

            # Log progress
            if page_found_recent:
                logging.info(f"Page {page}: Found recent PRs, continuing (consecutive old count reset)")
            else:
                logging.info(f"Page {page}: No recent PRs found (consecutive old: {consecutive_old_prs})")

            # Stop if we've seen too many consecutive old PRs
            if consecutive_old_prs >= old_pr_threshold:
                logging.info(f"Stopping after finding {consecutive_old_prs} consecutive PRs older than {since}")
                break

            page += 1

            # Safety check to avoid infinite loops
            if page > 50:
                logging.warning("Reached maximum page limit (50), stopping...")
                break

        logging.info(f"Found {len(all_prs)} merged PRs in date range")
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
        """Create a complete digest of PRs with full details, skipping cached ones."""
        logging.info(f"Creating PR digest for {self.repository} from {since} to {until}")

        # Step 1: Get cached PR numbers
        cached_pr_numbers = self.get_cached_pr_numbers()

        # Step 2: Get list of merged PRs (minimal data)
        all_merged_prs = self.get_merged_prs(since, until)
        logging.info(f"Found {len(all_merged_prs)} merged PRs in date range")

        # Step 3: Filter out cached PRs
        prs_to_fetch = []
        skipped_prs = []

        for pr in all_merged_prs:
            if pr["number"] in cached_pr_numbers:
                skipped_prs.append(pr)
            else:
                prs_to_fetch.append(pr)

        logging.info(f"Cache check: {len(skipped_prs)} already cached, {len(prs_to_fetch)} need fetching")

        if skipped_prs:
            logging.info("Skipped cached PRs:")
            for pr in skipped_prs[:10]:  # Show first 10
                logging.info(f"  - PR #{pr['number']}: {pr['title']}")
            if len(skipped_prs) > 10:
                logging.info(f"  ... and {len(skipped_prs) - 10} more")

        # Step 4: Get detailed information only for uncached PRs
        detailed_prs = []
        for i, pr in enumerate(prs_to_fetch):
            logging.info(f"Fetching details for PR {i + 1}/{len(prs_to_fetch)}: #{pr['number']}")

            details = self.get_pr_details(pr["number"])
            if details:
                detailed_prs.append(details)

        # Log summary
        logging.info(f"""
=== PR Digest Summary ===
Total merged PRs in range: {len(all_merged_prs)}
Already cached (skipped): {len(skipped_prs)}
Newly fetched: {len(detailed_prs)}
API calls saved: {len(skipped_prs) * 2} (1 for PR details + 1 for diff each)
========================
        """)

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
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

    # Define required environment variables
    required_env_vars = {
        "GITHUB_TOKEN": "GitHub Token",
        "GITHUB_REPOSITORY": "GitHub repository name",
        "DAYS_TO_SCAN": "Days to process",
    }

    # Validate required environment variables
    env_values = {}
    missing_vars = []

    for var_name, description in required_env_vars.items():
        value = os.getenv(var_name)
        if not value:
            missing_vars.append(f"{var_name} ({description})")
        else:
            env_values[var_name] = value

    # Report all missing variables at once
    if missing_vars:
        print("Error: Missing required environment variables:")
        for var in missing_vars:
            print(f"  - {var}")
        sys.exit(1)

    # Get configuration from environment
    github_token = env_values["GITHUB_TOKEN"]
    github_repository = env_values["GITHUB_REPOSITORY"]
    days_to_scan = env_values["DAYS_TO_SCAN"]
    output_file = os.getenv("PR_DIGEST_FILE", "pr_digest_output.json")

    days = int(days_to_scan)
    until = datetime.now()
    since = until - timedelta(days=days)
    since, until = since.isoformat() + "Z", until.isoformat() + "Z"

    logging.info(f"Date range: {since} to {until}")

    creator = PRDigestCreator(github_token, github_repository)

    try:
        digest = creator.create_digest(since, until)

        # Save digest to file
        output_path = Path(output_file)
        with open(output_path, "w") as f:
            json.dump(digest, f, indent=2)

        print(f"✅ Created PR digest with {len(digest)} NEW PRs (after cache check)")
        print(f"✅ Saved to {output_path}")

        # Handle case where all PRs are cached
        if len(digest) == 0:
            print("ℹ️  All PRs in the date range are already cached!")
            print("ℹ️  Creating minimal outputs to prevent workflow failure...")

            # Create minimal output files
            with open("discord_summary_output.txt", "w") as f:
                f.write(f"📊 **PR Summary Report** • Last {days} days\n\n")
                f.write(
                    "ℹ️ No new PRs to report - all changes in this period were already summarized "
                    + "in previous newsletters.\n"
                )

            with open("collection_summary_output.txt", "w") as f:
                f.write("No new PRs to summarize in this period.")

            # Create empty but valid JSON for pr_summary_data
            with open("pr_summary_data.json", "w") as f:
                json.dump([], f)

        # Set output for GitHub Actions
        if os.getenv("GITHUB_ACTIONS"):
            with open(os.environ["GITHUB_OUTPUT"], "a") as f:
                f.write(f"pr_count={len(digest)}\n")
                f.write(f"digest_file={output_path}\n")
                f.write(f"has_new_prs={'true' if digest else 'false'}\n")

    except Exception as e:
        logging.error(f"Failed to create PR digest: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
