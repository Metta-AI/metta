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
from typing import Dict, List, Optional, Tuple
from zoneinfo import ZoneInfo

import requests


class CachedPRSummary:
    """Represents a PR summary loaded from cache."""

    def __init__(self, pr_number: int, summary_file: Path):
        self.pr_number = pr_number
        self.summary_file = summary_file
        self._metadata = None

    def load_metadata(self) -> Optional[Dict]:
        """Load and parse metadata from the cached summary file."""
        if self._metadata is not None:
            return self._metadata

        try:
            content = self.summary_file.read_text(encoding="utf-8")
            lines = content.split("\n")

            # Extract metadata from the formatted summary
            metadata = {"number": self.pr_number, "_from_cache": True}

            # Parse the header section
            for _i, line in enumerate(lines[:20]):  # Check first 20 lines
                if line.startswith("PR #"):
                    # Format: "PR #123: Title"
                    parts = line.split(": ", 1)
                    if len(parts) > 1:
                        metadata["title"] = parts[1]
                elif line.startswith("Author: "):
                    metadata["author"] = line.replace("Author: ", "").strip()
                elif line.startswith("Merged: "):
                    metadata["merged_at"] = line.replace("Merged: ", "").strip()
                elif line.startswith("GitHub: "):
                    metadata["html_url"] = line.replace("GitHub: ", "").strip()
                elif line.startswith("Category: "):
                    metadata["category"] = line.replace("Category: ", "").strip()
                elif line.startswith("Impact: "):
                    metadata["impact_level"] = line.replace("Impact: ", "").strip()

            self._metadata = metadata
            return metadata

        except Exception as e:
            logging.error(f"Error loading metadata from {self.summary_file}: {e}")
            return None

    def is_in_date_range(self, since: datetime, until: datetime) -> bool:
        """Check if this PR was merged within the given date range."""
        metadata = self.load_metadata()
        if not metadata or not metadata.get("merged_at"):
            return False

        try:
            merged_at = datetime.fromisoformat(metadata["merged_at"].replace("Z", "+00:00"))
            return since <= merged_at <= until
        except Exception as e:
            logging.error(f"Error parsing merge date for PR #{self.pr_number}: {e}")
            return False


class PRDigestCreator:
    """Creates a digest of merged PRs, utilizing cache effectively."""

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

    def load_all_cached_summaries(self) -> Dict[int, CachedPRSummary]:
        """Load all cached PR summaries from disk."""
        cached_summaries = {}

        if self.summaries_dir.exists():
            for pr_file in self.summaries_dir.glob("pr_*.txt"):
                try:
                    # Extract PR number from filename (pr_1234.txt)
                    pr_number = int(pr_file.stem.split("_")[1])
                    cached_summaries[pr_number] = CachedPRSummary(pr_number, pr_file)
                except (ValueError, IndexError):
                    logging.warning(f"Couldn't parse PR number from {pr_file}")

            logging.info(f"Loaded {len(cached_summaries)} cached PR summaries from {self.summaries_dir}/")
        else:
            logging.info(f"PR summaries directory {self.summaries_dir}/ does not exist yet")

        return cached_summaries

    def get_merged_prs_in_range(self, since: datetime, until: datetime) -> List[Dict]:
        """Fetch list of merged PRs within the date range from GitHub API."""
        url = f"https://api.github.com/repos/{self.repository}/pulls"
        params = {"state": "closed", "sort": "updated", "direction": "desc", "per_page": 100}

        all_prs = []
        page = 1
        consecutive_old_prs = 0
        old_pr_threshold = 20

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
                if not pr.get("merged_at"):
                    continue

                merged_at = datetime.fromisoformat(pr["merged_at"].replace("Z", "+00:00"))

                if since <= merged_at <= until:
                    all_prs.append(
                        {
                            "number": pr["number"],
                            "title": pr["title"],
                            "merged_at": pr["merged_at"],
                            "html_url": pr["html_url"],
                            "author": pr["user"]["login"],
                        }
                    )
                    consecutive_old_prs = 0
                    page_found_recent = True
                elif merged_at < since:
                    consecutive_old_prs += 1

            if not page_found_recent:
                logging.info(f"Page {page}: No recent PRs found (consecutive old: {consecutive_old_prs})")

            if consecutive_old_prs >= old_pr_threshold:
                logging.info(f"Stopping after finding {consecutive_old_prs} consecutive PRs older than cutoff")
                break

            page += 1
            if page > 50:
                logging.warning("Reached maximum page limit (50), stopping...")
                break

        logging.info(f"Found {len(all_prs)} merged PRs in date range from GitHub API")
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

    def create_digest(self, since_str: str, until_str: str) -> Tuple[List[Dict], Dict[str, int]]:
        """Create a complete digest of PRs for the time period.

        Returns:
            - List of PR details (for PRs that need to be processed)
            - Statistics dictionary
        """
        since = datetime.fromisoformat(since_str.replace("Z", "+00:00"))
        until = datetime.fromisoformat(until_str.replace("Z", "+00:00"))

        logging.info(f"Creating PR digest for {self.repository} from {since_str} to {until_str}")

        # Step 1: Load all cached summaries
        cached_summaries = self.load_all_cached_summaries()

        # Step 2: Get list of PRs in date range from GitHub
        prs_from_github = self.get_merged_prs_in_range(since, until)
        github_pr_numbers = {pr["number"] for pr in prs_from_github}

        # Step 3: Find cached PRs that are in our date range
        cached_prs_in_range = []
        for pr_num, cached_summary in cached_summaries.items():
            if cached_summary.is_in_date_range(since, until):
                metadata = cached_summary.load_metadata()
                if metadata:
                    cached_prs_in_range.append(pr_num)

        # Step 4: Determine what needs to be fetched
        # PRs to fetch = (PRs from GitHub in range) - (Cached PRs in range)
        prs_to_fetch = []
        for pr in prs_from_github:
            if pr["number"] not in cached_summaries:
                prs_to_fetch.append(pr)

        # Step 5: Also check for cached PRs that GitHub didn't return (edge case)
        # This could happen if a PR was updated after our cutoff but merged within range
        additional_cached_prs = set(cached_prs_in_range) - github_pr_numbers
        if additional_cached_prs:
            logging.info(f"Found {len(additional_cached_prs)} additional PRs in cache not returned by GitHub API")

        # Calculate statistics
        stats = {
            "total_prs_in_range": len(github_pr_numbers) + len(additional_cached_prs),
            "cached_prs_in_range": len(cached_prs_in_range),
            "new_prs_to_fetch": len(prs_to_fetch),
            "github_api_returned": len(prs_from_github),
            "additional_from_cache": len(additional_cached_prs),
            "cached_pr_numbers": list(cached_prs_in_range),  # Include list of cached PR numbers
        }

        logging.info(f"""
=== PR Digest Statistics ===
Total PRs in date range: {stats["total_prs_in_range"]}
  - From GitHub API: {stats["github_api_returned"]}
  - Additional from cache: {stats["additional_from_cache"]}
Already summarized (cached): {stats["cached_prs_in_range"]}
Need to fetch and summarize: {stats["new_prs_to_fetch"]}
===========================
        """)

        # Step 6: Fetch details only for PRs that need processing
        prs_to_process = []
        for i, pr in enumerate(prs_to_fetch):
            logging.info(f"Fetching details for PR {i + 1}/{len(prs_to_fetch)}: #{pr['number']}")
            details = self.get_pr_details(pr["number"])
            if details:
                prs_to_process.append(details)

        return prs_to_process, stats


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
    until_date = datetime.now()
    since_date = until_date - timedelta(days=days)
    since = since_date.isoformat() + "Z"
    until = until_date.isoformat() + "Z"

    logging.info(f"Date range: {since} to {until}")

    creator = PRDigestCreator(github_token, github_repository)

    try:
        prs_to_process, stats = creator.create_digest(since, until)

        # Save PRs that need processing to file
        output_path = Path(output_file)
        with open(output_path, "w") as f:
            json.dump(prs_to_process, f, indent=2)

        print("✅ PR Digest created successfully")
        print(f"   - Total PRs in period: {stats['total_prs_in_range']}")
        print(f"   - Already summarized: {stats['cached_prs_in_range']}")
        print(f"   - New PRs to process: {stats['new_prs_to_fetch']}")
        print(f"✅ Saved {len(prs_to_process)} PRs to process in {output_path}")

        # Save statistics for the workflow
        stats_file = Path("pr_digest_stats.json")
        with open(stats_file, "w") as f:
            json.dump(stats, f, indent=2)

        # Set output for GitHub Actions
        if os.getenv("GITHUB_ACTIONS"):
            with open(os.environ["GITHUB_OUTPUT"], "a") as f:
                f.write(f"new_pr_count={stats['new_prs_to_fetch']}\n")
                f.write(f"total_pr_count={stats['total_prs_in_range']}\n")
                f.write(f"cached_pr_count={stats['cached_prs_in_range']}\n")
                f.write(f"digest_file={output_path}\n")
                f.write(f"stats_file={stats_file}\n")
                f.write(f"has_prs_in_range={'true' if stats['total_prs_in_range'] > 0 else 'false'}\n")
                f.write(f"has_new_prs={'true' if stats['new_prs_to_fetch'] > 0 else 'false'}\n")

                # Convert to PST (note: this handles daylight saving automatically)
                pst_zone = ZoneInfo("America/Los_Angeles")
                since_pst = since_date.astimezone(pst_zone)
                until_pst = until_date.astimezone(pst_zone)

                # Format for display
                since_formatted = since_pst.strftime("%B %d, %Y")
                until_formatted = until_pst.strftime("%B %d, %Y")
                f.write(f"date_range_display={since_formatted} to {until_formatted}\n")

    except Exception as e:
        logging.error(f"Failed to create PR digest: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
