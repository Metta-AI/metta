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

    def is_by_author(self, author: str) -> bool:
        """Check if this PR was authored by the specified user."""
        metadata = self.load_metadata()
        if not metadata or not metadata.get("author"):
            return False

        return metadata["author"].lower() == author.lower()


class PRDigestCreator:
    """Creates a digest of merged PRs, utilizing cache effectively."""

    def __init__(self, token: str, repository: str, filter_author: Optional[str] = None):
        self.token = token
        self.repository = repository
        self.filter_author = filter_author
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

    def get_merged_prs_in_range(self, since: datetime, until: datetime, include_drafts: bool = False) -> List[Dict]:
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
                # Skip if not merged (unless including drafts that were closed)
                if not pr.get("merged_at"):
                    if not (include_drafts and pr.get("draft") and pr.get("closed_at")):
                        continue

                # Use merged_at or closed_at for drafts
                date_field = pr.get("merged_at") or pr.get("closed_at")
                if not date_field:
                    continue

                closed_at = datetime.fromisoformat(date_field.replace("Z", "+00:00"))

                # Apply author filter if specified
                if self.filter_author and pr["user"]["login"].lower() != self.filter_author.lower():
                    continue

                if since <= closed_at <= until:
                    all_prs.append(
                        {
                            "number": pr["number"],
                            "title": pr["title"],
                            "merged_at": pr.get("merged_at"),
                            "closed_at": pr.get("closed_at"),
                            "html_url": pr["html_url"],
                            "author": pr["user"]["login"],
                            "draft": pr.get("draft", False),
                        }
                    )
                    consecutive_old_prs = 0
                    page_found_recent = True
                elif closed_at < since:
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

        author_msg = f" by {self.filter_author}" if self.filter_author else ""
        logging.info(f"Found {len(all_prs)} merged PRs{author_msg} in date range from GitHub API")
        return all_prs

    def get_pr_details(self, pr_number: int) -> Optional[Dict]:
        """Get detailed information for a specific PR including diff."""
        try:
            # Get basic PR info
            pr_url = f"https://api.github.com/repos/{self.repository}/pulls/{pr_number}"
            pr_response = self.session.get(pr_url)
            pr_response.raise_for_status()
            pr_data = pr_response.json()

            # Skip if author doesn't match filter
            if self.filter_author and pr_data["user"]["login"].lower() != self.filter_author.lower():
                return None

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
                "closed_at": pr_data.get("closed_at"),
                "html_url": pr_data["html_url"],
                "labels": labels,
                "diff": diff_response.text,
                "additions": pr_data.get("additions", 0),
                "deletions": pr_data.get("deletions", 0),
                "changed_files": pr_data.get("changed_files", 0),
                "draft": pr_data.get("draft", False),
            }

        except Exception as e:
            logging.error(f"Error fetching details for PR #{pr_number}: {e}")
            return None

    def create_digest(
        self, since_str: str, until_str: str, include_drafts: bool = False
    ) -> Tuple[List[Dict], Dict[str, int]]:
        """Create a complete digest of PRs for the time period.

        Returns:
            - List of PR details (for PRs that need to be processed)
            - Statistics dictionary
        """
        since = datetime.fromisoformat(since_str.replace("Z", "+00:00"))
        until = datetime.fromisoformat(until_str.replace("Z", "+00:00"))

        author_msg = f" by {self.filter_author}" if self.filter_author else ""
        logging.info(f"Creating PR digest for {self.repository}{author_msg} from {since_str} to {until_str}")

        # Step 1: Load all cached summaries
        cached_summaries = self.load_all_cached_summaries()

        # Step 2: Get list of PRs in date range from GitHub
        prs_from_github = self.get_merged_prs_in_range(since, until, include_drafts)
        github_pr_numbers = {pr["number"] for pr in prs_from_github}

        # Step 3: Find cached PRs that are in our date range (and by author if filtering)
        cached_prs_in_range = []
        for pr_num, cached_summary in cached_summaries.items():
            if cached_summary.is_in_date_range(since, until):
                # Apply author filter if specified
                if self.filter_author and not cached_summary.is_by_author(self.filter_author):
                    continue

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
            "filter_author": self.filter_author,
            "include_drafts": include_drafts,
        }

        logging.info(f"""
=== PR Digest Statistics ===
Author Filter: {self.filter_author or "None"}
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

    # Check if we're filtering by author
    filter_author_mode = "--filter-author" in sys.argv

    # Define required environment variables
    required_env_vars = {
        "GITHUB_TOKEN": "GitHub Token",
        "GITHUB_REPOSITORY": "GitHub repository name",
        "DAYS_TO_SCAN": "Days to process",
    }

    # Add author requirement if filtering
    if filter_author_mode:
        required_env_vars["PR_AUTHOR"] = "PR author to filter by"

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
    filter_author = env_values.get("PR_AUTHOR") if filter_author_mode else None
    include_drafts = os.getenv("INCLUDE_DRAFT_PRS", "false").lower() == "true"
    output_file = os.getenv("PR_DIGEST_FILE", "pr_digest_output.json")

    # Check for historical date support
    historical_end_date = os.getenv("HISTORICAL_END_DATE")
    is_historical = os.getenv("IS_HISTORICAL_RUN", "false").lower() == "true"

    # Determine the end date
    if historical_end_date and is_historical:
        try:
            until_date = datetime.strptime(historical_end_date, "%Y-%m-%d")
            # Make it end of day in UTC
            until_date = until_date.replace(hour=23, minute=59, second=59)
            logging.info(f"Using end date {historical_end_date}")
        except ValueError:
            logging.error(f"Invalid historical date format: {historical_end_date}. Expected YYYY-MM-DD")
            sys.exit(1)
    else:
        until_date = datetime.now()

    days = int(days_to_scan)
    since_date = until_date - timedelta(days=days)
    since = since_date.isoformat() + "Z"
    until = until_date.isoformat() + "Z"

    logging.info(f"Date range: {since} to {until}")
    if filter_author:
        logging.info(f"Filtering by author: {filter_author}")
    if include_drafts:
        logging.info("Including draft PRs")

    creator = PRDigestCreator(github_token, github_repository, filter_author)

    try:
        prs_to_process, stats = creator.create_digest(since, until, include_drafts)

        # Save PRs that need processing to file
        output_path = Path(output_file)
        with open(output_path, "w") as f:
            json.dump(prs_to_process, f, indent=2)

        author_msg = f" by {filter_author}" if filter_author else ""
        print(f"✅ PR Digest created successfully{author_msg}")
        print(f"   - Total PRs in period: {stats['total_prs_in_range']}")
        print(f"   - Already summarized: {stats['cached_prs_in_range']}")
        print(f"   - New PRs to process: {stats['new_prs_to_fetch']}")
        print(f"✅ Saved {len(prs_to_process)} PRs to process in {output_path}")

        # Save statistics for the workflow
        stats_file = Path("pr_digest_stats.json")
        stats["is_historical"] = is_historical

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

                # Add author info if filtering
                if filter_author:
                    f.write(f"filter_author={filter_author}\n")

    except Exception as e:
        logging.error(f"Failed to create PR digest: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
