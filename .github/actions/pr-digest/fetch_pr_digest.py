#!/usr/bin/env -S uv run
# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "requests>=2.31.0",
# ]
# ///
"""Fetch GitHub PR digest with incremental caching."""

import json
import os
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import List, Optional

import requests


@dataclass
class PullRequestDigest:
    """PR information for digest."""

    number: int
    title: str
    body: str
    merged_at: str
    html_url: str
    diff: str
    author: str
    labels: List[str]


@dataclass
class CacheData:
    """Cache with metadata."""

    version: str = "1.0"
    repository: str = ""
    pr_digests: List[PullRequestDigest] = None
    last_updated: str = ""
    last_pr_merged_at: Optional[str] = None

    def __post_init__(self):
        if self.pr_digests is None:
            self.pr_digests = []


class GitHubClient:
    """Handle GitHub API interactions."""

    API_BASE = "https://api.github.com"

    def __init__(self, token: str):
        self.token = token
        self.headers = {"Authorization": f"Bearer {token}", "Accept": "application/vnd.github.v3+json"}

    def _make_request(self, url: str, headers: dict | None = None, params: dict | None = None) -> requests.Response:
        """Make a request with rate limit handling."""
        response = requests.get(url, headers=headers or self.headers, params=params)

        if response.status_code == 403:
            # Check for rate limit
            if "X-RateLimit-Remaining" in response.headers:
                remaining = int(response.headers["X-RateLimit-Remaining"])
                if remaining == 0:
                    reset_time = int(response.headers["X-RateLimit-Reset"])
                    wait_time = reset_time - int(time.time()) + 1
                    print(f"Rate limited. Waiting {wait_time} seconds...")
                    time.sleep(wait_time)
                    return self._make_request(url, headers, params)

        response.raise_for_status()
        return response

    def search_merged_prs(self, repo: str, since: datetime, after_timestamp: Optional[str] = None) -> List[int]:
        """Search for merged PR numbers."""
        query_parts = [f"repo:{repo}", "is:pr", "is:merged", f"merged:>={since.date().isoformat()}"]

        if after_timestamp:
            # Get PRs merged after this specific timestamp
            query_parts.append(f"merged:>{after_timestamp}")

        query = " ".join(query_parts)
        url = f"{self.API_BASE}/search/issues"

        all_numbers = []
        page = 1

        print(f"Searching: {query}")

        while True:
            params = {"q": query, "per_page": 100, "page": page, "sort": "created", "order": "asc"}

            # Use rate-limited request
            response = self._make_request(url, params=params)

            data = response.json()
            items = data.get("items", [])

            if not items:
                break

            all_numbers.extend([item["number"] for item in items])

            if len(items) < 100:
                break
            page += 1

        return all_numbers

    def fetch_pr_details(self, repo: str, pr_number: int, diff_limit: int) -> PullRequestDigest:
        """Fetch detailed PR information."""
        # Get PR data
        pr_url = f"{self.API_BASE}/repos/{repo}/pulls/{pr_number}"
        pr_resp = self._make_request(pr_url)
        pr_data = pr_resp.json()

        # Get diff
        diff_headers = {**self.headers, "Accept": "application/vnd.github.v3.diff"}
        diff_resp = self._make_request(pr_data["diff_url"], headers=diff_headers)

        diff_text = diff_resp.text
        if len(diff_text) > diff_limit:
            diff_text = diff_text[:diff_limit] + "\n...[truncated]"

        return PullRequestDigest(
            number=pr_number,
            title=pr_data["title"],
            body=pr_data.get("body") or "",
            merged_at=pr_data["merged_at"],
            html_url=pr_data["html_url"],
            diff=diff_text,
            author=pr_data["user"]["login"],
            labels=[label["name"] for label in pr_data.get("labels", [])],
        )


def load_cache(cache_file: Path) -> CacheData | None:
    """Load cache if it exists and is valid."""
    if not cache_file.exists():
        return None

    try:
        with open(cache_file, "r") as f:
            data = json.load(f)

        # Handle old format (just a list)
        if isinstance(data, list):
            print("Converting old cache format...")
            return CacheData(pr_digests=[PullRequestDigest(**pr) for pr in data])

        # New format with metadata
        return CacheData(
            version=data.get("version", "1.0"),
            repository=data.get("repository", ""),
            pr_digests=[PullRequestDigest(**pr) for pr in data.get("pr_digests", [])],
            last_updated=data.get("last_updated", ""),
            last_pr_merged_at=data.get("last_pr_merged_at"),
        )
    except Exception as e:
        print(f"Cache load error: {e}")
        return None


def save_cache(cache_file: Path, cache_data: CacheData) -> None:
    """Save cache data."""
    cache_file.parent.mkdir(parents=True, exist_ok=True)

    data = {
        "version": cache_data.version,
        "repository": cache_data.repository,
        "pr_digests": [asdict(pr) for pr in cache_data.pr_digests],
        "last_updated": cache_data.last_updated,
        "last_pr_merged_at": cache_data.last_pr_merged_at,
    }

    with open(cache_file, "w") as f:
        json.dump(data, f, indent=2)


def prune_cache(cache_data: CacheData, max_age_days: int = 60) -> CacheData:
    """Remove cache entries older than max_age_days."""
    if not cache_data.pr_digests:
        return cache_data

    cutoff_date = datetime.now(timezone.utc) - timedelta(days=max_age_days)
    original_count = len(cache_data.pr_digests)

    cache_data.pr_digests = [
        pr for pr in cache_data.pr_digests if datetime.fromisoformat(pr.merged_at.replace("Z", "+00:00")) >= cutoff_date
    ]

    pruned_count = original_count - len(cache_data.pr_digests)
    if pruned_count > 0:
        print(f"Pruned {pruned_count} old cache entries (older than {max_age_days} days)")

    return cache_data


def main() -> None:
    """Main entry point."""
    # Read inputs
    repository = os.getenv("REPOSITORY", "")
    github_token = os.getenv("GITHUB_TOKEN", "")
    days_to_scan = int(os.getenv("DAYS_TO_SCAN", "7"))
    diff_limit = int(os.getenv("DIFF_LIMIT", "20000"))
    force_refresh = os.getenv("FORCE_REFRESH", "false").lower() == "true"

    cache_file_env = os.getenv("CACHE_FILE", ".pr-digest-cache/cache.json")
    if "{repository}" in cache_file_env:
        # If template, replace repository name
        safe_repo_name = repository.replace("/", "_")
        cache_file = Path(cache_file_env.replace("{repository}", safe_repo_name))
    else:
        # Use as-is or create safe default
        cache_file = Path(cache_file_env)
        if cache_file.name == "cache.json":
            safe_repo_name = repository.replace("/", "_")
            cache_file = cache_file.parent / f"{safe_repo_name}-cache.json"

    if not repository or not github_token:
        print("Error: REPOSITORY and GITHUB_TOKEN required")
        sys.exit(1)

    # Initialize
    client = GitHubClient(github_token)
    since_date = datetime.now(timezone.utc) - timedelta(days=days_to_scan)
    end_date = datetime.now(timezone.utc)

    # Load cache
    cache_data = None if force_refresh else load_cache(cache_file)
    cached_prs = []
    last_pr_timestamp = None

    if cache_data and cache_data.pr_digests:
        cache_data = prune_cache(cache_data, max_age_days=60)

        # Filter cached PRs to our date range
        cached_prs = [
            pr
            for pr in cache_data.pr_digests
            if datetime.fromisoformat(pr.merged_at.replace("Z", "+00:00")) >= since_date
        ]

        if cached_prs:
            # Find the most recent PR timestamp
            last_pr_timestamp = max(pr.merged_at for pr in cached_prs)
            print(f"Loaded {len(cached_prs)} PRs from cache (most recent: {last_pr_timestamp})")

    # Fetch new PRs
    if force_refresh or not cached_prs:
        print("Fetching all PRs...")
        pr_numbers = client.search_merged_prs(repository, since_date)
    else:
        print("Fetching new PRs...")
        pr_numbers = client.search_merged_prs(repository, since_date, last_pr_timestamp)
        # Filter out any we already have
        existing_numbers = {pr.number for pr in cached_prs}
        pr_numbers = [n for n in pr_numbers if n not in existing_numbers]

    # Fetch details
    new_prs = []
    for i, number in enumerate(pr_numbers):
        print(f"Fetching PR {i + 1}/{len(pr_numbers)}: #{number}")
        try:
            pr = client.fetch_pr_details(repository, number, diff_limit)
            new_prs.append(pr)
        except Exception as e:
            print(f"Error fetching PR #{number}: {e}")

    # Combine and sort
    all_prs = cached_prs + new_prs
    all_prs.sort(key=lambda pr: pr.merged_at, reverse=True)

    # Save updated cache
    if new_prs or force_refresh:
        cache_data = CacheData(
            repository=repository,
            pr_digests=all_prs,
            last_updated=datetime.now(timezone.utc).isoformat(),
            last_pr_merged_at=all_prs[0].merged_at if all_prs else None,
        )
        save_cache(cache_file, cache_data)
        print(f"Updated cache with {len(all_prs)} total PRs")

    # Write output
    output_file = Path("pr_digest_output.json")
    with open(output_file, "w") as f:
        json.dump([asdict(pr) for pr in all_prs], f, indent=2)

    # Set GitHub outputs
    github_output = os.getenv("GITHUB_OUTPUT")
    if github_output:
        with open(github_output, "a") as f:
            f.write(f"file={output_file}\n")
            f.write(f"count={len(all_prs)}\n")
            f.write(f"cache_stats=cached:{len(cached_prs)},new:{len(new_prs)}\n")
            f.write(f"date_range={since_date.date().isoformat()} to {end_date.date().isoformat()}\n")

    print(f"\n✓ Summary: {len(all_prs)} total PRs ({len(new_prs)} new, {len(cached_prs)} cached)")


if __name__ == "__main__":
    main()
