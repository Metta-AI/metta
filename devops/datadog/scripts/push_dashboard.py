#!/usr/bin/env -S uv run --quiet --script
# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "requests",
# ]
# ///
"""Push dashboard JSON files to Datadog.

This script uploads dashboard JSON files directly to Datadog, creating new
dashboards or updating existing ones.

Requirements:
    - DD_API_KEY environment variable
    - DD_APP_KEY environment variable
    - DD_SITE environment variable (optional, defaults to datadoghq.com)

Usage:
    export DD_API_KEY=your_api_key
    export DD_APP_KEY=your_app_key

    # Push a specific dashboard
    ./push_dashboard.py templates/my_dashboard.json

    # Push all dashboards
    ./push_dashboard.py templates/*.json

    # Dry run (validate without pushing)
    ./push_dashboard.py templates/*.json --dry-run
"""

import json
import os
import sys
from pathlib import Path

import requests


class DashboardPusher:
    """Pushes dashboard JSON to Datadog API."""

    def __init__(self):
        self.api_key = os.getenv("DD_API_KEY")
        self.app_key = os.getenv("DD_APP_KEY")
        self.site = os.getenv("DD_SITE", "datadoghq.com")

        if not self.api_key or not self.app_key:
            raise ValueError("Missing required environment variables: DD_API_KEY and DD_APP_KEY")

        self.base_url = f"https://api.{self.site}/api"
        self._session = self._create_session()

    def _create_session(self) -> requests.Session:
        """Create authenticated session for Datadog API."""
        session = requests.Session()
        session.headers.update(
            {
                "DD-API-KEY": self.api_key,
                "DD-APPLICATION-KEY": self.app_key,
                "Content-Type": "application/json",
            }
        )
        return session

    def dashboard_exists(self, dashboard_id: str) -> bool:
        """Check if a dashboard exists in Datadog."""
        url = f"{self.base_url}/v1/dashboard/{dashboard_id}"
        try:
            response = self._session.get(url)
            return response.status_code == 200
        except requests.exceptions.RequestException:
            return False

    def create_dashboard(self, dashboard_json: dict) -> dict:
        """Create a new dashboard in Datadog."""
        url = f"{self.base_url}/v1/dashboard"

        # Remove fields that shouldn't be in create request
        payload = dashboard_json.copy()
        for field in ["id", "url", "created_at", "modified_at", "author_handle", "author_name"]:
            payload.pop(field, None)

        response = self._session.post(url, json=payload)
        response.raise_for_status()
        return response.json()

    def update_dashboard(self, dashboard_id: str, dashboard_json: dict) -> dict:
        """Update an existing dashboard in Datadog."""
        url = f"{self.base_url}/v1/dashboard/{dashboard_id}"

        # Remove fields that shouldn't be in update request
        payload = dashboard_json.copy()
        for field in ["id", "url", "created_at", "modified_at", "author_handle", "author_name"]:
            payload.pop(field, None)

        response = self._session.put(url, json=payload)
        response.raise_for_status()
        return response.json()

    def push_dashboard(self, json_file: Path, dry_run: bool = False) -> tuple[bool, str]:
        """Push a dashboard JSON file to Datadog.

        Returns tuple of (success, message)
        """
        try:
            with open(json_file) as f:
                dashboard_json = json.load(f)

            dashboard_id = dashboard_json.get("id")
            title = dashboard_json.get("title", "Unknown")

            if dry_run:
                if dashboard_id and self.dashboard_exists(dashboard_id):
                    return True, f"[DRY RUN] Would UPDATE: {title} (ID: {dashboard_id})"
                else:
                    return True, f"[DRY RUN] Would CREATE: {title}"

            # Check if dashboard exists
            if dashboard_id and self.dashboard_exists(dashboard_id):
                # Update existing dashboard
                result = self.update_dashboard(dashboard_id, dashboard_json)
                return True, f"✓ Updated: {title} (ID: {dashboard_id})"
            else:
                # Create new dashboard
                result = self.create_dashboard(dashboard_json)
                new_id = result.get("id", "unknown")
                return True, f"✓ Created: {title} (ID: {new_id})"

        except json.JSONDecodeError as e:
            return False, f"✗ Invalid JSON in {json_file.name}: {e}"
        except requests.exceptions.HTTPError as e:
            return False, f"✗ API Error for {json_file.name}: {e}"
        except Exception as e:
            return False, f"✗ Failed to push {json_file.name}: {e}"

    def push_all(self, json_files: list[Path], dry_run: bool = False) -> tuple[int, int]:
        """Push multiple dashboard JSON files.

        Returns tuple of (success_count, total_count)
        """
        success_count = 0
        total = len(json_files)

        for i, json_file in enumerate(json_files, 1):
            print(f"[{i}/{total}] ", end="")
            success, message = self.push_dashboard(json_file, dry_run=dry_run)
            print(message)

            if success:
                success_count += 1

        return success_count, total


def main():
    """Main entry point."""
    import argparse

    parser = argparse.ArgumentParser(description="Push dashboard JSON files to Datadog")
    parser.add_argument(
        "json_files",
        nargs="+",
        type=Path,
        help="JSON files to push",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate without pushing to Datadog",
    )

    args = parser.parse_args()

    # Validate input files
    json_files = []
    for pattern in args.json_files:
        if "*" in str(pattern):
            # Handle glob patterns
            matches = list(pattern.parent.glob(pattern.name))
            json_files.extend(matches)
        else:
            if not pattern.exists():
                print(f"Error: File not found: {pattern}", file=sys.stderr)
                sys.exit(1)
            json_files.append(pattern)

    if not json_files:
        print("Error: No JSON files found", file=sys.stderr)
        sys.exit(1)

    try:
        pusher = DashboardPusher()

        print(f"Pushing {len(json_files)} dashboard(s) to Datadog...")
        if args.dry_run:
            print("[DRY RUN MODE - No changes will be made]")
        print()

        success_count, total = pusher.push_all(json_files, dry_run=args.dry_run)

        print()
        if args.dry_run:
            print(f"[DRY RUN] Would push {success_count}/{total} dashboard(s)")
        else:
            print(f"✓ Successfully pushed {success_count}/{total} dashboard(s)")
            if success_count < total:
                print(f"⚠ {total - success_count} dashboard(s) failed")
                sys.exit(1)

    except ValueError as e:
        print(f"Error: {e}", file=sys.stderr)
        print("\nSet environment variables:", file=sys.stderr)
        print("  export DD_API_KEY=your_api_key", file=sys.stderr)
        print("  export DD_APP_KEY=your_app_key", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
