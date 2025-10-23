#!/usr/bin/env -S uv run --quiet --script
# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "requests",
# ]
# ///
"""Batch export multiple Datadog dashboards to JSON files.

This script fetches all dashboards and exports each one to a separate JSON file
in the templates directory. Useful for bulk migration planning.

Requirements:
    - DD_API_KEY environment variable
    - DD_APP_KEY environment variable
    - DD_SITE environment variable (optional, defaults to datadoghq.com)

Usage:
    export DD_API_KEY=your_api_key
    export DD_APP_KEY=your_app_key

    # Export all dashboards
    ./devops/datadog/batch_export.py

    # Export specific dashboards by ID
    ./devops/datadog/batch_export.py abc-123-def xyz-456-ghi

    # Limit number of dashboards to export
    ./devops/datadog/batch_export.py --limit=5
"""

import json
import os
import re
import sys
from pathlib import Path

import requests


class BatchExporter:
    """Batch export Datadog dashboards."""

    def __init__(self):
        self.api_key = os.getenv("DD_API_KEY")
        self.app_key = os.getenv("DD_APP_KEY")
        self.site = os.getenv("DD_SITE", "datadoghq.com")

        if not self.api_key or not self.app_key:
            raise ValueError("Missing required environment variables: DD_API_KEY and DD_APP_KEY")

        self.base_url = f"https://api.{self.site}/api"
        self._session = self._create_session()
        self.templates_dir = Path(__file__).parent / "templates"

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

    def fetch_all_dashboards(self) -> list[dict]:
        """Fetch list of all dashboards."""
        url = f"{self.base_url}/v1/dashboard"
        response = self._session.get(url)
        response.raise_for_status()
        data = response.json()
        return data.get("dashboards", [])

    def fetch_dashboard(self, dashboard_id: str) -> dict:
        """Fetch full dashboard JSON."""
        url = f"{self.base_url}/v1/dashboard/{dashboard_id}"
        response = self._session.get(url)
        response.raise_for_status()
        return response.json()

    def sanitize_filename(self, title: str) -> str:
        """Convert dashboard title to safe filename."""
        # Remove/replace unsafe characters
        safe_title = re.sub(r'[<>:"/\\|?*]', "", title)
        safe_title = re.sub(r"\s+", "_", safe_title)
        safe_title = safe_title.lower()
        # Limit length
        return safe_title[:100]

    def export_dashboard(self, dashboard_id: str, dashboard_title: str | None = None):
        """Export single dashboard to JSON file."""
        try:
            dashboard_json = self.fetch_dashboard(dashboard_id)

            # Use title from JSON if not provided
            if not dashboard_title:
                dashboard_title = dashboard_json.get("title", dashboard_id)

            filename = self.sanitize_filename(dashboard_title)
            filepath = self.templates_dir / f"{filename}.json"

            # Ensure templates directory exists
            self.templates_dir.mkdir(exist_ok=True)

            # Write JSON file
            with open(filepath, "w") as f:
                json.dump(dashboard_json, f, indent=2)

            print(f"✓ Exported: {dashboard_title}")
            print(f"  → {filepath}")
            return True

        except requests.exceptions.HTTPError as e:
            print(f"✗ Failed to export {dashboard_id}: {e}", file=sys.stderr)
            return False
        except Exception as e:
            print(f"✗ Error exporting {dashboard_id}: {e}", file=sys.stderr)
            return False

    def batch_export(self, dashboard_ids: list[str] | None = None, limit: int | None = None):
        """Export multiple dashboards.

        If dashboard_ids is provided, export only those dashboards.
        Otherwise, export all dashboards.
        """
        if dashboard_ids:
            # Export specific dashboards
            dashboards = [{"id": dash_id} for dash_id in dashboard_ids]
        else:
            # Fetch all dashboards
            print("Fetching dashboard list...", file=sys.stderr)
            dashboards = self.fetch_all_dashboards()
            print(f"Found {len(dashboards)} dashboards\n", file=sys.stderr)

        # Apply limit if specified
        if limit:
            dashboards = dashboards[:limit]

        success_count = 0
        total = len(dashboards)

        for i, dashboard in enumerate(dashboards, 1):
            dashboard_id = dashboard.get("id")
            dashboard_title = dashboard.get("title", dashboard_id)

            print(f"[{i}/{total}] ", end="")
            if self.export_dashboard(dashboard_id, dashboard_title):
                success_count += 1
            print()

        print(
            f"\nExported {success_count}/{total} dashboards to {self.templates_dir}",
            file=sys.stderr,
        )


def main():
    """Main entry point."""
    import argparse

    parser = argparse.ArgumentParser(description="Batch export Datadog dashboards to JSON files")
    parser.add_argument(
        "dashboard_ids",
        nargs="*",
        help="Dashboard IDs to export (if not specified, exports all)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        help="Maximum number of dashboards to export",
    )

    args = parser.parse_args()

    try:
        exporter = BatchExporter()
        exporter.batch_export(
            dashboard_ids=args.dashboard_ids if args.dashboard_ids else None,
            limit=args.limit,
        )

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
