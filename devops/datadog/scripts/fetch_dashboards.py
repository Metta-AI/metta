#!/usr/bin/env -S uv run --quiet --script
# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "requests",
# ]
# ///
"""Fetch all Datadog dashboards and export their metadata.

This script retrieves all dashboards from Datadog using the API and exports
their metadata (ID, title, description, URL) to help with migration to Terraform.

Requirements:
    - DD_API_KEY environment variable
    - DD_APP_KEY environment variable
    - DD_SITE environment variable (optional, defaults to datadoghq.com)

Usage:
    export DD_API_KEY=your_api_key
    export DD_APP_KEY=your_app_key
    ./devops/datadog/fetch_dashboards.py

    # Save to file
    ./devops/datadog/fetch_dashboards.py > dashboards.json

    # Get summary
    ./devops/datadog/fetch_dashboards.py --format=summary
"""

import json
import os
import sys
from typing import Any

import requests


class DatadogDashboardFetcher:
    """Fetches dashboard metadata from Datadog API."""

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

    def fetch_all_dashboards(self) -> list[dict[str, Any]]:
        """Fetch all dashboards from Datadog.

        Returns list of dashboard summaries with id, title, url, etc.
        """
        url = f"{self.base_url}/v1/dashboard"
        response = self._session.get(url)
        response.raise_for_status()

        data = response.json()
        dashboards = data.get("dashboards", [])

        print(f"Found {len(dashboards)} dashboards", file=sys.stderr)
        return dashboards

    def fetch_dashboard_details(self, dashboard_id: str) -> dict[str, Any]:
        """Fetch detailed information for a specific dashboard."""
        url = f"{self.base_url}/v1/dashboard/{dashboard_id}"
        response = self._session.get(url)
        response.raise_for_status()
        return response.json()

    def export_dashboards_summary(self, dashboards: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Export simplified dashboard metadata suitable for migration planning."""
        result = []

        for dashboard in dashboards:
            dashboard_id = dashboard.get("id", "")
            result.append(
                {
                    "id": dashboard_id,
                    "title": dashboard.get("title", ""),
                    "description": dashboard.get("description", ""),
                    "author_handle": dashboard.get("author_handle", ""),
                    "url": dashboard.get("url", ""),
                    "created_at": dashboard.get("created_at", ""),
                    "modified_at": dashboard.get("modified_at", ""),
                    "is_read_only": dashboard.get("is_read_only", False),
                    "layout_type": dashboard.get("layout_type", ""),
                }
            )

        return result


def format_summary(dashboards: list[dict[str, Any]]) -> str:
    """Format dashboards as human-readable summary."""
    lines = [
        "Datadog Dashboards Summary",
        "=" * 80,
        f"Total: {len(dashboards)} dashboards",
        "",
    ]

    for i, dash in enumerate(dashboards, 1):
        lines.append(f"{i}. {dash['title']}")
        lines.append(f"   ID: {dash['id']}")
        lines.append(f"   URL: {dash['url']}")
        if dash.get("description"):
            lines.append(f"   Description: {dash['description']}")
        lines.append(f"   Layout: {dash.get('layout_type', 'unknown')}")
        lines.append(f"   Modified: {dash.get('modified_at', 'unknown')}")
        lines.append("")

    return "\n".join(lines)


def main():
    """Main entry point."""
    import argparse

    parser = argparse.ArgumentParser(description="Fetch Datadog dashboards metadata")
    parser.add_argument(
        "--format",
        choices=["json", "summary"],
        default="json",
        help="Output format (default: json)",
    )
    parser.add_argument(
        "--details",
        action="store_true",
        help="Fetch full details for each dashboard (slower)",
    )

    args = parser.parse_args()

    try:
        fetcher = DatadogDashboardFetcher()
        dashboards = fetcher.fetch_all_dashboards()

        if args.details:
            print("Fetching detailed information...", file=sys.stderr)
            detailed = []
            for dash in dashboards:
                dashboard_id = dash.get("id")
                if dashboard_id:
                    details = fetcher.fetch_dashboard_details(dashboard_id)
                    detailed.append(details)
            dashboards = detailed

        summary = fetcher.export_dashboards_summary(dashboards)

        if args.format == "summary":
            print(format_summary(summary))
        else:
            print(json.dumps(summary, indent=2))

    except ValueError as e:
        print(f"Error: {e}", file=sys.stderr)
        print("\nSet environment variables:", file=sys.stderr)
        print("  export DD_API_KEY=your_api_key", file=sys.stderr)
        print("  export DD_APP_KEY=your_app_key", file=sys.stderr)
        sys.exit(1)
    except requests.exceptions.RequestException as e:
        print(f"API Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
