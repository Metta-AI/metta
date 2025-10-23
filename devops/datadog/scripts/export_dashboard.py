#!/usr/bin/env -S uv run --quiet --script
# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "requests",
# ]
# ///
"""Export a specific Datadog dashboard to JSON.

This script exports a single dashboard's full JSON definition, which can be used
as a reference when migrating to Terraform.

Requirements:
    - DD_API_KEY environment variable
    - DD_APP_KEY environment variable
    - DD_SITE environment variable (optional, defaults to datadoghq.com)

Usage:
    export DD_API_KEY=your_api_key
    export DD_APP_KEY=your_app_key

    # Export by dashboard ID
    ./devops/datadog/export_dashboard.py abc-123-def

    # Save to file
    ./devops/datadog/export_dashboard.py abc-123-def > templates/my_dashboard.json

    # Export by URL (extracts ID automatically)
    ./devops/datadog/export_dashboard.py "https://app.datadoghq.com/dashboard/abc-123-def"
"""

import json
import os
import re
import sys

import requests


class DashboardExporter:
    """Exports Datadog dashboard JSON."""

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

    def extract_dashboard_id(self, input_str: str) -> str:
        """Extract dashboard ID from URL or return as-is if already an ID."""
        # Check if it's a URL
        url_pattern = r"dashboard/([a-z0-9-]+)"
        match = re.search(url_pattern, input_str)
        if match:
            return match.group(1)

        # Assume it's already a dashboard ID
        return input_str

    def export_dashboard(self, dashboard_id: str) -> dict:
        """Export dashboard JSON from Datadog API."""
        url = f"{self.base_url}/v1/dashboard/{dashboard_id}"
        response = self._session.get(url)
        response.raise_for_status()
        return response.json()


def main():
    """Main entry point."""
    if len(sys.argv) < 2:
        print("Usage: export_dashboard.py <dashboard-id-or-url>", file=sys.stderr)
        print("\nExample: export_dashboard.py abc-123-def", file=sys.stderr)
        print(
            "Example: export_dashboard.py https://app.datadoghq.com/dashboard/abc-123-def",
            file=sys.stderr,
        )
        sys.exit(1)

    try:
        exporter = DashboardExporter()
        dashboard_input = sys.argv[1]
        dashboard_id = exporter.extract_dashboard_id(dashboard_input)

        print(f"Fetching dashboard: {dashboard_id}...", file=sys.stderr)

        dashboard_json = exporter.export_dashboard(dashboard_id)

        # Print to stdout for easy redirection
        print(json.dumps(dashboard_json, indent=2))

        print(
            f"\n✓ Dashboard '{dashboard_json.get('title', 'Unknown')}' exported successfully",
            file=sys.stderr,
        )

    except ValueError as e:
        print(f"Error: {e}", file=sys.stderr)
        print("\nSet environment variables:", file=sys.stderr)
        print("  export DD_API_KEY=your_api_key", file=sys.stderr)
        print("  export DD_APP_KEY=your_app_key", file=sys.stderr)
        sys.exit(1)
    except requests.exceptions.HTTPError as e:
        print(f"API Error: {e}", file=sys.stderr)
        if e.response.status_code == 404:
            print(
                f"\nDashboard '{dashboard_id}' not found. Check the ID and try again.",
                file=sys.stderr,
            )
        sys.exit(1)
    except requests.exceptions.RequestException as e:
        print(f"Request Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
