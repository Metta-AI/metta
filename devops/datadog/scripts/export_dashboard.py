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
    - DD_API_KEY environment variable (or AWS Secrets Manager)
    - DD_APP_KEY environment variable (or AWS Secrets Manager)
    - DD_SITE environment variable (optional, defaults to datadoghq.com)

Usage:
    # Export by dashboard ID
    ./devops/datadog/scripts/export_dashboard.py abc-123-def

    # Save to file
    ./devops/datadog/scripts/export_dashboard.py abc-123-def > dashboards/templates/my_dashboard.json

    # Export by URL (extracts ID automatically)
    ./devops/datadog/scripts/export_dashboard.py "https://app.datadoghq.com/dashboard/abc-123-def"
"""

import json
import re
import sys
from pathlib import Path

# Add parent directory to path for local imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.dashboard_client import DatadogDashboardClient


def extract_dashboard_id(input_str: str) -> str:
    """Extract dashboard ID from URL or return as-is if already an ID."""
    # Check if it's a URL
    url_pattern = r"dashboard/([a-z0-9-]+)"
    match = re.search(url_pattern, input_str)
    if match:
        return match.group(1)

    # Assume it's already a dashboard ID
    return input_str


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
        client = DatadogDashboardClient()
        dashboard_input = sys.argv[1]
        dashboard_id = extract_dashboard_id(dashboard_input)

        print(f"Fetching dashboard: {dashboard_id}...", file=sys.stderr)

        dashboard_json = client.get_dashboard(dashboard_id)

        # Print to stdout for easy redirection
        print(json.dumps(dashboard_json, indent=2))

        print(
            f"\n✓ Dashboard '{dashboard_json.get('title', 'Unknown')}' exported successfully",
            file=sys.stderr,
        )

    except ValueError as e:
        print(f"Error: {e}", file=sys.stderr)
        print("\nSet environment variables or configure AWS Secrets Manager:", file=sys.stderr)
        print("  export DD_API_KEY=your_api_key", file=sys.stderr)
        print("  export DD_APP_KEY=your_app_key", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        if "404" in str(e):
            print(
                f"\nDashboard '{dashboard_id}' not found. Check the ID and try again.",
                file=sys.stderr,
            )
        sys.exit(1)


if __name__ == "__main__":
    main()
