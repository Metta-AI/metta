#!/usr/bin/env -S uv run --quiet --script
# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "requests",
# ]
# ///
"""Batch export multiple Datadog dashboards to JSON files.

This script fetches all dashboards and exports each one to a separate JSON file
in the dashboards/templates directory. Useful for bulk migration planning.

Requirements:
    - DD_API_KEY environment variable (or AWS Secrets Manager)
    - DD_APP_KEY environment variable (or AWS Secrets Manager)
    - DD_SITE environment variable (optional, defaults to datadoghq.com)

Usage:
    # Export all dashboards
    ./devops/datadog/scripts/batch_export.py

    # Export specific dashboards by ID
    ./devops/datadog/scripts/batch_export.py abc-123-def xyz-456-ghi

    # Limit number of dashboards to export
    ./devops/datadog/scripts/batch_export.py --limit=5
"""

import json
import re
import sys
from pathlib import Path

# Add parent directory to path for local imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.dashboard_client import DatadogDashboardClient


def sanitize_filename(title: str) -> str:
    """Convert dashboard title to safe filename."""
    # Remove/replace unsafe characters
    safe_title = re.sub(r'[<>:"/\\|?*]', "", title)
    safe_title = re.sub(r"\s+", "_", safe_title)
    safe_title = safe_title.lower()
    # Limit length
    return safe_title[:100]


def export_dashboard(
    client: DatadogDashboardClient,
    dashboard_id: str,
    dashboard_title: str | None,
    templates_dir: Path,
) -> bool:
    """Export single dashboard to JSON file."""
    try:
        dashboard_json = client.get_dashboard(dashboard_id)

        # Use title from JSON if not provided
        if not dashboard_title:
            dashboard_title = dashboard_json.get("title", dashboard_id)

        filename = sanitize_filename(dashboard_title)
        filepath = templates_dir / f"{filename}.json"

        # Ensure templates directory exists
        templates_dir.mkdir(exist_ok=True, parents=True)

        # Write JSON file
        with open(filepath, "w") as f:
            json.dump(dashboard_json, f, indent=2)

        print(f"✓ Exported: {dashboard_title}")
        print(f"  → {filepath}")
        return True

    except Exception as e:
        print(f"✗ Failed to export {dashboard_id}: {e}", file=sys.stderr)
        return False


def batch_export(
    client: DatadogDashboardClient,
    templates_dir: Path,
    dashboard_ids: list[str] | None = None,
    limit: int | None = None,
) -> tuple[int, int]:
    """Export multiple dashboards.

    If dashboard_ids is provided, export only those dashboards.
    Otherwise, export all dashboards.

    Returns (success_count, total_count)
    """
    if dashboard_ids:
        # Export specific dashboards
        dashboards = [{"id": dash_id} for dash_id in dashboard_ids]
    else:
        # Fetch all dashboards
        print("Fetching dashboard list...", file=sys.stderr)
        dashboards = client.list_dashboards()
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
        if export_dashboard(client, dashboard_id, dashboard_title, templates_dir):
            success_count += 1
        print()

    return success_count, total


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
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).parent.parent / "dashboards" / "templates",
        help="Output directory for JSON files (default: dashboards/templates)",
    )

    args = parser.parse_args()

    try:
        client = DatadogDashboardClient()

        success_count, total = batch_export(
            client,
            args.output_dir,
            dashboard_ids=args.dashboard_ids if args.dashboard_ids else None,
            limit=args.limit,
        )

        print(f"\nExported {success_count}/{total} dashboards to {args.output_dir}", file=sys.stderr)

        if success_count < total:
            sys.exit(1)

    except ValueError as e:
        print(f"Error: {e}", file=sys.stderr)
        print("\nSet environment variables or configure AWS Secrets Manager:", file=sys.stderr)
        print("  export DD_API_KEY=your_api_key", file=sys.stderr)
        print("  export DD_APP_KEY=your_app_key", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
