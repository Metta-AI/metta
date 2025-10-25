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
    - DD_API_KEY environment variable (or AWS Secrets Manager)
    - DD_APP_KEY environment variable (or AWS Secrets Manager)
    - DD_SITE environment variable (optional, defaults to datadoghq.com)

Usage:
    # Push a specific dashboard
    ./push_dashboard.py dashboards/templates/my_dashboard.json

    # Push all dashboards
    ./push_dashboard.py dashboards/templates/*.json

    # Dry run (validate without pushing)
    ./push_dashboard.py dashboards/templates/*.json --dry-run
"""

import json
import sys
from pathlib import Path

from devops.datadog.utils.dashboard_client import DatadogDashboardClient


def push_dashboard(client: DatadogDashboardClient, json_file: Path, dry_run: bool = False) -> tuple[bool, str]:
    """Push a dashboard JSON file to Datadog.

    Returns tuple of (success, message)
    """
    try:
        with open(json_file) as f:
            dashboard_json = json.load(f)

        dashboard_id = dashboard_json.get("id")
        title = dashboard_json.get("title", "Unknown")

        if dry_run:
            if dashboard_id and client.dashboard_exists(dashboard_id):
                return True, f"[DRY RUN] Would UPDATE: {title} (ID: {dashboard_id})"
            else:
                return True, f"[DRY RUN] Would CREATE: {title}"

        # Check if dashboard exists
        if dashboard_id and client.dashboard_exists(dashboard_id):
            # Update existing dashboard
            result = client.update_dashboard(dashboard_id, dashboard_json)
            return True, f"✓ Updated: {title} (ID: {dashboard_id})"
        else:
            # Create new dashboard
            result = client.create_dashboard(dashboard_json)
            new_id = result.get("id", "unknown")
            return True, f"✓ Created: {title} (ID: {new_id})"

    except json.JSONDecodeError as e:
        return False, f"✗ Invalid JSON in {json_file.name}: {e}"
    except Exception as e:
        return False, f"✗ Failed to push {json_file.name}: {e}"


def push_all(client: DatadogDashboardClient, json_files: list[Path], dry_run: bool = False) -> tuple[int, int]:
    """Push multiple dashboard JSON files.

    Returns tuple of (success_count, total_count)
    """
    success_count = 0
    total = len(json_files)

    for i, json_file in enumerate(json_files, 1):
        print(f"[{i}/{total}] ", end="")
        success, message = push_dashboard(client, json_file, dry_run=dry_run)
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
        client = DatadogDashboardClient()

        print(f"Pushing {len(json_files)} dashboard(s) to Datadog...")
        if args.dry_run:
            print("[DRY RUN MODE - No changes will be made]")
        print()

        success_count, total = push_all(client, json_files, dry_run=args.dry_run)

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
        print("\nSet environment variables or configure AWS Secrets Manager:", file=sys.stderr)
        print("  export DD_API_KEY=your_api_key", file=sys.stderr)
        print("  export DD_APP_KEY=your_app_key", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
