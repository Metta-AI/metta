#!/usr/bin/env -S uv run --quiet --script
# /// script
# dependencies = [
#   "requests",
# ]
# ///
"""Delete dashboard(s) from Datadog.

Usage:
    ./delete_dashboard.py <dashboard-id>
    ./delete_dashboard.py <dashboard-id1> <dashboard-id2> ...
    ./delete_dashboard.py --dry-run <dashboard-id>

Environment variables:
    DD_API_KEY: Datadog API key
    DD_APP_KEY: Datadog application key
    DD_SITE: Datadog site (default: datadoghq.com)
"""

import os
import sys

import requests


def delete_dashboard(dashboard_id: str, dry_run: bool = False) -> bool:
    """Delete a dashboard from Datadog.

    Args:
        dashboard_id: The dashboard ID to delete
        dry_run: If True, only print what would be deleted

    Returns:
        True if successful, False otherwise
    """
    api_key = os.environ.get("DD_API_KEY")
    app_key = os.environ.get("DD_APP_KEY")
    site = os.environ.get("DD_SITE", "datadoghq.com")

    if not api_key or not app_key:
        print("Error: DD_API_KEY and DD_APP_KEY environment variables must be set")
        print("Run: source ./load_env.sh")
        return False

    url = f"https://api.{site}/api/v1/dashboard/{dashboard_id}"
    headers = {
        "DD-API-KEY": api_key,
        "DD-APPLICATION-KEY": app_key,
    }

    if dry_run:
        # Get dashboard info first
        try:
            response = requests.get(url, headers=headers)
            response.raise_for_status()
            dashboard = response.json()
            print(f"[DRY RUN] Would delete: {dashboard.get('title', 'Unknown')} (ID: {dashboard_id})")
            return True
        except requests.exceptions.RequestException as e:
            print(f"[DRY RUN] Error fetching dashboard {dashboard_id}: {e}")
            return False
    else:
        # Actually delete
        try:
            # Get dashboard title first for better output
            response = requests.get(url, headers=headers)
            response.raise_for_status()
            dashboard = response.json()
            title = dashboard.get("title", "Unknown")

            # Delete it
            response = requests.delete(url, headers=headers)
            response.raise_for_status()
            print(f"✓ Deleted: {title} (ID: {dashboard_id})")
            return True
        except requests.exceptions.RequestException as e:
            print(f"✗ Error deleting dashboard {dashboard_id}: {e}")
            return False


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)

    args = sys.argv[1:]
    dry_run = False

    # Check for --dry-run flag
    if "--dry-run" in args:
        dry_run = True
        args.remove("--dry-run")

    if not args:
        print("Error: No dashboard IDs provided")
        print(__doc__)
        sys.exit(1)

    dashboard_ids = args

    if dry_run:
        print(f"Dry run - would delete {len(dashboard_ids)} dashboard(s):\n")
    else:
        print(f"Deleting {len(dashboard_ids)} dashboard(s)...\n")

    success_count = 0
    for dashboard_id in dashboard_ids:
        if delete_dashboard(dashboard_id, dry_run=dry_run):
            success_count += 1

    action = "would delete" if dry_run else "deleted"
    prefix = "[DRY RUN] " if dry_run else ""
    print(f"\n{prefix}Successfully {action} {success_count}/{len(dashboard_ids)} dashboard(s)")

    if success_count < len(dashboard_ids):
        sys.exit(1)


if __name__ == "__main__":
    main()
