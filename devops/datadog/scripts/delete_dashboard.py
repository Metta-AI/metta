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
    DD_API_KEY: Datadog API key (or AWS Secrets Manager)
    DD_APP_KEY: Datadog application key (or AWS Secrets Manager)
    DD_SITE: Datadog site (default: datadoghq.com)
"""

import sys
from pathlib import Path

# Add parent directory to path for local imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from common.dashboard_client import DatadogDashboardClient


def delete_dashboard(client: DatadogDashboardClient, dashboard_id: str, dry_run: bool = False) -> bool:
    """Delete a dashboard from Datadog.

    Args:
        client: Datadog dashboard client
        dashboard_id: The dashboard ID to delete
        dry_run: If True, only print what would be deleted

    Returns:
        True if successful, False otherwise
    """
    try:
        # Get dashboard info first
        dashboard = client.get_dashboard(dashboard_id)
        title = dashboard.get("title", "Unknown")

        if dry_run:
            print(f"[DRY RUN] Would delete: {title} (ID: {dashboard_id})")
            return True
        else:
            # Actually delete
            client.delete_dashboard(dashboard_id)
            print(f"✓ Deleted: {title} (ID: {dashboard_id})")
            return True

    except Exception as e:
        action = "[DRY RUN] Error fetching" if dry_run else "✗ Error deleting"
        print(f"{action} dashboard {dashboard_id}: {e}")
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

    try:
        client = DatadogDashboardClient()

        if dry_run:
            print(f"Dry run - would delete {len(dashboard_ids)} dashboard(s):\n")
        else:
            print(f"Deleting {len(dashboard_ids)} dashboard(s)...\n")

        success_count = 0
        for dashboard_id in dashboard_ids:
            if delete_dashboard(client, dashboard_id, dry_run=dry_run):
                success_count += 1

        action = "would delete" if dry_run else "deleted"
        prefix = "[DRY RUN] " if dry_run else ""
        print(f"\n{prefix}Successfully {action} {success_count}/{len(dashboard_ids)} dashboard(s)")

        if success_count < len(dashboard_ids):
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
