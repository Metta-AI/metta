#!/usr/bin/env python3
"""List all metrics from Datadog in the last N minutes.

Useful for verifying which metrics are available in Datadog.
"""

import argparse
from collections import defaultdict
from datetime import datetime, timedelta

from datadog_api_client import ApiClient, Configuration
from datadog_api_client.v1.api.metrics_api import MetricsApi

from devops.datadog.utils.dashboard_client import get_datadog_credentials


def list_metrics(minutes: int = 60, prefix: str | None = None, summary: bool = False) -> list[str]:
    """List all active metrics from Datadog.

    Args:
        minutes: How many minutes back to search
        prefix: Optional prefix filter (e.g., "github")
        summary: If True, show summary by prefix

    Returns:
        List of metric names
    """
    config = Configuration()
    api_key, app_key, site = get_datadog_credentials()
    config.api_key["apiKeyAuth"] = api_key
    config.api_key["appKeyAuth"] = app_key
    config.server_variables["site"] = site

    with ApiClient(config) as api_client:
        api = MetricsApi(api_client)
        from_time = int((datetime.now() - timedelta(minutes=minutes)).timestamp())

        response = api.list_active_metrics(from_time)
        all_metrics = response.metrics if hasattr(response, "metrics") else []

        # Filter by prefix if specified
        if prefix:
            all_metrics = [m for m in all_metrics if m.startswith(prefix)]

        return sorted(all_metrics)


def show_summary(metrics: list[str]) -> None:
    """Show metrics grouped by prefix."""
    # Group by first component (before first dot)
    by_prefix = defaultdict(list)
    for metric in metrics:
        prefix = metric.split(".")[0] if "." in metric else metric
        by_prefix[prefix].append(metric)

    print("\n=== Summary by Prefix ===")
    for prefix in sorted(by_prefix.keys()):
        count = len(by_prefix[prefix])
        print(f"  {prefix}.*: {count} metrics")

    print(f"\nTotal: {len(metrics)} metrics")


def main():
    parser = argparse.ArgumentParser(description="List metrics from Datadog")
    parser.add_argument(
        "--minutes",
        type=int,
        default=60,
        help="How many minutes back to search (default: 60)",
    )
    parser.add_argument(
        "--prefix",
        type=str,
        default=None,
        help="Filter metrics by prefix (e.g., 'github', 'wandb')",
    )
    parser.add_argument(
        "--summary",
        action="store_true",
        help="Show summary grouped by prefix",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Show full list of metrics",
    )

    args = parser.parse_args()

    print(f"=== Datadog Metrics (last {args.minutes} minutes) ===")
    if args.prefix:
        print(f"Filter: {args.prefix}.*")

    metrics = list_metrics(minutes=args.minutes, prefix=args.prefix)

    print(f"\nFound {len(metrics)} metrics")

    if args.verbose and metrics:
        print("\n=== Metrics ===")
        for metric in metrics:
            print(f"  {metric}")

    if args.summary:
        show_summary(metrics)

    if not metrics:
        if args.prefix:
            print(f"\n⚠️  No metrics found with prefix '{args.prefix}'")
        else:
            print("\n⚠️  No metrics found in time range")


if __name__ == "__main__":
    main()
