#!/usr/bin/env -S uv run --quiet --script
# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "requests",
# ]
# ///
"""List available Datadog metrics and data sources.

This script helps discover what metrics are available in your Datadog account,
making it easier to build modular dashboard components.

Requirements:
    - DD_API_KEY environment variable (or AWS Secrets Manager)
    - DD_APP_KEY environment variable (or AWS Secrets Manager)
    - DD_SITE environment variable (optional, defaults to datadoghq.com)

Usage:
    # List all metrics
    ./list_metrics.py

    # Search for specific metrics
    ./list_metrics.py --search=cpu

    # List metrics with tags
    ./list_metrics.py --tags

    # Output to file
    ./list_metrics.py > metrics.txt
"""

import json
import sys

from devops.datadog.utils.dashboard_client import DatadogDashboardClient


def categorize_metrics(metrics: list[str]) -> dict[str, list[str]]:
    """Categorize metrics by prefix/namespace."""
    categories = {}

    for metric in metrics:
        # Get prefix (first part before .)
        parts = metric.split(".")
        prefix = parts[0] if parts else "other"

        if prefix not in categories:
            categories[prefix] = []
        categories[prefix].append(metric)

    return categories


def format_metrics_summary(metrics: list[str], show_categories: bool = True) -> str:
    """Format metrics as human-readable summary."""
    lines = [
        "Available Metrics",
        "=" * 80,
        f"Total: {len(metrics)} metrics",
        "",
    ]

    if show_categories:
        categories = categorize_metrics(metrics)

        lines.append("Metrics by Category:")
        lines.append("-" * 80)

        for category, cat_metrics in sorted(categories.items()):
            lines.append(f"\n{category}: ({len(cat_metrics)} metrics)")
            # Show first 10 metrics in each category
            for metric in cat_metrics[:10]:
                lines.append(f"  - {metric}")
            if len(cat_metrics) > 10:
                lines.append(f"  ... and {len(cat_metrics) - 10} more")
    else:
        lines.append("Metrics:")
        lines.append("-" * 80)
        for metric in metrics:
            lines.append(f"  {metric}")

    return "\n".join(lines)


def main():
    """Main entry point."""
    import argparse

    parser = argparse.ArgumentParser(description="List available Datadog metrics and data sources")
    parser.add_argument(
        "--search",
        help="Search for metrics containing this string",
    )
    parser.add_argument(
        "--tags",
        action="store_true",
        help="List available tags instead of metrics",
    )
    parser.add_argument(
        "--format",
        choices=["summary", "list", "json"],
        default="summary",
        help="Output format (default: summary)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        help="Limit number of metrics shown",
    )

    args = parser.parse_args()

    try:
        client = DatadogDashboardClient()

        if args.tags:
            print("Fetching available tags...", file=sys.stderr)
            tags = client.list_tags()
            print(json.dumps(tags, indent=2))
            return

        print("Fetching metrics (last 24 hours)...", file=sys.stderr)
        metrics = client.list_metrics(search=args.search)

        if args.limit:
            metrics = metrics[: args.limit]

        print(f"Found {len(metrics)} metrics", file=sys.stderr)
        print("", file=sys.stderr)

        if args.format == "json":
            print(json.dumps(metrics, indent=2))
        elif args.format == "list":
            for metric in metrics:
                print(metric)
        else:  # summary
            print(format_metrics_summary(metrics))

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
