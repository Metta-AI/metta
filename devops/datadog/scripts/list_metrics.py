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
    - DD_API_KEY environment variable
    - DD_APP_KEY environment variable
    - DD_SITE environment variable (optional, defaults to datadoghq.com)

Usage:
    export DD_API_KEY=your_api_key
    export DD_APP_KEY=your_app_key

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
import os
import sys
from typing import Any

import requests


class MetricsDiscovery:
    """Discovers available Datadog metrics and data sources."""

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

    def list_metrics(self, search: str | None = None) -> list[str]:
        """List all active metrics in the account.

        Args:
            search: Optional search filter

        Returns list of metric names
        """
        # Get active metrics from last 24 hours
        url = f"{self.base_url}/v1/metrics"
        params = {"from": "-86400"}  # Last 24 hours

        response = self._session.get(url, params=params)
        response.raise_for_status()

        data = response.json()
        metrics = data.get("metrics", [])

        if search:
            search_lower = search.lower()
            metrics = [m for m in metrics if search_lower in m.lower()]

        return sorted(metrics)

    def get_metric_metadata(self, metric_name: str) -> dict[str, Any]:
        """Get metadata for a specific metric."""
        url = f"{self.base_url}/v1/metrics/{metric_name}"
        response = self._session.get(url)
        response.raise_for_status()
        return response.json()

    def list_tags(self) -> dict[str, list[str]]:
        """List all available tags."""
        url = f"{self.base_url}/v1/tags/hosts"
        response = self._session.get(url)
        response.raise_for_status()
        return response.json()

    def categorize_metrics(self, metrics: list[str]) -> dict[str, list[str]]:
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
        discovery = MetricsDiscovery()
        categories = discovery.categorize_metrics(metrics)

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
        discovery = MetricsDiscovery()

        if args.tags:
            print("Fetching available tags...", file=sys.stderr)
            tags = discovery.list_tags()
            print(json.dumps(tags, indent=2))
            return

        print("Fetching metrics (last 24 hours)...", file=sys.stderr)
        metrics = discovery.list_metrics(search=args.search)

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
        print("\nSet environment variables:", file=sys.stderr)
        print("  export DD_API_KEY=your_api_key", file=sys.stderr)
        print("  export DD_APP_KEY=your_app_key", file=sys.stderr)
        sys.exit(1)
    except requests.exceptions.RequestException as e:
        print(f"API Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
