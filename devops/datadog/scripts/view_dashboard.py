#!/usr/bin/env -S uv run --quiet --script
# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "requests",
#     "datadog-api-client",
# ]
# ///
"""View live Datadog dashboard widget values.

Fetches a dashboard definition and queries all widget metrics to display
current values, helping troubleshoot dashboard issues.

Requirements:
    - DD_API_KEY environment variable (or AWS Secrets Manager)
    - DD_APP_KEY environment variable (or AWS Secrets Manager)
    - DD_SITE environment variable (optional, defaults to datadoghq.com)

Usage:
    # View dashboard by ID
    ./devops/datadog/scripts/view_dashboard.py dr3-pdj-rrw

    # View with verbose output
    ./devops/datadog/scripts/view_dashboard.py dr3-pdj-rrw --verbose

    # Export to JSON
    ./devops/datadog/scripts/view_dashboard.py dr3-pdj-rrw --format=json
"""

import argparse
import json
import logging
import re
import sys
import time
from typing import Any

from devops.datadog.utils.dashboard_client import DatadogDashboardClient
from devops.datadog.utils.datadog_client import DatadogClient

logger = logging.getLogger(__name__)


def extract_query_from_widget(widget: dict[str, Any]) -> list[str]:
    """Extract metric queries from a widget definition.

    Supports query_value, timeseries, and other common widget types.
    """
    queries = []
    definition = widget.get("definition", {})

    # Handle different widget types
    requests = definition.get("requests", [])
    if requests:
        for req in requests:
            # Check for query_value widgets (single metric)
            if "q" in req:
                queries.append(req["q"])
            # Check for formulas/queries structure
            elif "queries" in req:
                for q in req["queries"]:
                    if "query" in q:
                        queries.append(q["query"])
            # Check for APM/metrics queries
            elif "apm_query" in req:
                metric = req["apm_query"].get("compute", {}).get("metric")
                if metric:
                    queries.append(metric)
            elif "log_query" in req:
                metric = req["log_query"].get("compute", {}).get("metric")
                if metric:
                    queries.append(metric)

    return queries


def parse_query(query: str) -> dict[str, str] | None:
    """Parse a Datadog query string to extract metric name and aggregation.

    Examples:
        "avg:github.ci.tests_passing{*}" -> {"metric": "github.ci.tests_passing", "agg": "avg"}
        "sum:wandb.runs.completed_7d{*}" -> {"metric": "wandb.runs.completed_7d", "agg": "sum"}
    """
    # Match pattern: aggregation:metric_name{tags}
    match = re.match(r"(\w+):([a-z0-9_.]+)(?:\{[^}]*\})?", query)
    if match:
        return {
            "aggregation": match.group(1),
            "metric": match.group(2),
        }

    # If no aggregation prefix, try just metric name
    match = re.match(r"([a-z0-9_.]+)(?:\{[^}]*\})?", query)
    if match:
        return {
            "aggregation": "avg",
            "metric": match.group(1),
        }

    return None


def query_widget_value(
    dd_client: DatadogClient,
    query: str,
    lookback_seconds: int = 3600,
) -> float | None:
    """Query a widget's metric value."""
    parsed = parse_query(query)
    if not parsed:
        logger.warning(f"Could not parse query: {query}")
        return None

    metric_name = parsed["metric"]
    aggregation = parsed["aggregation"]

    logger.debug(f"Querying {metric_name} with {aggregation} aggregation")

    return dd_client.query_metric(
        metric_name=metric_name,
        aggregation=aggregation,
        lookback_seconds=lookback_seconds,
    )


def view_dashboard(
    dashboard_id: str,
    verbose: bool = False,
    lookback_seconds: int = 3600,
) -> dict[str, Any]:
    """Fetch and display dashboard widget values.

    Args:
        dashboard_id: Dashboard ID (e.g., "dr3-pdj-rrw")
        verbose: Enable verbose logging
        lookback_seconds: Lookback window for metric queries

    Returns:
        Dict with dashboard info and widget values
    """
    if verbose:
        logging.basicConfig(level=logging.DEBUG)
    else:
        logging.basicConfig(level=logging.INFO)

    # Initialize clients
    dashboard_client = DatadogDashboardClient()
    dd_client = DatadogClient(
        api_key=dashboard_client.api_key,
        app_key=dashboard_client.app_key,
        site=dashboard_client.site,
    )

    # Fetch dashboard
    print(f"Fetching dashboard: {dashboard_id}", file=sys.stderr)
    dashboard = dashboard_client.get_dashboard(dashboard_id)

    title = dashboard.get("title", "Unknown")
    description = dashboard.get("description", "")
    url = dashboard.get("url", "")

    print(f"\n{'=' * 80}", file=sys.stderr)
    print(f"Dashboard: {title}", file=sys.stderr)
    if description:
        print(f"Description: {description}", file=sys.stderr)
    if url:
        print(f"URL: {url}", file=sys.stderr)
    print(f"{'=' * 80}\n", file=sys.stderr)

    # Process widgets
    widgets = dashboard.get("widgets", [])
    widget_values = []

    for i, widget in enumerate(widgets, 1):
        widget_title = widget.get("definition", {}).get("title", f"Widget {i}")
        widget_type = widget.get("definition", {}).get("type", "unknown")

        print(f"\n{i}. {widget_title} ({widget_type})", file=sys.stderr)

        # Extract and query metrics
        queries = extract_query_from_widget(widget)

        if not queries:
            print("   [No queryable metrics]", file=sys.stderr)
            widget_values.append(
                {
                    "title": widget_title,
                    "type": widget_type,
                    "queries": [],
                    "values": {},
                }
            )
            continue

        values = {}
        for query in queries:
            print(f"   Query: {query}", file=sys.stderr)
            value = query_widget_value(dd_client, query, lookback_seconds)
            values[query] = value

            if value is not None:
                print(f"   → Value: {value}", file=sys.stderr)
            else:
                print("   → No data", file=sys.stderr)

        widget_values.append(
            {
                "title": widget_title,
                "type": widget_type,
                "queries": queries,
                "values": values,
            }
        )

    return {
        "dashboard_id": dashboard_id,
        "title": title,
        "description": description,
        "url": url,
        "widget_count": len(widgets),
        "widgets": widget_values,
        "timestamp": int(time.time()),
    }


def format_summary(result: dict[str, Any]) -> str:
    """Format dashboard result as human-readable summary."""
    lines = [
        f"\n{'=' * 80}",
        f"Dashboard: {result['title']}",
        f"ID: {result['dashboard_id']}",
        f"Widgets: {result['widget_count']}",
        f"{'=' * 80}\n",
    ]

    for widget in result["widgets"]:
        lines.append(f"• {widget['title']} ({widget['type']})")

        if not widget["values"]:
            lines.append("  [No data]")
        else:
            for query, value in widget["values"].items():
                if value is not None:
                    lines.append(f"  {query} = {value}")
                else:
                    lines.append(f"  {query} = [No data]")

        lines.append("")

    return "\n".join(lines)


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="View live Datadog dashboard widget values",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # View WandB Training Metrics dashboard
  %(prog)s dr3-pdj-rrw

  # View with verbose logging
  %(prog)s dr3-pdj-rrw --verbose

  # Export to JSON
  %(prog)s dr3-pdj-rrw --format=json > dashboard_snapshot.json
        """,
    )
    parser.add_argument(
        "dashboard_id",
        help="Dashboard ID (e.g., 'dr3-pdj-rrw')",
    )
    parser.add_argument(
        "--format",
        choices=["summary", "json"],
        default="summary",
        help="Output format (default: summary)",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Enable verbose logging",
    )
    parser.add_argument(
        "--lookback",
        type=int,
        default=3600,
        help="Lookback window in seconds (default: 3600 = 1 hour)",
    )

    args = parser.parse_args()

    try:
        result = view_dashboard(
            dashboard_id=args.dashboard_id,
            verbose=args.verbose,
            lookback_seconds=args.lookback,
        )

        if args.format == "json":
            print(json.dumps(result, indent=2))
        else:
            print(format_summary(result))

    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        if args.verbose:
            import traceback

            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
