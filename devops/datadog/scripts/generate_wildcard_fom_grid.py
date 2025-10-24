#!/usr/bin/env python3
"""Generate System Health FoM Grid using Datadog Wildcard widget with Vega-Lite.

Creates a single Wildcard widget that displays a 7×7 grid of FoM metrics over 7 days.
Uses Vega-Lite for visualization with data transforms to reshape Datadog query results.
"""

import json
from pathlib import Path

# Define metrics (rows)
METRICS = [
    {
        "name": "Tests Passing",
        "metric": "health.ci.tests_passing.fom",
        "key": "tests_passing",
    },
    {
        "name": "Failing Workflows",
        "metric": "health.ci.failing_workflows.fom",
        "key": "failing_workflows",
    },
    {
        "name": "Hotfix Count",
        "metric": "health.ci.hotfix_count.fom",
        "key": "hotfix_count",
    },
    {
        "name": "Revert Count",
        "metric": "health.ci.revert_count.fom",
        "key": "revert_count",
    },
    {
        "name": "CI Duration P90",
        "metric": "health.ci.duration_p90.fom",
        "key": "duration_p90",
    },
    {
        "name": "Stale PRs",
        "metric": "health.ci.stale_prs.fom",
        "key": "stale_prs",
    },
    {
        "name": "PR Cycle Time",
        "metric": "health.ci.pr_cycle_time.fom",
        "key": "pr_cycle_time",
    },
]

# Define days (columns) - oldest to newest
DAYS = [
    {"offset": -6, "label": "-6d", "key": "6d"},
    {"offset": -5, "label": "-5d", "key": "5d"},
    {"offset": -4, "label": "-4d", "key": "4d"},
    {"offset": -3, "label": "-3d", "key": "3d"},
    {"offset": -2, "label": "-2d", "key": "2d"},
    {"offset": -1, "label": "-1d", "key": "1d"},
    {"offset": 0, "label": "Today", "key": "today"},
]


def generate_queries() -> list[dict]:
    """Generate Datadog metric queries for all metric-day combinations.

    Returns:
        List of query configurations for Datadog API
    """
    queries = []

    for metric_def in METRICS:
        for day in DAYS:
            # Build query
            query = f"avg:{metric_def['metric']}{{*}}"
            if day["offset"] != 0:
                query += f".timeshift({day['offset']}d)"

            # Create query name: metric_key + day_key (e.g., "tests_passing_today")
            query_name = f"{metric_def['key']}_{day['key']}"

            queries.append(
                {
                    "query": query,
                    "data_source": "metrics",
                    "name": query_name,
                    "aggregator": "last",
                }
            )

    return queries


def generate_fold_fields() -> list[str]:
    """Generate list of all query result fields for Vega-Lite fold transform.

    Returns:
        List of field names like ["tests_passing_today", "tests_passing_1d", ...]
    """
    fields = []
    for metric_def in METRICS:
        for day in DAYS:
            fields.append(f"{metric_def['key']}_{day['key']}")
    return fields


def generate_vega_spec() -> dict:
    """Generate Vega-Lite specification for heatmap visualization.

    Returns:
        Vega-Lite spec dict
    """
    # Build field list for fold transform
    fold_fields = generate_fold_fields()

    # Build metric name mapping for sorting
    metric_sort_order = [m["name"] for m in METRICS]
    day_sort_order = [d["label"] for d in DAYS]

    return {
        "$schema": "https://vega.github.io/schema/vega-lite/v6.json",
        "data": {"name": "queryResults"},
        "transform": [
            # Unfold all 49 fields into rows
            {"fold": fold_fields, "as": ["metric_day", "value"]},
            # Extract metric name from field name
            # e.g., "tests_passing_today" -> metric_key = "tests_passing", day_key = "today"
            {
                "calculate": "split(datum.metric_day, '_')[0] + (indexof(datum.metric_day, 'passing') > 0 ? ' passing' : (indexof(datum.metric_day, 'workflows') > 0 ? ' workflows' : (indexof(datum.metric_day, 'count') > 0 ? ' count' : (indexof(datum.metric_day, 'p90') > 0 ? ' p90' : (indexof(datum.metric_day, 'prs') > 0 ? ' prs' : ' time')))))",
                "as": "metric_raw",
            },
            # Extract day label from field name
            {
                "calculate": "indexof(datum.metric_day, 'today') > 0 ? 'Today' : ('-' + split(datum.metric_day, '_')[length(split(datum.metric_day, '_')) - 1])",
                "as": "day",
            },
            # Create proper metric display names
            {
                "calculate": "replace(replace(replace(replace(replace(replace(replace(datum.metric_day, '_today', ''), '_1d', ''), '_2d', ''), '_3d', ''), '_4d', ''), '_5d', ''), '_6d', '')",
                "as": "metric_key",
            },
            {
                "calculate": "datum.metric_key == 'tests_passing' ? 'Tests Passing' : (datum.metric_key == 'failing_workflows' ? 'Failing Workflows' : (datum.metric_key == 'hotfix_count' ? 'Hotfix Count' : (datum.metric_key == 'revert_count' ? 'Revert Count' : (datum.metric_key == 'duration_p90' ? 'CI Duration P90' : (datum.metric_key == 'stale_prs' ? 'Stale PRs' : 'PR Cycle Time')))))",
                "as": "metric",
            },
        ],
        "layer": [
            # Layer 1: Rectangle heatmap with color encoding
            {
                "mark": {"type": "rect", "stroke": "#333", "strokeWidth": 1},
                "encoding": {
                    "y": {
                        "field": "metric",
                        "type": "nominal",
                        "title": None,
                        "sort": metric_sort_order,
                        "axis": {"labelAngle": 0, "labelPadding": 10},
                    },
                    "x": {
                        "field": "day",
                        "type": "ordinal",
                        "title": None,
                        "sort": day_sort_order,
                        "axis": {"labelAngle": 0},
                    },
                    "color": {
                        "field": "value",
                        "type": "quantitative",
                        "scale": {
                            "domain": [0, 0.3, 0.7, 1.0],
                            "range": ["#dc3545", "#ffc107", "#ffc107", "#28a745"],
                            "type": "threshold",
                        },
                        "legend": {"title": "FoM Score"},
                    },
                    "tooltip": [
                        {"field": "metric", "type": "nominal", "title": "Metric"},
                        {"field": "day", "type": "ordinal", "title": "Day"},
                        {
                            "field": "value",
                            "type": "quantitative",
                            "title": "FoM",
                            "format": ".3f",
                        },
                    ],
                },
            },
            # Layer 2: Text labels with values
            {
                "mark": {"type": "text", "fontSize": 14, "fontWeight": "bold"},
                "encoding": {
                    "y": {
                        "field": "metric",
                        "type": "nominal",
                        "sort": metric_sort_order,
                    },
                    "x": {"field": "day", "type": "ordinal", "sort": day_sort_order},
                    "text": {"field": "value", "type": "quantitative", "format": ".2f"},
                    "color": {
                        "condition": {"test": "datum.value < 0.5", "value": "white"},
                        "value": "black",
                    },
                },
            },
        ],
        "config": {
            "view": {"stroke": None},
            "axis": {"grid": False, "domain": False, "ticks": False},
        },
    }


def generate_dashboard() -> dict:
    """Generate complete dashboard JSON with Wildcard widget.

    Returns:
        Dashboard configuration dict
    """
    queries = generate_queries()
    vega_spec = generate_vega_spec()

    widget = {
        "definition": {
            "title": "System Health FoM Grid (7×7)",
            "title_size": "16",
            "title_align": "left",
            "type": "wildcard",
            "requests": [
                {
                    "response_format": "scalar",
                    "queries": queries,
                }
            ],
            "custom_viz": vega_spec,
        },
        "layout": {"x": 0, "y": 0, "width": 12, "height": 8},
    }

    # Add description widget
    description = {
        "definition": {
            "type": "note",
            "content": (
                "## System Health Rollup\n\n"
                "**Figure of Merit (FoM)** values for CI/CD metrics over 7 days.\n\n"
                "🟢 Green (0.7-1.0) = Healthy | "
                "🟡 Yellow (0.3-0.7) = Warning | "
                "🔴 Red (0.0-0.3) = Critical\n\n"
                "Hover over cells for exact values. Click for detailed metric views."
            ),
            "background_color": "gray",
            "font_size": "14",
            "text_align": "left",
            "show_tick": False,
            "has_padding": True,
        },
        "layout": {"x": 0, "y": 8, "width": 12, "height": 2},
    }

    dashboard = {
        "title": "System Health Rollup (Wildcard)",
        "description": "Figure of Merit (FoM) grid using Wildcard widget with Vega-Lite visualization - no external storage required!",
        "layout_type": "ordered",
        "template_variables": [],
        "widgets": [widget, description],
        "notify_list": [],
        "reflow_type": "fixed",
    }

    return dashboard


def main():
    """Generate and save dashboard JSON."""
    # Generate dashboard
    dashboard = generate_dashboard()

    # Save to templates directory
    output_path = Path(__file__).parent.parent / "templates" / "system_health_rollup_wildcard.json"

    with open(output_path, "w") as f:
        json.dump(dashboard, f, indent=2)

    queries = generate_queries()
    print("✓ Generated Wildcard widget dashboard")
    print(f"  - 1 Wildcard widget with {len(queries)} metric queries")
    print(f"  - {len(METRICS)} metrics × {len(DAYS)} days = {len(METRICS) * len(DAYS)} data cells")
    print("  - Vega-Lite heatmap with text overlays")
    print("  - No external storage required (S3-free!)")
    print(f"\nSaved to: {output_path}")
    print("\nNext steps:")
    print("  1. Review: cat templates/system_health_rollup_wildcard.json")
    print("  2. Push: uv run python cli.py dashboard push --template wildcard")
    print("  3. Compare with current grid: Both use Datadog-managed data!")
    print("\nAdvantages over widget grid:")
    print("  ✓ 1 widget instead of 65")
    print("  ✓ Text labels showing exact values")
    print("  ✓ Better grid alignment")
    print("  ✓ Interactive tooltips")
    print("  ✓ Easier to maintain")


if __name__ == "__main__":
    main()
