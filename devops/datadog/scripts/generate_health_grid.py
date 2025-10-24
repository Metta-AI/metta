#!/usr/bin/env python3
"""Generate System Health Rollup dashboard with grid layout.

Creates a 7x7 grid of query_value widgets showing FoM metrics over 7 days.
Each cell is color-coded: Green (0.7-1.0), Yellow (0.3-0.7), Red (0.0-0.3).
"""

import json
from pathlib import Path

# Define metrics (rows)
METRICS = [
    {
        "name": "Tests Passing",
        "metric": "health.ci.tests_passing.fom",
        "description": "All tests passing on main",
    },
    {
        "name": "Failing Workflows",
        "metric": "health.ci.failing_workflows.fom",
        "description": "Failed workflows (7d)",
    },
    {
        "name": "Hotfix Count",
        "metric": "health.ci.hotfix_count.fom",
        "description": "Hotfixes (7d)",
    },
    {
        "name": "Revert Count",
        "metric": "health.ci.revert_count.fom",
        "description": "Reverts (7d)",
    },
    {
        "name": "CI Duration P90",
        "metric": "health.ci.duration_p90.fom",
        "description": "CI pipeline duration",
    },
    {
        "name": "Stale PRs",
        "metric": "health.ci.stale_prs.fom",
        "description": "Stale PRs (14d)",
    },
    {
        "name": "PR Cycle Time",
        "metric": "health.ci.pr_cycle_time.fom",
        "description": "PR cycle time",
    },
]

# Define days (columns)
DAYS = [
    {"offset": 0, "label": "Today"},
    {"offset": -1, "label": "-1d"},
    {"offset": -2, "label": "-2d"},
    {"offset": -3, "label": "-3d"},
    {"offset": -4, "label": "-4d"},
    {"offset": -5, "label": "-5d"},
    {"offset": -6, "label": "-6d"},
]

# Layout configuration (must fit in Datadog's 12-column grid)
CELL_WIDTH = 1  # 7 days × 1 = 7 columns
CELL_HEIGHT = 2
HEADER_HEIGHT = 1
ROW_LABEL_WIDTH = 5  # 5 for labels + 7 for data = 12 total ✓


def create_query_value_widget(metric: str, day_offset: int, title: str, x: int, y: int) -> dict:
    """Create a single query_value widget with conditional formatting.

    Args:
        metric: Datadog metric name (e.g., "health.ci.tests_passing.fom")
        day_offset: Days to shift (0 = today, -1 = yesterday, etc.)
        title: Widget title
        x: X position in grid
        y: Y position in grid

    Returns:
        Widget configuration dict
    """
    # Build query with timeshift if needed
    query = f"avg:{metric}{{*}}"
    if day_offset != 0:
        query += f".timeshift({day_offset}d)"

    return {
        "definition": {
            "title": title,
            "title_size": "13",
            "title_align": "center",
            "type": "query_value",
            "requests": [
                {
                    "formulas": [{"formula": "query1"}],
                    "queries": [
                        {
                            "data_source": "metrics",
                            "name": "query1",
                            "query": query,
                            "aggregator": "last",
                        }
                    ],
                    "response_format": "scalar",
                    "conditional_formats": [
                        {
                            "comparator": ">=",
                            "value": 0.7,
                            "palette": "white_on_green",
                        },
                        {
                            "comparator": ">=",
                            "value": 0.3,
                            "palette": "white_on_yellow",
                        },
                        {
                            "comparator": "<",
                            "value": 0.3,
                            "palette": "white_on_red",
                        },
                    ],
                }
            ],
            "autoscale": False,
            "precision": 2,
        },
        "layout": {
            "x": x,
            "y": y,
            "width": CELL_WIDTH,
            "height": CELL_HEIGHT,
        },
    }


def create_note_widget(content: str, x: int, y: int, width: int, height: int) -> dict:
    """Create a note widget for labels/headers."""
    return {
        "definition": {
            "type": "note",
            "content": content,
            "background_color": "gray",
            "font_size": "14",
            "text_align": "center",
            "show_tick": False,
            "tick_pos": "50%",
            "tick_edge": "left",
            "has_padding": True,
        },
        "layout": {
            "x": x,
            "y": y,
            "width": width,
            "height": height,
        },
    }


def generate_dashboard() -> dict:
    """Generate complete dashboard JSON with grid layout."""
    widgets = []

    # Add title/description at top
    widgets.append(
        create_note_widget(
            content=(
                "## System Health Rollup\n\n"
                "**Figure of Merit (FoM)** values for CI/CD metrics over 7 days.\n\n"
                "🟢 Green (0.7-1.0) = Healthy | "
                "🟡 Yellow (0.3-0.7) = Warning | "
                "🔴 Red (0.0-0.3) = Critical"
            ),
            x=0,
            y=0,
            width=ROW_LABEL_WIDTH + (len(DAYS) * CELL_WIDTH),
            height=HEADER_HEIGHT,
        )
    )

    # Add column headers (day labels)
    for col, day in enumerate(DAYS):
        x = ROW_LABEL_WIDTH + (col * CELL_WIDTH)
        y = HEADER_HEIGHT
        widgets.append(
            create_note_widget(
                content=f"**{day['label']}**",
                x=x,
                y=y,
                width=CELL_WIDTH,
                height=HEADER_HEIGHT,
            )
        )

    # Add data grid: rows (metrics) × columns (days)
    for row, metric_def in enumerate(METRICS):
        # Row label (metric name)
        y = HEADER_HEIGHT + HEADER_HEIGHT + (row * CELL_HEIGHT)
        widgets.append(
            create_note_widget(
                content=f"**{metric_def['name']}**\n_{metric_def['description']}_",
                x=0,
                y=y,
                width=ROW_LABEL_WIDTH,
                height=CELL_HEIGHT,
            )
        )

        # Data cells for this metric across all days
        for col, day in enumerate(DAYS):
            x = ROW_LABEL_WIDTH + (col * CELL_WIDTH)

            # Create title for this cell (hide it to keep grid clean)
            title = ""  # Empty title for cleaner look

            widget = create_query_value_widget(
                metric=metric_def["metric"],
                day_offset=day["offset"],
                title=title,
                x=x,
                y=y,
            )
            widgets.append(widget)

    # Build complete dashboard
    dashboard = {
        "title": "System Health Rollup",
        "description": "Figure of Merit (FoM) grid showing CI/CD health metrics over 7 days with color-coded status.",
        "layout_type": "ordered",
        "template_variables": [],
        "widgets": widgets,
        "notify_list": [],
        "reflow_type": "fixed",
    }

    return dashboard


def main():
    """Generate and save dashboard JSON."""
    # Generate dashboard
    dashboard = generate_dashboard()

    # Save to templates directory
    output_path = Path(__file__).parent.parent / "templates" / "system_health_rollup.json"

    with open(output_path, "w") as f:
        json.dump(dashboard, f, indent=2)

    print(f"✓ Generated dashboard with {len(dashboard['widgets'])} widgets")
    print(f"  - {len(METRICS)} metrics × {len(DAYS)} days = {len(METRICS) * len(DAYS)} data cells")
    print(f"  - {len(METRICS)} row labels + {len(DAYS)} column headers = {len(METRICS) + len(DAYS)} labels")
    print("  - 1 title note")
    print(f"  - Total: {len(dashboard['widgets'])} widgets")
    print(f"\nSaved to: {output_path}")
    print("\nNext steps:")
    print("  1. Review: cat templates/system_health_rollup.json")
    print("  2. Push: uv run python cli.py dashboard push")
    print("  3. View: https://app.datadoghq.com/dashboard/2mx-kfj-8pi/system-health-rollup")


if __name__ == "__main__":
    main()
