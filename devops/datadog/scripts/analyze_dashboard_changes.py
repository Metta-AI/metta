#!/usr/bin/env -S uv run --quiet --script
# /// script
# requires-python = ">=3.11"
# dependencies = []
# ///
"""Analyze changes between two dashboard versions.

This script helps understand what changed when a user edits a dashboard
in the Datadog UI. It compares two dashboard JSON exports and produces
a human-readable summary of changes.

Usage:
    ./analyze_dashboard_changes.py before.json after.json
    ./analyze_dashboard_changes.py before.json after.json --section="Push-to-Main"
    ./analyze_dashboard_changes.py before.json after.json --format=markdown
"""

import argparse
import json
import sys
from typing import Any


def load_dashboard(path: str) -> dict[str, Any]:
    """Load dashboard JSON from file."""
    with open(path) as f:
        return json.load(f)


def normalize_widget(widget: dict[str, Any]) -> dict[str, Any]:
    """Normalize widget for comparison (remove auto-generated fields)."""
    normalized = widget.copy()
    # Remove fields that change automatically
    normalized.pop("id", None)
    return normalized


def extract_widget_summary(widget: dict[str, Any]) -> dict[str, Any]:
    """Extract key information from widget for comparison."""
    definition = widget.get("definition", {})
    layout = widget.get("layout", {})

    summary = {
        "type": definition.get("type", "unknown"),
        "title": definition.get("title") or definition.get("content", "")[:50],
        "layout": {
            "x": layout.get("x"),
            "y": layout.get("y"),
            "width": layout.get("width"),
            "height": layout.get("height"),
        },
    }

    # Extract queries for query-based widgets
    if summary["type"] in ["timeseries", "query_value"]:
        queries = []
        for request in definition.get("requests", []):
            for query in request.get("queries", []):
                if "query" in query:
                    queries.append(query["query"])
        summary["queries"] = queries

    return summary


def find_section_widgets(widgets: list[dict], section_name: str) -> list[dict]:
    """Find widgets in a specific section (between note headers)."""
    section_start = None
    section_end = None

    for i, widget in enumerate(widgets):
        content = widget.get("definition", {}).get("content", "")
        if section_name.lower() in content.lower():
            section_start = i
        elif section_start is not None and widget.get("definition", {}).get("type") == "note":
            section_end = i
            break

    if section_start is None:
        return []

    if section_end is None:
        section_end = len(widgets)

    return widgets[section_start:section_end]


def compare_dashboards(
    before: dict[str, Any],
    after: dict[str, Any],
    section: str | None = None,
) -> dict[str, Any]:
    """Compare two dashboards and return changes."""
    changes = {
        "metadata": {},
        "widgets_added": [],
        "widgets_removed": [],
        "widgets_modified": [],
        "widgets_moved": [],
        "summary": {},
    }

    # Compare metadata
    if before.get("title") != after.get("title"):
        changes["metadata"]["title"] = {
            "before": before.get("title"),
            "after": after.get("title"),
        }

    if before.get("description") != after.get("description"):
        changes["metadata"]["description"] = {
            "before": before.get("description"),
            "after": after.get("description"),
        }

    # Get widgets
    before_widgets = before.get("widgets", [])
    after_widgets = after.get("widgets", [])

    # Filter by section if requested
    if section:
        before_widgets = find_section_widgets(before_widgets, section)
        after_widgets = find_section_widgets(after_widgets, section)

    # Compare widgets
    before_summaries = [extract_widget_summary(w) for w in before_widgets]
    after_summaries = [extract_widget_summary(w) for w in after_widgets]

    # Find added/removed widgets (by comparing titles and queries)
    before_keys = [(w["title"], tuple(w.get("queries", []))) for w in before_summaries]
    after_keys = [(w["title"], tuple(w.get("queries", []))) for w in after_summaries]

    for i, (summary, key) in enumerate(zip(after_summaries, after_keys, strict=True)):
        if key not in before_keys:
            changes["widgets_added"].append(
                {
                    "index": i,
                    "type": summary["type"],
                    "title": summary["title"],
                    "layout": summary["layout"],
                }
            )

    for i, (summary, key) in enumerate(zip(before_summaries, before_keys, strict=True)):
        if key not in after_keys:
            changes["widgets_removed"].append(
                {
                    "index": i,
                    "type": summary["type"],
                    "title": summary["title"],
                }
            )

    # Find moved/modified widgets
    for _after_idx, (after_summary, after_key) in enumerate(zip(after_summaries, after_keys, strict=True)):
        if after_key in before_keys:
            before_idx = before_keys.index(after_key)
            before_summary = before_summaries[before_idx]

            # Check if moved
            if before_summary["layout"] != after_summary["layout"]:
                changes["widgets_moved"].append(
                    {
                        "title": after_summary["title"],
                        "before_layout": before_summary["layout"],
                        "after_layout": after_summary["layout"],
                    }
                )

            # Check if modified (title changed)
            if before_summary["title"] != after_summary["title"]:
                changes["widgets_modified"].append(
                    {
                        "type": "title_change",
                        "before_title": before_summary["title"],
                        "after_title": after_summary["title"],
                        "queries": after_summary.get("queries", []),
                    }
                )

    # Summary statistics
    changes["summary"] = {
        "before_count": len(before_widgets),
        "after_count": len(after_widgets),
        "added": len(changes["widgets_added"]),
        "removed": len(changes["widgets_removed"]),
        "moved": len(changes["widgets_moved"]),
        "modified": len(changes["widgets_modified"]),
    }

    return changes


def format_text(changes: dict[str, Any]) -> str:
    """Format changes as plain text."""
    lines = []

    lines.append("Dashboard Changes Analysis")
    lines.append("=" * 80)
    lines.append("")

    # Summary
    summary = changes["summary"]
    lines.append("Summary:")
    lines.append(f"  Before: {summary['before_count']} widgets")
    lines.append(f"  After:  {summary['after_count']} widgets")
    lines.append(f"  Added:  {summary['added']}")
    lines.append(f"  Removed: {summary['removed']}")
    lines.append(f"  Moved:  {summary['moved']}")
    lines.append(f"  Modified: {summary['modified']}")
    lines.append("")

    # Metadata changes
    if changes["metadata"]:
        lines.append("Metadata Changes:")
        for key, value in changes["metadata"].items():
            lines.append(f"  {key}:")
            lines.append(f"    Before: {value['before']}")
            lines.append(f"    After:  {value['after']}")
        lines.append("")

    # Added widgets
    if changes["widgets_added"]:
        lines.append("Widgets Added:")
        for widget in changes["widgets_added"]:
            layout = widget["layout"]
            lines.append(f"  + [{widget['type']}] {widget['title']}")
            lines.append(f"    Position: x={layout['x']}, y={layout['y']}, w={layout['width']}, h={layout['height']}")
        lines.append("")

    # Removed widgets
    if changes["widgets_removed"]:
        lines.append("Widgets Removed:")
        for widget in changes["widgets_removed"]:
            lines.append(f"  - [{widget['type']}] {widget['title']}")
        lines.append("")

    # Moved widgets
    if changes["widgets_moved"]:
        lines.append("Widgets Moved:")
        for widget in changes["widgets_moved"]:
            before = widget["before_layout"]
            after = widget["after_layout"]
            lines.append(f"  ↔ {widget['title']}")
            lines.append(f"    Before: x={before['x']}, y={before['y']}, w={before['width']}, h={before['height']}")
            lines.append(f"    After:  x={after['x']}, y={after['y']}, w={after['width']}, h={after['height']}")
        lines.append("")

    # Modified widgets
    if changes["widgets_modified"]:
        lines.append("Widgets Modified:")
        for widget in changes["widgets_modified"]:
            lines.append(f"  ✎ {widget['type']}")
            lines.append(f"    Before: {widget['before_title']}")
            lines.append(f"    After:  {widget['after_title']}")
            if widget.get("queries"):
                lines.append(f"    Queries: {', '.join(widget['queries'])}")
        lines.append("")

    return "\n".join(lines)


def format_markdown(changes: dict[str, Any]) -> str:
    """Format changes as markdown."""
    lines = []

    lines.append("# Dashboard Changes Analysis")
    lines.append("")

    # Summary
    summary = changes["summary"]
    lines.append("## Summary")
    lines.append("")
    lines.append(f"- **Before**: {summary['before_count']} widgets")
    lines.append(f"- **After**: {summary['after_count']} widgets")
    lines.append(f"- **Added**: {summary['added']}")
    lines.append(f"- **Removed**: {summary['removed']}")
    lines.append(f"- **Moved**: {summary['moved']}")
    lines.append(f"- **Modified**: {summary['modified']}")
    lines.append("")

    # Metadata changes
    if changes["metadata"]:
        lines.append("## Metadata Changes")
        lines.append("")
        for key, value in changes["metadata"].items():
            lines.append(f"### {key}")
            lines.append(f"- Before: `{value['before']}`")
            lines.append(f"- After: `{value['after']}`")
            lines.append("")

    # Added widgets
    if changes["widgets_added"]:
        lines.append("## Widgets Added")
        lines.append("")
        for widget in changes["widgets_added"]:
            layout = widget["layout"]
            lines.append(f"### ✅ {widget['title']}")
            lines.append(f"- Type: `{widget['type']}`")
            lines.append(f"- Position: `x={layout['x']}, y={layout['y']}, w={layout['width']}, h={layout['height']}`")
            lines.append("")

    # Removed widgets
    if changes["widgets_removed"]:
        lines.append("## Widgets Removed")
        lines.append("")
        for widget in changes["widgets_removed"]:
            lines.append(f"### ❌ {widget['title']}")
            lines.append(f"- Type: `{widget['type']}`")
            lines.append("")

    # Moved widgets
    if changes["widgets_moved"]:
        lines.append("## Widgets Moved")
        lines.append("")
        for widget in changes["widgets_moved"]:
            before = widget["before_layout"]
            after = widget["after_layout"]
            lines.append(f"### 📐 {widget['title']}")
            lines.append(f"- Before: `x={before['x']}, y={before['y']}, w={before['width']}, h={before['height']}`")
            lines.append(f"- After: `x={after['x']}, y={after['y']}, w={after['width']}, h={after['height']}`")
            lines.append("")

    # Modified widgets
    if changes["widgets_modified"]:
        lines.append("## Widgets Modified")
        lines.append("")
        for widget in changes["widgets_modified"]:
            lines.append(f"### ✏️ {widget['after_title']}")
            lines.append(f"- Type: `{widget['type']}`")
            lines.append(f"- Before: `{widget['before_title']}`")
            lines.append(f"- After: `{widget['after_title']}`")
            if widget.get("queries"):
                lines.append("- Queries:")
                for query in widget["queries"]:
                    lines.append(f"  - `{query}`")
            lines.append("")

    return "\n".join(lines)


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Analyze changes between two dashboard versions",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("before", help="Path to before dashboard JSON")
    parser.add_argument("after", help="Path to after dashboard JSON")
    parser.add_argument(
        "--section",
        help="Analyze only this section (e.g., 'Push-to-Main')",
    )
    parser.add_argument(
        "--format",
        choices=["text", "markdown", "json"],
        default="text",
        help="Output format (default: text)",
    )

    args = parser.parse_args()

    try:
        before = load_dashboard(args.before)
        after = load_dashboard(args.after)

        changes = compare_dashboards(before, after, section=args.section)

        if args.format == "json":
            print(json.dumps(changes, indent=2))
        elif args.format == "markdown":
            print(format_markdown(changes))
        else:
            print(format_text(changes))

    except FileNotFoundError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)
    except json.JSONDecodeError as e:
        print(f"Error parsing JSON: {e}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
