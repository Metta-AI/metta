"""
Marimo-native scorecard visualization using interactive components.
"""

from typing import Any, Optional, Dict, Tuple
import marimo as mo
import pandas as pd
import altair as alt
from metta.app_backend.routes.scorecard_routes import ScorecardData

__all__ = [
    "prepare_scorecard_data",
    "render_scorecard_content",
    "render_comparison_charts",
    "format_policy_name",
    "format_eval_name",
]


def prepare_scorecard_data(
    data: ScorecardData, metric: str, num_policies: int = 20
) -> Tuple[Optional[pd.DataFrame], Optional[Dict]]:
    """Prepare data for scorecard visualization."""
    if not data:
        return None, None

    # Extract data
    cells = data.cells
    eval_names = data.evalNames
    policy_names = data.policyNames
    policy_averages = data.policyAverageScores

    # Sort policies by average score (best first)
    sorted_policies = sorted(
        policy_names, key=lambda p: policy_averages.get(p, 0), reverse=True
    )[:num_policies]

    # Group evaluations by category for better organization
    eval_by_category = {}
    for eval_name in eval_names:
        category = eval_name.split("/")[0] if "/" in eval_name else "misc"
        if category not in eval_by_category:
            eval_by_category[category] = []
        eval_by_category[category].append(eval_name)

    # Prepare data for visualization
    rows = []
    for policy in sorted_policies:
        row = {
            "Policy": format_policy_name(policy),
            "Overall Score": round(policy_averages.get(policy, 0), 1),
            "_original_policy": policy,  # Keep original for linking
        }

        # Add evaluation scores
        for eval_name in eval_names:
            cell_data = cells.get(policy, {}).get(eval_name)
            if cell_data:
                value = cell_data.value if hasattr(cell_data, "value") else cell_data
                if isinstance(value, (int, float)):
                    row[format_eval_name(eval_name)] = round(value, 1)
                else:
                    row[format_eval_name(eval_name)] = value
            else:
                row[format_eval_name(eval_name)] = None

        rows.append(row)

    # Create DataFrame
    df = pd.DataFrame(rows)

    # Return data and context
    return df, {
        "eval_by_category": eval_by_category,
        "cells": cells,
        "metric": metric,
        "sorted_policies": sorted_policies,
        "policy_averages": policy_averages,
        "policy_names": policy_names,
    }


def render_scorecard_content(tabs_value: str, df: pd.DataFrame, context: Dict) -> Any:
    """Render scorecard content based on selected tab."""
    if tabs_value == "table":
        return create_table_view(
            df, context["eval_by_category"], context["cells"], context["metric"]
        )
    elif tabs_value == "heatmap":
        return create_heatmap_view(df, context["metric"])
    else:  # comparison
        return create_comparison_view(df, context["metric"])


def create_table_view(df, eval_by_category, cells, metric):
    """Create an interactive table view."""

    # Create the table without _original_policy column
    display_df = df.drop(columns=["_original_policy"])

    table = mo.ui.table(
        data=display_df, selection="single", pagination=True, page_size=15
    )

    # Return just the table component - details will be handled separately
    return mo.vstack(
        [mo.md("*Click on a row to see detailed scores and replay links*"), table],
        gap=2,
    )


def create_heatmap_view(df, metric):
    """Create an interactive heatmap visualization."""
    # Prepare data for heatmap
    display_df = df.drop(columns=["_original_policy"])
    df_melted = display_df.melt(
        id_vars=["Policy", "Overall Score"], var_name="Evaluation", value_name="Score"
    ).dropna(subset=["Score"])

    # Create the heatmap
    heatmap = (
        alt.Chart(df_melted)
        .mark_rect(stroke="white", strokeWidth=1)
        .encode(
            x=alt.X(
                "Evaluation:N",
                title="Evaluation",
                axis=alt.Axis(labelAngle=-45, labelLimit=200),
            ),
            y=alt.Y(
                "Policy:N",
                title="Policy",
                sort=alt.SortField("Overall Score", order="descending"),
            ),
            color=alt.Color(
                "Score:Q",
                scale=alt.Scale(domain=[0, 100], scheme="redyellowgreen", clamp=True),
                title="Score",
                legend=alt.Legend(orient="right", gradientLength=200),
            ),
            tooltip=[
                alt.Tooltip("Policy:N", title="Policy"),
                alt.Tooltip("Evaluation:N", title="Evaluation"),
                alt.Tooltip("Score:Q", title="Score", format=".1f"),
            ],
        )
        .properties(
            width=800,
            height=max(400, len(df) * 25),  # Dynamic height based on policies
            title={
                "text": f"Policy Performance Heatmap - {metric.upper()}",
                "subtitle": "Hover over cells for details",
                "fontSize": 16,
                "subtitleFontSize": 12,
            },
        )
        .configure_axis(labelFontSize=11, titleFontSize=13)
        .configure_legend(labelFontSize=11, titleFontSize=12)
    )

    return mo.ui.altair_chart(heatmap)


def create_comparison_view(df, metric):
    """Create an interactive comparison chart selector."""
    # Allow selecting policies to compare

    policy_selector = mo.ui.multiselect(
        options=df["Policy"].tolist(),
        value=df["Policy"].head(5).tolist(),  # Default to top 5
        label="Select policies to compare:",
        max_selections=10,
    )

    # Just return the selector - charts will be created in a separate function
    return policy_selector


def render_comparison_charts(policy_selector_value, df, metric):
    """Render comparison charts based on selected policies."""
    if not policy_selector_value:
        return mo.md("*Select policies to compare*")

    display_df = df.drop(columns=["_original_policy"])

    # Filter data for selected policies
    selected_df = display_df[display_df["Policy"].isin(policy_selector_value)]

    # Melt for visualization
    comparison_data = selected_df.melt(
        id_vars=["Policy", "Overall Score"], var_name="Evaluation", value_name="Score"
    ).dropna(subset=["Score"])

    # Create radar chart using custom encoding
    base = alt.Chart(comparison_data).encode(
        theta=alt.Theta("Evaluation:N", stack=None),
        radius=alt.Radius("Score:Q", scale=alt.Scale(domain=[0, 100])),
        color=alt.Color("Policy:N", legend=alt.Legend(orient="top", columns=2)),
        tooltip=["Policy:N", "Evaluation:N", alt.Tooltip("Score:Q", format=".1f")],
    )

    # Combine line and point marks for better visibility
    radar = base.mark_line(opacity=0.6, strokeWidth=2) + base.mark_point(size=100)

    radar_chart = radar.properties(
        width=600, height=600, title=f"Policy Comparison Radar - {metric.upper()}"
    ).configure_axis(grid=True, gridOpacity=0.3, labelFontSize=10)

    # Alternative bar chart view
    bar_chart = (
        alt.Chart(comparison_data)
        .mark_bar()
        .encode(
            x=alt.X("Evaluation:N", axis=alt.Axis(labelAngle=-45)),
            y=alt.Y("Score:Q", scale=alt.Scale(domain=[0, 100])),
            color=alt.Color("Policy:N"),
            xOffset="Policy:N",
            tooltip=["Policy:N", "Evaluation:N", alt.Tooltip("Score:Q", format=".1f")],
        )
        .properties(
            width=700, height=400, title=f"Policy Comparison Bars - {metric.upper()}"
        )
    )

    # Show both visualizations
    return mo.vstack(
        [mo.ui.altair_chart(radar_chart), mo.ui.altair_chart(bar_chart)], gap=3
    )


def format_policy_name(policy: str) -> str:
    """Format policy name for display."""
    if ":v" in policy:
        run_id, version = policy.split(":v")
        # Shorten long run IDs
        if len(run_id) > 40:
            run_id = run_id[:37] + "..."
        return f"{run_id} v{version}"
    return policy


def format_eval_name(eval_name: str) -> str:
    """Format evaluation name for display."""
    if "/" in eval_name:
        _, name = eval_name.split("/", 1)
        return name.replace("_", " ").title()
    return eval_name.replace("_", " ").title()
