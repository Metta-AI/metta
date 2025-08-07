from typing import Any

from metta.app_backend.routes.scorecard_routes import ScorecardData

import marimo as mo


def create_marimo_scorecard(
    data: ScorecardData, metric: str, num_policies: int = 20
) -> Any:
    """Create a marimo-native scorecard visualization."""
    if not data:
        return mo.Html(
            """
            <div style="padding: 40px; text-align: center; color: #6c757d;">
                <svg width="64" height="64" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" style="margin: 0 auto 16px;">
                    <path d="M9 11l3 3 8-8M21 12a9 9 0 11-18 0 9 9 0 0118 0z"/>
                </svg>
                <p style="font-size: 18px; margin: 0;">No scorecard data available</p>
                <p style="font-size: 14px; color: #adb5bd; margin-top: 8px;">Select policies and evaluations to generate a scorecard</p>
            </div>
            """
        )

    # ScorecardData has these attributes: cells, evalNames, policyNames, policyAverageScores
    cells = data.cells
    eval_names = data.evalNames
    policy_names = data.policyNames
    policy_averages = data.policyAverageScores

    # Sort policies by average score (worst to best, for display)
    sorted_policies = sorted(policy_names, key=lambda p: policy_averages.get(p, 0))[
        -num_policies:
    ]

    # Group evaluations by category
    eval_by_category = {}
    for eval_name in eval_names:
        if "/" in eval_name:
            category, name = eval_name.split("/", 1)
        else:
            category = "misc"
            name = eval_name
        if category not in eval_by_category:
            eval_by_category[category] = []
        eval_by_category[category].append((name, eval_name))

    # Build HTML table
    html = f"""
    <style>
        .scorecard-container {{
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif;
            margin: 0;
            background: #ffffff;
            border-radius: 12px;
            box-shadow: 0 1px 3px rgba(0, 0, 0, 0.05), 0 1px 2px rgba(0, 0, 0, 0.1);
            overflow: hidden;
        }}
        .scorecard-header {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 24px;
            display: flex;
            align-items: center;
            justify-content: space-between;
        }}
        .scorecard-title {{
            font-size: 24px;
            font-weight: 600;
            margin: 0;
        }}
        .scorecard-subtitle {{
            font-size: 14px;
            opacity: 0.9;
            margin-top: 4px;
        }}
        .scorecard-metric-badge {{
            background: rgba(255, 255, 255, 0.2);
            padding: 6px 16px;
            border-radius: 20px;
            font-size: 14px;
            font-weight: 500;
            backdrop-filter: blur(10px);
        }}
        .scorecard-wrapper {{
            overflow-x: auto;
            background: #f8f9fa;
        }}
        .scorecard-table {{
            border-collapse: separate;
            border-spacing: 0;
            font-size: 13px;
            width: 100%;
            min-width: 800px;
            background: white;
        }}
        .scorecard-table th {{
            padding: 12px 8px;
            text-align: center;
            background-color: #ffffff;
            border-bottom: 2px solid #e9ecef;
            font-weight: 600;
            font-size: 12px;
            color: #495057;
            position: sticky;
            top: 0;
            z-index: 10;
            white-space: nowrap;
        }}
        .scorecard-table th:first-child {{
            text-align: left;
            padding-left: 24px;
            font-size: 13px;
        }}
        .scorecard-table td {{
            padding: 10px 8px;
            text-align: center;
            border-bottom: 1px solid #f1f3f5;
            transition: all 0.15s ease;
            position: relative;
        }}
        .scorecard-table td:first-child {{
            text-align: left;
            font-weight: 500;
            background-color: #fcfcfc;
            position: sticky;
            left: 0;
            z-index: 5;
            padding-left: 24px;
            border-right: 1px solid #e9ecef;
            color: #212529;
            font-size: 13px;
            max-width: 300px;
            overflow: hidden;
            text-overflow: ellipsis;
            white-space: nowrap;
        }}
        .scorecard-table tr {{
            transition: all 0.15s ease;
        }}
        .scorecard-table tbody tr:hover {{
            background-color: #f8f9fa;
            transform: translateY(-1px);
            box-shadow: 0 2px 4px rgba(0, 0, 0, 0.04);
        }}
        .scorecard-table tbody tr:hover td:first-child {{
            background-color: #f1f3f5;
        }}
        .scorecard-cell {{
            position: relative;
            min-width: 70px;
            font-weight: 500;
            border-radius: 6px;
            margin: 0 2px;
        }}
        .category-header {{
            background: linear-gradient(to bottom, #f8f9fa, #e9ecef) !important;
            font-size: 11px;
            text-transform: uppercase;
            letter-spacing: 0.5px;
            color: #6c757d;
            border-bottom: 2px solid #dee2e6 !important;
        }}
        .policy-link {{
            color: #4c6ef5;
            text-decoration: none;
            font-weight: 500;
            transition: color 0.15s ease;
        }}
        .policy-link:hover {{
            color: #364fc7;
            text-decoration: underline;
        }}
        .score-cell {{
            font-weight: 600;
            color: #212529;
            border-radius: 6px;
            position: relative;
            padding: 4px 8px;
            display: inline-block;
            min-width: 48px;
        }}
        .overall-score {{
            font-size: 14px;
            font-weight: 700;
        }}
        .empty-cell {{
            color: #adb5bd;
            font-size: 16px;
        }}
        .scorecard-legend {{
            padding: 16px 24px;
            background: #f8f9fa;
            border-top: 1px solid #e9ecef;
            display: flex;
            align-items: center;
            gap: 24px;
            font-size: 12px;
            color: #6c757d;
        }}
        .legend-item {{
            display: flex;
            align-items: center;
            gap: 8px;
        }}
        .legend-color {{
            width: 16px;
            height: 16px;
            border-radius: 4px;
            border: 1px solid rgba(0, 0, 0, 0.1);
        }}
    </style>
    <div class="scorecard-container">
        <div class="scorecard-header">
            <div>
                <h2 class="scorecard-title">Policy Performance Scorecard</h2>
                <div class="scorecard-subtitle">Showing top {len(sorted_policies)} policies by performance</div>
            </div>
            <div class="scorecard-metric-badge">
                Metric: {metric.upper()}
            </div>
        </div>
        <div class="scorecard-wrapper">
            <table class="scorecard-table">
                <thead>
                    <tr>
                        <th>Policy</th>
                        <th style="background: #f0f8ff; border-bottom-color: #4c6ef5;">Overall</th>
    """

    # Add category headers
    for category in sorted(eval_by_category.keys()):
        colspan = len(eval_by_category[category])
        category_display = category.replace("_", " ").title()
        html += (
            f'<th colspan="{colspan}" class="category-header">{category_display}</th>'
        )
    html += "</tr><tr><th></th><th style='background: #f0f8ff;'></th>"

    # Add evaluation name headers
    for category in sorted(eval_by_category.keys()):
        for name, _ in eval_by_category[category]:
            display_name = name.replace("_", " ").title()
            html += f'<th title="{name}">{display_name}</th>'
    html += "</tr></thead><tbody>"

    # Add data rows
    for i, policy in enumerate(reversed(sorted_policies)):  # Best policies at top
        policy_avg = policy_averages.get(policy, 0)

        # Create policy link if it contains version info
        if ":v" in policy:
            run_id = policy.split(":v")[0]
            version = policy.split(":v")[1]
            policy_display = f'<a href="https://wandb.ai/softmax-ai/metta/runs/{run_id}" target="_blank" class="policy-link" title="View in W&B">{run_id} <span style="color: #868e96; font-weight: normal;">v{version}</span></a>'
        else:
            policy_display = f'<span title="{policy}">{policy}</span>'

        row_style = 'style="background: #fafbfc;"' if i == 0 else ""
        html += f"<tr {row_style}><td>{policy_display}</td>"

        # Overall score with color
        color = get_score_color(policy_avg, 0, 100)
        html += f'<td><span class="score-cell overall-score" style="background-color: {color};">{policy_avg:.1f}</span></td>'

        # Individual evaluation scores
        for category in sorted(eval_by_category.keys()):
            for _, eval_name in eval_by_category[category]:
                # In ScorecardData, cells[policy][eval_name] contains a ScorecardCell object
                cell_data = cells.get(policy, {}).get(eval_name)
                if cell_data:
                    # For ScorecardCell objects from the API
                    if hasattr(cell_data, "value"):
                        value = cell_data.value  # This is already a float
                    # For dict representation (when parsed from JSON)
                    elif isinstance(cell_data, dict) and "value" in cell_data:
                        value = cell_data["value"]
                    else:
                        # Fallback: try to convert directly
                        try:
                            value = float(cell_data)
                        except (TypeError, ValueError):
                            value = 0

                    color = get_score_color(value, 0, 100)
                    html += f'<td><span class="score-cell" style="background-color: {color};" title="{eval_name}: {value:.2f}">{value:.1f}</span></td>'
                else:
                    html += '<td><span class="empty-cell">-</span></td>'

        html += "</tr>"

    return mo.Html(html)


def get_score_color(value: float, min_val: float = 0, max_val: float = 100) -> str:
    """Get color for score visualization using a diverging color scale."""
    if value is None:
        return "#f8f9fa"

    # Normalize to 0-1
    norm_value = (value - min_val) / (max_val - min_val)
    norm_value = max(0, min(1, norm_value))

    # Use a diverging color scale (red -> yellow -> green)
    # Based on RdYlGn color scheme
    if norm_value < 0.33:
        # Poor performance (red tones)
        return "rgba(215, 48, 39, 0.2)"  # #d73027
    elif norm_value < 0.67:
        # Fair performance (yellow tones)
        return "rgba(254, 224, 139, 0.3)"  # #fee08b
    else:
        # Good performance (green tones)
        return "rgba(26, 152, 80, 0.2)"  # #1a9850"
