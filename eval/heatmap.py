"""
Generate heatmap visualizations for policy evaluation metrics.
"""

from typing import Optional, Tuple

import pandas as pd
import plotly.graph_objects as go

def format_metric(metric: str) -> str:
    """Format a metric name for display."""
    return metric.replace("_", " ").capitalize()


def build_wandb_url(policy_uri: str, entity="metta-research", project="metta") -> str:
    """Build a wandb URL from a policy URI."""
    # Strip prefix and version if present
    if policy_uri.startswith("wandb://run/"):
        policy_uri = policy_uri[len("wandb://run/") :]
    if ":v" in policy_uri:
        policy_uri = policy_uri.split(":v")[0]
    return f"https://wandb.ai/{entity}/{project}/runs/{policy_uri}"


def create_matrix_visualization(
    matrix_data: pd.DataFrame,
    metric: str,
    colorscale: str = "RdYlGn",
    score_range: Optional[Tuple[float, float]] = None,
    height: int = 600,
    width: int = 900,
) -> go.Figure:
    """
    Create policy-evaluation matrix visualization.

    Args:
        matrix_data: DataFrame with policies as rows and evaluations as columns
        metric: Name of the metric being visualized
        colorscale: Plotly colorscale to use
        score_range: Optional (min, max) for color scaling
        height: Figure height in pixels
        width: Figure width in pixels

    Returns:
        Plotly figure object
    """
    if matrix_data.empty:
        # Return an empty figure with a message
        fig = go.Figure()
        fig.add_annotation(text="No data available for visualization", showarrow=False, font=dict(size=14))
        fig.update_layout(height=height, width=width)
        return fig

        return fig

    # Get policy names and evaluation names
    policy_uris = (
        matrix_data.pop("policy_uri").tolist() if "policy_uri" in matrix_data.columns else matrix_data.index.tolist()
    )
    eval_names = matrix_data.columns.tolist()

    # Calculate aggregates across policies

    # Convert the matrix to a list for heatmap
    # Convert the matrix to a list for heatmap
    z_values = matrix_data.values.tolist()

    # Calculate aggregates across policies
    mean_values = matrix_data.mean().tolist()
    max_values = matrix_data.max().tolist()
    # Add aggregate rows at the beginning so they appear at the bottom
    # (Plotly heatmaps display first row at the top)
    z_values = [mean_values, max_values] + z_values
    # Add aggregate policy names in the same order as z_values
    policy_uris_with_aggregates = ["Mean", "Max"] + policy_uris

    # Set score range if not provided
    if score_range is None:
        vmin = matrix_data.min().min()
        vmax = matrix_data.max().max()
        # Add a small buffer
        score_range = (vmin * 0.95 if vmin > 0 else vmin * 1.05, vmax * 1.05)

    # Create the heatmap
    fig = go.Figure(
        data=go.Heatmap(
            z=z_values,
            x=eval_names,
            y=policy_uris_with_aggregates,
            colorscale=colorscale,
            zmin=score_range[0],
            zmax=score_range[1],
            colorbar=dict(title="Score"),
            hoverongaps=False,
            hovertemplate="<b>Policy:</b> %{y}<br><b>Evaluation:</b> %{x}<br><b>Score:</b> %{z:.2f}<extra></extra>",
        )
    )

    # Prepare ticktext with clickable links for regular policies and bold text for aggregates
    ticktext = []
    for i, name in enumerate(policy_uris_with_aggregates):
        if i >= 2:  # Skip the first two rows (aggregates)
            # Regular policy - make clickable
            ticktext.append(f'<a href="{build_wandb_url(name)}" target="_blank">{name}</a>')
        else:
            # Aggregate row - make bold
            ticktext.append(f"<b>{name}</b>")

    # Make policy names clickable by converting them to HTML links
    fig.update_layout(
        yaxis=dict(
            tickmode="array",
            tickvals=list(range(len(policy_uris_with_aggregates))),
            ticktext=ticktext,
            tickfont=dict(family="Arial", size=12),
        )
    )

    # Define consistent border styling
    border_color = "rgba(0, 0, 0, 0.5)"  # 50% transparent black (gray)
    border_width = 2

    # Add a box around the aggregates section (Mean/Max rows)
    fig.add_shape(
        type="rect",
        x0=-0.5,  # Left edge
        x1=len(eval_names) - 0.5,  # Right edge
        y0=-0.5,  # Top edge
        y1=1.5,  # Bottom edge (covers 'Mean' and 'Max')
        line=dict(color=border_color, width=border_width, dash="dash"),
        fillcolor="rgba(0,0,0,0)",
        layer="above",
    )

    # Update layout
    formatted_metric = format_metric(metric)
    fig.update_layout(
        title=f"{formatted_metric} Policy-Evaluation Matrix",
        xaxis=dict(
            title="Evaluation",
            tickangle=-45,
            showgrid=False,  # Turn off grid lines
            zeroline=False,
        ),
        yaxis_title="Policy",
        height=height + 50,  # Increase height to accommodate aggregate rows
        width=width,
        margin=dict(l=50, r=50, t=50, b=100),
        plot_bgcolor="white",
        showlegend=False,
    )

    # Special styling for "Overall" column if present
    if "Overall" in eval_names:
        overall_idx = eval_names.index("Overall")

        # Add a box around the "Overall" column
        fig.add_shape(
            type="rect",
            x0=overall_idx - 0.5,
            x1=overall_idx + 0.5,
            y0=-0.5,
            y1=len(policy_uris_with_aggregates) - 0.5,
            line=dict(color=border_color, width=border_width, dash="dash"),
            fillcolor="rgba(0, 0, 0, 0)",
            layer="above",
        )

        # Custom tick labels for x-axis to make Overall bold
        ticktext = []
        for _i, name in enumerate(eval_names):
            if name == "Overall":
                ticktext.append(f"<b>{name}</b>")
            else:
                ticktext.append(name)

        fig.update_layout(
            xaxis=dict(
                title="Evaluation",
                tickangle=-45,
                tickmode="array",
                tickvals=list(range(len(eval_names))),
                ticktext=ticktext,
            )
        )

    return fig


def save_heatmap_to_html(fig: go.Figure, output_path: str, title: str = "Policy Evaluation Heatmap") -> None:
    """
    Save a Plotly figure as a standalone HTML file.

    Args:
        fig: Plotly figure object
        output_path: Path to save the HTML file
        title: HTML page title
    """
    with open(output_path, "w") as f:
        f.write(f"""
        <!DOCTYPE html>
        <html>
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>{title}</title>
            <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
            <style>
                body {{
                    font-family: Arial, sans-serif;
                    margin: 0;
                    padding: 20px;
                    background-color: #f8f9fa;
                }}
                .container {{
                    max-width: 1200px;
                    margin: 0 auto;
                    background-color: white;
                    padding: 20px;
                    border-radius: 5px;
                    box-shadow: 0 2px 4px rgba(0,0,0,0.1);
                }}
                h1 {{
                    color: #333;
                    border-bottom: 1px solid #ddd;
                    padding-bottom: 10px;
                }}
            </style>
        </head>
        <body>
            <div class="container">
                <h1>{title}</h1>
                <div id="heatmap"></div>
            </div>
            <script>
                var figure = {fig.to_json()};
                Plotly.newPlot('heatmap', figure.data, figure.layout);
            </script>
        </body>
        </html>
        """)
