"""
Generate heatmap visualizations for policy evaluation metrics.
"""

import plotly.graph_objects as go
import pandas as pd
from typing import Dict, Optional, Tuple, List

def format_metric(metric: str) -> str:
    """Format a metric name for display."""
    return metric.replace('_', ' ').capitalize()

def create_matrix_visualization(
    matrix_data: pd.DataFrame,
    metric: str,
    colorscale: str = 'RdYlGn',
    score_range: Optional[Tuple[float, float]] = None,
    height: int = 600,
    width: int = 900
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
        fig.add_annotation(
            text="No data available for visualization",
            showarrow=False,
            font=dict(size=14)
        )
        fig.update_layout(height=height, width=width)
        return fig
    
    # Get policy names and evaluation names
    policy_names = matrix_data.pop('policy_name').tolist() if 'policy_name' in matrix_data.columns else matrix_data.index.tolist()
    
    # Get evaluation names from attrs or use column names
    if hasattr(matrix_data, 'attrs') and 'eval_names' in matrix_data.attrs:
        eval_names = [matrix_data.attrs['eval_names'].get(col, col) for col in matrix_data.columns]
    else:
        eval_names = matrix_data.columns.tolist()
    
    # Convert the matrix to a list for heatmap
    z_values = matrix_data.values.tolist()
    
    # Set score range if not provided
    if score_range is None:
        vmin = matrix_data.min().min()
        vmax = matrix_data.max().max()
        # Add a small buffer
        score_range = (vmin * 0.95 if vmin > 0 else vmin * 1.05, vmax * 1.05)
    
    # Create the heatmap figure
    fig = go.Figure(data=go.Heatmap(
        z=z_values,
        x=eval_names,
        y=policy_names,
        colorscale=colorscale,
        zmin=score_range[0],
        zmax=score_range[1],
        colorbar=dict(
            title="Score"
        ),
        hoverongaps=False,
        hovertemplate='<b>Policy:</b> %{y}<br><b>Evaluation:</b> %{x}<br><b>Score:</b> %{z:.2f}<extra></extra>'
    ))
    
    # Update layout
    formatted_metric = format_metric(metric)
    fig.update_layout(
        title=f"{formatted_metric} Policy-Evaluation Matrix",
        xaxis=dict(
            title="Evaluation",
            tickangle=-45
        ),
        yaxis=dict(
            title="Policy"
        ),
        height=height,
        width=width,
        margin=dict(l=50, r=50, t=50, b=100),
    )
    
    # Special styling for "Overall" column if present
    if "Overall" in eval_names:
        overall_idx = eval_names.index("Overall")
        fig.add_shape(
            type="rect",
            x0=overall_idx - 0.5,
            x1=overall_idx + 0.5,
            y0=-0.5,
            y1=len(policy_names) - 0.5,
            line=dict(width=2),
            fillcolor="rgba(0, 0, 0, 0)",
            line_color="rgba(0, 0, 0, 0.5)"
        )
    
    return fig

def save_heatmap_to_html(
    fig: go.Figure,
    output_path: str,
    title: str = "Policy Evaluation Heatmap"
) -> None:
    """
    Save a Plotly figure as a standalone HTML file.
    
    Args:
        fig: Plotly figure object
        output_path: Path to save the HTML file
        title: HTML page title
    """
    with open(output_path, 'w') as f:
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