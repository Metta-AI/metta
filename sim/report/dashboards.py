"""
Matrix visualization of policy-evaluation performance.
"""

import dash
from dash import html, dcc
import dash_bootstrap_components as dbc
from dash.dependencies import Input, Output, State
import plotly.graph_objects as go
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Tuple

def create_matrix_visualization(matrix_data: Dict[str, Any], config: Dict[str, Any]) -> html.Div:
    """
    Create policy-evaluation matrix visualization component.
    
    Args:
        matrix_data: Dictionary containing matrix data
        config: Application configuration
        
    Returns:
        Dash layout component for matrix visualization
    """
    # Create the heatmap figure
    fig = go.Figure(data=go.Heatmap(
        z=matrix_data['z'],
        x=matrix_data['x'],
        y=matrix_data['y'],
        colorscale=config['matrix_colorscale'],
        zmin=config['matrix_score_range'][0],
        zmax=config['matrix_score_range'][1],
        hoverongaps=False,
        colorbar=dict(
            title="Score",
            titleside="right"
        ),
        hovertemplate='<b>Policy:</b> %{y}<br><b>Evaluation:</b> %{x}<br><b>Score:</b> %{z:.2f}<extra></extra>'
    ))
    
    # Update layout
    fig.update_layout(
        title="Policy-Evaluation Matrix",
        xaxis=dict(
            title="Evaluation",
            tickangle=-45
        ),
        yaxis=dict(
            title="Policy"
        ),
        height=config['default_graph_height'],
        width=config['default_graph_width'],
        margin=dict(l=50, r=50, t=50, b=100),
    )
    
    return