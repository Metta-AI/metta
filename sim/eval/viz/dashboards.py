
import plotly.graph_objects as go
import plotly.express as px
from dash import html, dcc
import dash_bootstrap_components as dbc
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Tuple, Optional, Union

def format_metric_name(metric_name: str) -> str:
    return metric_name.replace('_', ' ').capitalize()

def create_matrix_visualization(
    matrix_data: Dict[str, Any], 
    config: Dict[str, Any],
    metric_name: str,
    filtered_policies: Optional[List[str]] = None,
    filtered_evals: Optional[List[str]] = None
) -> html.Div:
    """
    Create policy-evaluation matrix visualization component.
    
    Args:
        matrix_data: Dictionary containing matrix data
        config: Application configuration
        metric_name: Name of the metric being visualized
        filtered_policies: Optional list of policies to include
        filtered_evals: Optional list of evaluations to include
        
    Returns:
        Dash layout component for matrix visualization
    """
    # Handle empty data case
    if not matrix_data or not matrix_data.get('z') or not matrix_data.get('x') or not matrix_data.get('y'):
        return html.Div([
            html.P("No data available for visualization.")
        ], className="visualization-card mb-4")
        
    # Apply filters if provided
    if filtered_policies or filtered_evals:
        # Create a copy to avoid modifying the original data
        filtered_data = {
            'z': [],
            'x': [],
            'y': [],
            'policy_ids': [],
            'eval_ids': []
        }
        
        # Get indices of policies and evals to keep
        policy_indices = []
        if filtered_policies:
            policy_indices = [i for i, pid in enumerate(matrix_data['policy_ids']) 
                             if pid in filtered_policies]
        else:
            policy_indices = list(range(len(matrix_data['policy_ids'])))
            
        eval_indices = []
        if filtered_evals:
            eval_indices = [i for i, eid in enumerate(matrix_data['eval_ids']) 
                           if eid in filtered_evals]
        else:
            eval_indices = list(range(len(matrix_data['eval_ids'])))
            
        # Filter the data
        filtered_data['y'] = [matrix_data['y'][i] for i in policy_indices]
        filtered_data['x'] = [matrix_data['x'][i] for i in eval_indices]
        filtered_data['policy_ids'] = [matrix_data['policy_ids'][i] for i in policy_indices]
        filtered_data['eval_ids'] = [matrix_data['eval_ids'][i] for i in eval_indices]
        
        # Filter the z values (2D matrix)
        z_array = np.array(matrix_data['z'])
        if policy_indices and eval_indices:
            filtered_z = z_array[np.ix_(policy_indices, eval_indices)]
            filtered_data['z'] = filtered_z.tolist()
        else:
            filtered_data['z'] = []
        
        # Use filtered data for visualization
        matrix_data = filtered_data
        
    # Handle empty filtered result
    if not matrix_data.get('z') or not matrix_data.get('x') or not matrix_data.get('y') or \
       len(matrix_data['z']) == 0 or len(matrix_data['x']) == 0 or len(matrix_data['y']) == 0:
        return html.Div([
            html.P("No data available for selected filters.")
        ], className="visualization-card mb-4")
    
    # Create the heatmap figure
    fig = go.Figure(data=go.Heatmap(
        z=matrix_data['z'],
        x=matrix_data['x'],
        y=matrix_data['y'],
        colorscale=config.get('matrix_colorscale', 'RdYlGn'),
        zmin=config.get('matrix_score_range', (0, 3))[0],
        zmax=config.get('matrix_score_range', (0, 3))[1],
        hoverongaps=False,
        colorbar=dict(
            title="Score"
        ),
        hovertemplate='<b>Policy:</b> %{y}<br><b>Evaluation:</b> %{x}<br><b>Score:</b> %{z:.2f}<extra></extra>'
    ))
    
    # Update layout
    formatted_metric = format_metric_name(metric_name)
    fig.update_layout(
        title=f"{formatted_metric} Policy-Evaluation Matrix",
        xaxis=dict(
            title="Evaluation",
            tickangle=-45
        ),
        yaxis=dict(
            title="Policy"
        ),
        height=config.get('default_graph_height', 500),
        width=config.get('default_graph_width', 800),
        margin=dict(l=50, r=50, t=50, b=100),
    )
    
    # Return visualization component
    viz_id = f"matrix-{metric_name.replace('.', '-')}"
    return html.Div([
        dcc.Graph(
            id=viz_id,
            figure=fig,
            config={'displayModeBar': True}
        )
    ], id=f"{viz_id}-container", className="visualization-card mb-4")
