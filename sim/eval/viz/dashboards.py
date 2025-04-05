
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
            html.H5("Matrix Visualization"),
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
            html.H5(f"Matrix Visualization: {format_metric_name(metric_name)}"),
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
            title="Score",
            titleside="right"
        ),
        hovertemplate='<b>Policy:</b> %{y}<br><b>Evaluation:</b> %{x}<br><b>Score:</b> %{z:.2f}<extra></extra>'
    ))
    
    # Update layout
    formatted_metric = format_metric_name(metric_name)
    fig.update_layout(
        title=f"Policy-Evaluation Matrix: {formatted_metric}",
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
        html.H5(f"Matrix Visualization: {formatted_metric}", className="visualization-title"),
        dcc.Graph(
            id=viz_id,
            figure=fig,
            config={'displayModeBar': True}
        )
    ], id=f"{viz_id}-container", className="visualization-card mb-4")

def create_pass_rate_chart(
    pass_rates: List[Dict[str, Any]], 
    config: Dict[str, Any],
    filtered_policies: Optional[List[str]] = None
) -> html.Div:
    """
    Create pass rate bar chart component.
    
    Args:
        pass_rates: List of dictionaries containing pass rate data
        config: Application configuration
        filtered_policies: Optional list of policies to include
        
    Returns:
        Dash layout component for pass rate visualization
    """
    # Handle empty data case
    if not pass_rates:
        return html.Div([
            html.H5("Pass Rate Visualization"),
            html.P("No data available for visualization.")
        ], className="visualization-card mb-4")
    
    # Apply policy filter if provided
    if filtered_policies:
        pass_rates = [pr for pr in pass_rates if pr['policy_id'] in filtered_policies]
    
    # Handle empty filtered result
    if not pass_rates:
        return html.Div([
            html.H5("Pass Rate Visualization"),
            html.P("No data available for selected filters.")
        ], className="visualization-card mb-4")
    
    # Sort by pass rate (descending)
    pass_rates = sorted(pass_rates, key=lambda x: x.get('pass_rate', 0), reverse=True)
    
    # Create dataframe for plotting
    df = pd.DataFrame(pass_rates)
    
    # Create the figure
    fig = go.Figure(data=[
        go.Bar(
            x=df['display_name'],
            y=df['pass_rate'],
            text=[f"{pr.get('passed', 0)}/{pr.get('total', 0)}" for pr in pass_rates],
            textposition='auto',
            hovertemplate='<b>%{x}</b><br>Pass Rate: %{y:.1f}%<br>Passing: %{text}<extra></extra>'
        )
    ])
    
    # Update layout
    threshold = config.get('pass_threshold', 2.95)
    fig.update_layout(
        title=f"Policy Pass Rate (>= {threshold})",
        xaxis=dict(
            title="Policy",
            tickangle=-45
        ),
        yaxis=dict(
            title="Pass Rate (%)",
            range=[0, 100]
        ),
        height=config.get('default_graph_height', 500),
        width=config.get('default_graph_width', 800),
        margin=dict(l=50, r=50, t=50, b=100),
    )
    
    # Return visualization component
    return html.Div([
        html.H5("Policy Pass Rate", className="visualization-title"),
        dcc.Graph(
            id="pass-rate-chart",
            figure=fig,
            config={'displayModeBar': True}
        )
    ], id="pass-rate-container", className="visualization-card mb-4")

def create_policy_ranking_chart(
    policy_ranking: List[Dict[str, Any]], 
    config: Dict[str, Any],
    filtered_policies: Optional[List[str]] = None
) -> html.Div:
    """
    Create policy ranking bar chart component.
    
    Args:
        policy_ranking: List of dictionaries containing policy ranking data
        config: Application configuration
        filtered_policies: Optional list of policies to include
        
    Returns:
        Dash layout component for policy ranking visualization
    """
    # Handle empty data case
    if not policy_ranking:
        return html.Div([
            html.H5("Policy Leaderboard"),
            html.P("No data available for visualization.")
        ], className="visualization-card mb-4")
    
    # Apply policy filter if provided
    if filtered_policies:
        policy_ranking = [pr for pr in policy_ranking if pr.get('policy_id') in filtered_policies]
    
    # Handle empty filtered result
    if not policy_ranking:
        return html.Div([
            html.H5("Policy Leaderboard"),
            html.P("No data available for selected filters.")
        ], className="visualization-card mb-4")
    
    # Sort by wins (descending)
    policy_ranking = sorted(policy_ranking, key=lambda x: x.get('wins', 0), reverse=True)
    
    # Create dataframe for plotting
    df = pd.DataFrame(policy_ranking)
    
    # Create the figure
    fig = go.Figure(data=[
        go.Bar(
            x=df['display_name'],
            y=df['wins'],
            hovertemplate='<b>%{x}</b><br>Wins: %{y}<extra></extra>'
        )
    ])
    
    # Update layout
    fig.update_layout(
        title="Policy Leaderboard (# of eval wins)",
        xaxis=dict(
            title="Policy",
            tickangle=-45
        ),
        yaxis=dict(
            title="Wins"
        ),
        height=config.get('default_graph_height', 500),
        width=config.get('default_graph_width', 800),
        margin=dict(l=50, r=50, t=50, b=100),
    )
    
    # Return visualization component
    return html.Div([
        html.H5("Policy Leaderboard", className="visualization-title"),
        dcc.Graph(
            id="policy-ranking-chart",
            figure=fig,
            config={'displayModeBar': True}
        )
    ], id="policy-ranking-container", className="visualization-card mb-4")

def create_highest_scores_chart(
    highest_scores: List[Dict[str, Any]], 
    config: Dict[str, Any],
    filtered_evals: Optional[List[str]] = None
) -> html.Div:
    """
    Create highest scores bar chart component.
    
    Args:
        highest_scores: List of dictionaries containing highest score data
        config: Application configuration
        filtered_evals: Optional list of evaluations to include
        
    Returns:
        Dash layout component for highest scores visualization
    """
    # Handle empty data case
    if not highest_scores:
        return html.Div([
            html.H5("Highest Scores by Evaluation"),
            html.P("No data available for visualization.")
        ], className="visualization-card mb-4")
    
    # Apply evaluation filter if provided
    if filtered_evals:
        highest_scores = [hs for hs in highest_scores if hs.get('eval_id') in filtered_evals]
    
    # Handle empty filtered result
    if not highest_scores:
        return html.Div([
            html.H5("Highest Scores by Evaluation"),
            html.P("No data available for selected filters.")
        ], className="visualization-card mb-4")
    
    # Sort by evaluation name
    highest_scores = sorted(highest_scores, key=lambda x: x.get('display_eval', ''))
    
    # Create dataframe for plotting
    df = pd.DataFrame(highest_scores)
    
    # Create the figure
    fig = go.Figure(data=[
        go.Bar(
            x=df['display_eval'],
            y=df['score'],
            text=df['display_policy'],
            textposition='auto',
            hovertemplate='<b>Eval: %{x}</b><br>Score: %{y:.2f}<br>Policy: %{text}<extra></extra>'
        )
    ])
    
    # Update layout
    max_score = max([hs.get('score', 0) for hs in highest_scores], default=3)
    fig.update_layout(
        title="Highest Scores by Evaluation",
        xaxis=dict(
            title="Evaluation",
            tickangle=-45
        ),
        yaxis=dict(
            title="Score",
            range=[0, max_score * 1.1]  # Add 10% margin
        ),
        height=config.get('default_graph_height', 500),
        width=config.get('default_graph_width', 800),
        margin=dict(l=50, r=50, t=50, b=100),
    )
    
    # Return visualization component
    return html.Div([
        html.H5("Highest Scores by Evaluation", className="visualization-title"),
        dcc.Graph(
            id="highest-scores-chart",
            figure=fig,
            config={'displayModeBar': True}
        )
    ], id="highest-scores-container", className="visualization-card mb-4")

def create_metric_comparison_chart(
    metrics_data: Dict[str, pd.DataFrame],
    metric_names: List[str],
    config: Dict[str, Any],
    filtered_policies: Optional[List[str]] = None
) -> html.Div:
    """
    Create a comparison chart of multiple metrics for selected policies.
    
    Args:
        metrics_data: Dictionary of metric names to dataframes
        metric_names: List of metrics to compare
        config: Application configuration
        filtered_policies: Optional list of policies to include
        
    Returns:
        Dash layout component for metric comparison visualization
    """
    # Validate input data
    if not metrics_data or not metric_names:
        return html.Div([
            html.H5("Metric Comparison"),
            html.P("No metrics available for comparison.")
        ], className="visualization-card mb-4")
    
    # Filter available metrics
    available_metrics = [m for m in metric_names if m in metrics_data]
    
    if not available_metrics:
        return html.Div([
            html.H5("Metric Comparison"),
            html.P("No data available for selected metrics.")
        ], className="visualization-card mb-4")
    
    # Create comparison data structure
    comparison_data = []
    
    for metric_name in available_metrics:
        df = metrics_data[metric_name]
        
        # Apply policy filter if provided
        if filtered_policies:
            df = df[df['policy_name'].isin(filtered_policies)]
        
        if df.empty:
            continue
            
        # Get mean column
        mean_col = next((col for col in df.columns if col.startswith('mean_')), None)
        if not mean_col:
            continue
            
        # Aggregate by policy
        policy_means = df.groupby('policy_name')[mean_col].mean().reset_index()
        
        for _, row in policy_means.iterrows():
            comparison_data.append({
                'policy_name': row['policy_name'],
                'metric': format_metric_name(metric_name),
                'value': row[mean_col]
            })
    
    # Handle empty result
    if not comparison_data:
        return html.Div([
            html.H5("Metric Comparison"),
            html.P("No data available for selected policies and metrics.")
        ], className="visualization-card mb-4")
    
    # Create dataframe for plotting
    comparison_df = pd.DataFrame(comparison_data)
    
    # Create the figure
    fig = px.bar(
        comparison_df, 
        x='policy_name', 
        y='value', 
        color='metric',
        barmode='group',
        labels={
            'policy_name': 'Policy',
            'value': 'Score',
            'metric': 'Metric'
        },
        title="Metric Comparison Across Policies"
    )
    
    # Update layout
    fig.update_layout(
        xaxis=dict(tickangle=-45),
        height=config.get('default_graph_height', 500),
        width=config.get('default_graph_width', 800),
        margin=dict(l=50, r=50, t=50, b=100),
    )
    
    # Return visualization component
    return html.Div([
        html.H5("Metric Comparison", className="visualization-title"),
        dcc.Graph(
            id="metric-comparison-chart",
            figure=fig,
            config={'displayModeBar': True}
        )
    ], id="metric-comparison-container", className="visualization-card mb-4")