#!/usr/bin/env python
# coding: utf-8

"""
Interactive Sweep Analysis Dashboard - Plotly Dash
This creates a professional, interactive dashboard for sweep analysis using Plotly Dash.
The dashboard provides WandB-quality visualizations with full interactivity and detailed hover information.
"""

# Import required libraries
import argparse
import os
import sys
import warnings

# Add the metta module to path before importing from it
sys.path.append(os.path.abspath("../.."))

import subprocess
import threading

import dash
import dash_bootstrap_components as dbc
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from dash import ALL, Input, Output, State, callback_context, dcc, html
from metta.sweep.wandb_utils import deep_clean, get_sweep_runs

warnings.filterwarnings("ignore")


# Parse command line arguments
def parse_args():
    parser = argparse.ArgumentParser(description="Interactive Sweep Analysis Dashboard")
    parser.add_argument(
        "--sweep-name", required=True, help="Name of the sweep to analyze"
    )
    parser.add_argument(
        "--entity",
        default="metta-research",
        help="WandB entity (default: metta-research)",
    )
    parser.add_argument(
        "--project", default="metta", help="WandB project (default: metta)"
    )
    parser.add_argument(
        "--max-observations",
        type=int,
        default=1000,
        help="Maximum observations to load (default: 1000)",
    )
    parser.add_argument(
        "--hourly-cost",
        type=float,
        default=4.6,
        help="Dollar cost per hour for instance (default: 4.6)",
    )
    return parser.parse_args()


# Get configuration from command line arguments
args = parse_args()

# Configuration
WANDB_ENTITY = args.entity
WANDB_PROJECT = args.project
WANDB_SWEEP_NAME = args.sweep_name
MAX_OBSERVATIONS = args.max_observations
HOURLY_COST = args.hourly_cost


# Helper functions from original notebook
def flatten_nested_dict(d, parent_key="", sep="."):
    """Recursively flatten a nested dictionary structure."""
    items = []

    if not isinstance(d, dict):
        return {parent_key: d} if parent_key else {}

    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.extend(flatten_nested_dict(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))

    return dict(items)


def extract_observations_to_dataframe(observations):
    """Convert protein observations to a pandas DataFrame."""
    all_rows = []

    for obs in observations:
        if obs.get("is_failure", False):
            continue

        row_data = {}

        # Get suggestion
        suggestion = obs.get("suggestion", {})

        # Flatten the suggestion dictionary
        if isinstance(suggestion, dict) and suggestion:
            flattened = flatten_nested_dict(suggestion)
            row_data.update(flattened)

        # Add metrics
        row_data["score"] = obs.get("objective", np.nan)
        row_data["cost"] = obs.get("cost", np.nan)
        row_data["runtime"] = obs.get("cost", np.nan)
        row_data["timestamp"] = obs.get("timestamp", obs.get("created_at", np.nan))
        row_data["run_name"] = obs.get("run_name", "")
        row_data["run_id"] = obs.get("run_id", "")

        all_rows.append(row_data)

    df = pd.DataFrame(all_rows)

    # Convert timestamp to datetime
    if "timestamp" in df.columns and not df["timestamp"].isna().all():
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        df = df.sort_values("timestamp").reset_index(drop=True)

    # Add dollar cost
    if "runtime" in df.columns:
        df["dollar_cost"] = (df["runtime"] / 3600.0) * HOURLY_COST

    return df


# Load data
print("Loading sweep data from WandB...")
print(f"Entity: {WANDB_ENTITY}")
print(f"Project: {WANDB_PROJECT}")
print(f"Sweep Name: {WANDB_SWEEP_NAME}")

runs = get_sweep_runs(
    sweep_name=WANDB_SWEEP_NAME, entity=WANDB_ENTITY, project=WANDB_PROJECT
)

print(f"\nLoaded {len(runs)} runs")

# Extract observations
observations = []
for run in runs[:MAX_OBSERVATIONS]:
    protein_obs = run.summary.get("protein_observation")
    protein_suggestion = run.summary.get("protein_suggestion")

    if protein_obs:
        obs = deep_clean(protein_obs)
        if "suggestion" not in obs and protein_suggestion:
            obs["suggestion"] = deep_clean(protein_suggestion)
        obs["timestamp"] = run.created_at
        obs["run_name"] = run.name
        obs["run_id"] = run.id
        observations.append(obs)
    elif protein_suggestion:
        obs = {
            "suggestion": deep_clean(protein_suggestion),
            "objective": run.summary.get(
                "score", run.summary.get("protein.objective", np.nan)
            ),
            "cost": run.summary.get("_wandb", {}).get("runtime", 0),
            "is_failure": run.state != "finished",
            "timestamp": run.created_at,
            "run_name": run.name,
            "run_id": run.id,
        }
        observations.append(obs)

# Convert to DataFrame
df = extract_observations_to_dataframe(observations)
print(f"Valid observations: {len(df)}")

# Handle empty DataFrame case
if df.empty:
    print("Warning: No valid observations found. Creating empty dashboard.")
    # Create a minimal DataFrame with expected columns to prevent crashes
    df = pd.DataFrame(
        {
            "score": [0],
            "cost": [0],
            "runtime": [0],
            "dollar_cost": [0],
            "timestamp": [pd.NaT],
            "run_name": [""],
            "run_id": [""],
        }
    )
    param_cols = []
else:
    # Get parameter columns
    param_cols = [
        col
        for col in df.columns
        if col
        not in [
            "score",
            "cost",
            "runtime",
            "timestamp",
            "run_name",
            "run_id",
            "dollar_cost",
        ]
    ]
    print(f"Found {len(param_cols)} hyperparameters")

# Create Plotly Dash App with Bootstrap theme
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])

# Define color scheme for professional look
COLORS = {
    "background": "#f8f9fa",
    "card_bg": "#ffffff",
    "primary": "#007bff",
    "success": "#28a745",
    "danger": "#dc3545",
    "warning": "#ffc107",
    "info": "#17a2b8",
    "dark": "#343a40",
    "text": "#212529",
    "text_muted": "#6c757d",
}


# Create interactive plots
def create_cost_vs_score_plot(df, selected_points=None):
    """Create interactive cost vs score scatter plot with hover data."""

    # Create hover text with all run information
    hover_text = []
    for idx, row in df.iterrows():
        text = f"<b>Run: {row['run_name']}</b><br>"
        text += f"Score: {row['score']:.4f}<br>"
        text += f"Cost: ${row['dollar_cost']:.4f}<br>"
        text += f"Runtime: {row['runtime']:.1f}s<br>"
        if "trainer.total_timesteps" in row:
            text += f"Timesteps: {row['trainer.total_timesteps'] / 1e6:.1f}M<br>"
        hover_text.append(text)

    fig = go.Figure()

    # Add main scatter plot
    fig.add_trace(
        go.Scatter(
            x=df["dollar_cost"],
            y=df["score"],
            mode="markers",
            marker=dict(
                size=10,
                color=df["score"],
                colorscale="Viridis",
                showscale=True,
                colorbar=dict(
                    title="Score",
                    x=1.02,  # Move colorbar slightly to the right
                    xanchor="left",
                    y=0.5,
                    yanchor="middle",
                ),
                line=dict(width=1, color="white"),
            ),
            text=hover_text,
            hovertemplate="%{text}<extra></extra>",
            customdata=df[["run_id", "run_name"]],
            selectedpoints=selected_points,
        )
    )

    # Highlight best point
    best_idx = df["score"].idxmax()
    fig.add_trace(
        go.Scatter(
            x=[df.loc[best_idx, "dollar_cost"]],
            y=[df.loc[best_idx, "score"]],
            mode="markers",
            marker=dict(
                size=20, color="red", symbol="star", line=dict(width=2, color="darkred")
            ),
            name="Best Score",
            hovertemplate=f"<b>BEST RUN</b><br>Score: {df.loc[best_idx, 'score']:.4f}<br>Cost: ${df.loc[best_idx, 'dollar_cost']:.4f}<extra></extra>",
        )
    )

    fig.update_layout(
        title="Cost vs Score Analysis",
        xaxis_title="Cost ($)",
        yaxis_title="Score",
        hovermode="closest",
        showlegend=True,
        template="plotly_white",
        height=550,
        margin=dict(t=80, b=60, l=60, r=120),  # Extra right margin for colorbar
        legend=dict(
            x=0.02,
            y=0.98,
            xanchor="left",
            yanchor="top",
            bgcolor="rgba(255, 255, 255, 0.8)",
            bordercolor="rgba(0, 0, 0, 0.2)",
            borderwidth=1,
        ),
    )

    return fig


def create_timeline_plot(df):
    """Create score progression over time."""

    hover_text = []
    for idx, row in df.iterrows():
        text = f"<b>Run: {row.get('run_name', 'N/A')}</b><br>"
        text += f"Score: {row['score']:.4f}<br>"
        text += f"Time: {row.get('timestamp', 'N/A')}<br>"
        text += f"Runtime: {row.get('runtime', 0):.1f}s"
        hover_text.append(text)

    fig = go.Figure()

    # Add scatter plot
    if "timestamp" in df.columns and not df["timestamp"].isna().all():
        fig.add_trace(
            go.Scatter(
                x=df["timestamp"],
                y=df["score"],
                mode="markers+lines",
                marker=dict(
                    size=8,
                    color=list(range(len(df))),
                    colorscale="Plasma",
                    showscale=True,
                    colorbar=dict(
                        title="Run Order",
                        x=1.02,
                        xanchor="left",
                        y=0.5,
                        yanchor="middle",
                    ),
                ),
                line=dict(color="rgba(100,100,100,0.3)", width=1),
                text=hover_text,
                hovertemplate="%{text}<extra></extra>",
                name="Score",
            )
        )

        # Add moving average
        window = min(5, len(df) // 3)
        if window > 1:
            df_sorted = df.sort_values("timestamp")
            rolling_mean = df_sorted["score"].rolling(window=window, center=True).mean()

            fig.add_trace(
                go.Scatter(
                    x=df_sorted["timestamp"],
                    y=rolling_mean,
                    mode="lines",
                    line=dict(color="red", width=2),
                    name=f"Moving Avg (window={window})",
                    hovertemplate="Avg: %{y:.4f}<extra></extra>",
                )
            )
    else:
        # If no timestamp, just plot by index
        fig.add_trace(
            go.Scatter(
                x=list(range(len(df))),
                y=df["score"],
                mode="markers+lines",
                marker=dict(size=8, color="blue"),
                text=hover_text,
                hovertemplate="%{text}<extra></extra>",
                name="Score",
            )
        )

    fig.update_layout(
        title="Score Progression Over Time",
        xaxis_title="Timestamp" if "timestamp" in df.columns else "Run Index",
        yaxis_title="Score",
        hovermode="x unified",
        template="plotly_white",
        height=450,
        margin=dict(t=60, b=60, l=60, r=120),  # Extra right margin for colorbar
        legend=dict(
            x=0.02,
            y=0.98,
            xanchor="left",
            yanchor="top",
            bgcolor="rgba(255, 255, 255, 0.8)",
            bordercolor="rgba(0, 0, 0, 0.2)",
            borderwidth=1,
        ),
    )

    return fig


def create_parameter_importance_plot(df, top_n=15):
    """Create parameter importance bar chart."""

    # Calculate correlations with score
    param_cols = [
        col
        for col in df.columns
        if col.startswith("trainer.") and pd.api.types.is_numeric_dtype(df[col])
    ]
    correlations = []

    for col in param_cols:
        if df[col].nunique() >= 2:
            corr = df[col].corr(df["score"])
            if not np.isnan(corr):
                correlations.append(
                    {"parameter": col.replace("trainer.", ""), "correlation": corr}
                )

    # Sort by absolute correlation
    corr_df = pd.DataFrame(correlations)
    corr_df["abs_corr"] = corr_df["correlation"].abs()
    corr_df = corr_df.nlargest(top_n, "abs_corr")

    # Create bar chart
    fig = go.Figure()

    colors = ["green" if x > 0 else "red" for x in corr_df["correlation"]]

    fig.add_trace(
        go.Bar(
            x=corr_df["correlation"],
            y=corr_df["parameter"],
            orientation="h",
            marker=dict(color=colors, opacity=0.7),
            text=[f"{x:.3f}" for x in corr_df["correlation"]],
            textposition="outside",
            hovertemplate="%{y}: %{x:.3f}<extra></extra>",
        )
    )

    fig.update_layout(
        title=f"Top {top_n} Parameter Correlations with Score",
        xaxis_title="Correlation",
        yaxis_title="Parameter",
        template="plotly_white",
        height=550,
        xaxis=dict(range=[-1, 1]),
        margin=dict(t=80, b=60, l=180, r=60),  # Extra left margin for parameter names
    )

    return fig


def create_score_distribution_plot(df):
    """Create score distribution histogram."""
    fig = go.Figure()

    fig.add_trace(
        go.Histogram(
            x=df["score"],
            nbinsx=30,
            marker=dict(
                color="green", opacity=0.7, line=dict(color="darkgreen", width=1)
            ),
            name="Score Distribution",
            hovertemplate="Score: %{x:.4f}<br>Count: %{y}<extra></extra>",
        )
    )

    # Add mean and median lines
    mean_score = df["score"].mean()
    median_score = df["score"].median()

    fig.add_vline(
        x=mean_score,
        line_dash="dash",
        line_color="red",
        annotation_text=f"Mean: {mean_score:.4f}",
    )
    fig.add_vline(
        x=median_score,
        line_dash="dash",
        line_color="blue",
        annotation_text=f"Median: {median_score:.4f}",
    )

    fig.update_layout(
        title="Score Distribution",
        xaxis_title="Score",
        yaxis_title="Count",
        template="plotly_white",
        height=450,
        showlegend=False,
        margin=dict(t=60, b=60, l=60, r=60),  # Add margins
    )

    return fig


def create_cost_distribution_plot(df):
    """Create cost distribution histogram."""
    fig = go.Figure()

    cost_col = "dollar_cost" if "dollar_cost" in df.columns else "cost"

    fig.add_trace(
        go.Histogram(
            x=df[cost_col],
            nbinsx=30,
            marker=dict(
                color="blue", opacity=0.7, line=dict(color="darkblue", width=1)
            ),
            name="Cost Distribution",
            hovertemplate="Cost: $%{x:.2f}<br>Count: %{y}<extra></extra>",
        )
    )

    # Add mean and median lines
    mean_cost = df[cost_col].mean()
    median_cost = df[cost_col].median()

    fig.add_vline(
        x=mean_cost,
        line_dash="dash",
        line_color="red",
        annotation_text=f"Mean: ${mean_cost:.2f}",
    )
    fig.add_vline(
        x=median_cost,
        line_dash="dash",
        line_color="blue",
        annotation_text=f"Median: ${median_cost:.2f}",
    )

    fig.update_layout(
        title="Cost Distribution",
        xaxis_title="Cost ($)",
        yaxis_title="Count",
        template="plotly_white",
        height=450,
        showlegend=False,
        margin=dict(t=60, b=60, l=60, r=60),  # Add margins
    )

    return fig


def create_parameter_correlation_plots(df, show_all=True):
    """Create scatterplots for all parameters vs score."""
    from plotly.subplots import make_subplots

    # Get parameter columns
    param_cols = [
        col
        for col in df.columns
        if col.startswith("trainer.") and pd.api.types.is_numeric_dtype(df[col])
    ]

    # Calculate correlations for all parameters
    correlations = []
    for col in param_cols:
        if df[col].nunique() >= 2:
            corr = abs(df[col].corr(df["score"]))
            if not np.isnan(corr):
                correlations.append((col, corr, df[col].corr(df["score"])))

    # Sort by absolute correlation
    correlations.sort(key=lambda x: x[1], reverse=True)

    # Use all parameters
    params_to_show = correlations

    if not params_to_show:
        fig = go.Figure()
        fig.add_annotation(text="No parameters to correlate", x=0.5, y=0.5)
        return fig

    # Calculate grid dimensions - use 3 columns for better visibility
    n_params = len(params_to_show)
    cols = 3
    rows = (n_params + cols - 1) // cols  # Ceiling division

    # Create subplots with more spacing
    fig = make_subplots(
        rows=rows,
        cols=cols,
        subplot_titles=[
            f"{p[0].replace('trainer.', '')}<br>r={p[2]:.3f}" for p in params_to_show
        ],
        vertical_spacing=0.10,  # Adjusted for more plots
        horizontal_spacing=0.15,  # Increased for better readability
    )

    for idx, (param, abs_corr, corr) in enumerate(params_to_show):
        row = idx // cols + 1
        col = idx % cols + 1

        # Create hover text
        hover_text = []
        for _, data_row in df.iterrows():
            text = f"<b>{param.replace('trainer.', '')}</b>: {data_row[param]:.6f}<br>"
            text += f"Score: {data_row['score']:.4f}<br>"
            if "run_name" in data_row:
                text += f"Run: {data_row.get('run_name', 'N/A')}"
            hover_text.append(text)

        # Add scatter plot
        fig.add_trace(
            go.Scatter(
                x=df[param],
                y=df["score"],
                mode="markers",
                marker=dict(
                    size=8,
                    color=df["score"],
                    colorscale="Viridis",
                    showscale=False,
                    line=dict(width=0.5, color="white"),
                ),
                text=hover_text,
                hovertemplate="%{text}<extra></extra>",
                showlegend=False,
            ),
            row=row,
            col=col,
        )

        # Add trend line
        if len(df) > 2:
            z = np.polyfit(df[param].values, df["score"].values, 1)
            p = np.poly1d(z)
            x_trend = np.linspace(df[param].min(), df[param].max(), 100)

            fig.add_trace(
                go.Scatter(
                    x=x_trend,
                    y=p(x_trend),
                    mode="lines",
                    line=dict(color="red", width=2, dash="dash"),
                    showlegend=False,
                    hoverinfo="skip",
                ),
                row=row,
                col=col,
            )

        # Update axes
        fig.update_xaxes(
            title_text=param.replace("trainer.", "")[:20], row=row, col=col
        )
        fig.update_yaxes(title_text="Score", row=row, col=col)

    # Dynamic height based on number of rows
    plot_height = max(800, rows * 250)  # 250px per row, minimum 800px

    fig.update_layout(
        title_text=f"All Parameter Correlations with Score ({n_params} parameters)",
        height=plot_height,
        template="plotly_white",
        showlegend=False,
        margin=dict(t=100, b=50, l=50, r=50),  # Add margins for better spacing
    )

    return fig


def create_pareto_frontier_plot(df):
    """Create Pareto frontier plot."""

    # Find Pareto frontier
    sorted_df = df.sort_values("dollar_cost")
    pareto_front = []
    max_score = -np.inf

    for _, row in sorted_df.iterrows():
        if row["score"] >= max_score:
            pareto_front.append(row)
            max_score = row["score"]

    pareto_df = pd.DataFrame(pareto_front)

    fig = go.Figure()

    # All points
    fig.add_trace(
        go.Scatter(
            x=df["dollar_cost"],
            y=df["score"],
            mode="markers",
            marker=dict(size=6, color="lightgray"),
            name="All Runs",
            hovertemplate="Cost: $%{x:.2f}<br>Score: %{y:.4f}<extra></extra>",
        )
    )

    # Pareto frontier
    if not pareto_df.empty:
        fig.add_trace(
            go.Scatter(
                x=pareto_df["dollar_cost"],
                y=pareto_df["score"],
                mode="lines+markers",
                line=dict(color="red", width=2),
                marker=dict(size=10, color="red"),
                name="Pareto Frontier",
                hovertemplate="<b>Efficient</b><br>Cost: $%{x:.2f}<br>Score: %{y:.4f}<extra></extra>",
            )
        )

    fig.update_layout(
        title="Efficiency Frontier (Pareto Optimal Runs)",
        xaxis_title="Cost ($)",
        yaxis_title="Score",
        template="plotly_white",
        height=400,
        hovermode="closest",
        legend=dict(
            x=0.02,
            y=0.98,
            xanchor="left",
            yanchor="top",
            bgcolor="rgba(255, 255, 255, 0.8)",
            bordercolor="rgba(0, 0, 0, 0.2)",
            borderwidth=1,
        ),
    )

    return fig


# Build the dashboard layout
app.layout = dbc.Container(
    [
        # Header
        dbc.Row(
            [
                dbc.Col(
                    [
                        html.H1(
                            "Sweep Analysis Dashboard", className="text-center mb-2"
                        ),
                        html.H5(
                            f"Sweep: {WANDB_SWEEP_NAME}",
                            className="text-center text-muted mb-4",
                        ),
                    ]
                )
            ]
        ),
        # Sky Jobs Monitoring Panel (moved to top)
        dbc.Row(
            [
                dbc.Col(
                    [
                        dbc.Card(
                            [
                                dbc.CardBody(
                                    [
                                        html.H5(
                                            "Sky Jobs Monitor",
                                            className="card-title",
                                        ),
                                        dbc.Row(
                                            [
                                                dbc.Col(
                                                    [
                                                        dbc.Button(
                                                            "Refresh Jobs Status",
                                                            id="refresh-jobs-button",
                                                            color="primary",
                                                            size="sm",
                                                            className="mb-2",
                                                        ),
                                                        dbc.Button(
                                                            "Show/Hide Logs",
                                                            id="toggle-logs-button",
                                                            color="secondary",
                                                            size="sm",
                                                            className="mb-2 ms-2",
                                                        ),
                                                        dbc.Button(
                                                            "Start New Worker",
                                                            id="start-worker-button",
                                                            color="success",
                                                            size="sm",
                                                            className="mb-2 ms-2",
                                                        ),
                                                        dcc.Loading(
                                                            id="loading-jobs",
                                                            type="circle",
                                                            children=[
                                                                html.Span(
                                                                    id="loading-jobs-trigger",
                                                                    style={
                                                                        "display": "none"
                                                                    },
                                                                )
                                                            ],
                                                            style={
                                                                "display": "inline-block",
                                                                "marginLeft": "10px",
                                                            },
                                                        ),
                                                        html.Div(
                                                            id="worker-status-message",
                                                            style={
                                                                "display": "inline-block",
                                                                "marginLeft": "10px",
                                                            },
                                                        ),
                                                    ],
                                                    width=12,
                                                ),
                                            ]
                                        ),
                                        html.Div(
                                            id="sky-jobs-output",
                                            style={
                                                "padding": "10px",
                                                "maxHeight": "400px",
                                                "overflowY": "auto",
                                            },
                                            children=html.Div(
                                                "Click 'Refresh Jobs Status' to load Sky jobs information",
                                                style={
                                                    "color": "#6c757d",
                                                    "fontStyle": "italic",
                                                },
                                            ),
                                        ),
                                        dcc.Interval(
                                            id="auto-refresh-interval",
                                            interval=30
                                            * 1000,  # Refresh every 30 seconds
                                            n_intervals=0,
                                            disabled=True,  # Start disabled
                                        ),
                                        dbc.Checklist(
                                            id="auto-refresh-toggle",
                                            options=[
                                                {
                                                    "label": "Auto-refresh (30s)",
                                                    "value": 1,
                                                }
                                            ],
                                            value=[],
                                            inline=True,
                                            className="mt-2",
                                        ),
                                    ]
                                )
                            ]
                        )
                    ],
                    width=12,
                )
            ],
            className="mb-4",
        ),
        # Summary Statistics Cards
        dbc.Row(
            [
                dbc.Col(
                    [
                        dbc.Card(
                            [
                                dbc.CardBody(
                                    [
                                        html.H4(
                                            max(0, len(df) - 1)
                                            if len(df) == 1
                                            and df.iloc[0]["run_name"] == ""
                                            else len(df),
                                            className="card-title text-primary",
                                        ),
                                        html.P("Total Runs", className="card-text"),
                                    ]
                                )
                            ]
                        )
                    ],
                    width=3,
                ),
                dbc.Col(
                    [
                        dbc.Card(
                            [
                                dbc.CardBody(
                                    [
                                        html.H4(
                                            f"{df['score'].max():.4f}"
                                            if not df.empty and df["score"].max() > 0
                                            else "N/A",
                                            className="card-title text-success",
                                        ),
                                        html.P("Best Score", className="card-text"),
                                    ]
                                )
                            ]
                        )
                    ],
                    width=3,
                ),
                dbc.Col(
                    [
                        dbc.Card(
                            [
                                dbc.CardBody(
                                    [
                                        html.H4(
                                            f"${df['dollar_cost'].sum():.2f}"
                                            if df["dollar_cost"].sum() > 0
                                            else "$0.00",
                                            className="card-title text-warning",
                                        ),
                                        html.P("Total Cost", className="card-text"),
                                    ]
                                )
                            ]
                        )
                    ],
                    width=3,
                ),
                dbc.Col(
                    [
                        dbc.Card(
                            [
                                dbc.CardBody(
                                    [
                                        html.H4(
                                            f"{df['runtime'].mean():.1f}s"
                                            if df["runtime"].mean() > 0
                                            else "N/A",
                                            className="card-title text-info",
                                        ),
                                        html.P("Avg Runtime", className="card-text"),
                                    ]
                                )
                            ]
                        )
                    ],
                    width=3,
                ),
            ],
            className="mb-4",
        ),
        # Filter Controls
        dbc.Row(
            [
                dbc.Col(
                    [
                        dbc.Card(
                            [
                                dbc.CardBody(
                                    [
                                        html.H5("Filters", className="card-title"),
                                        html.Label("Score Range:"),
                                        dcc.RangeSlider(
                                            id="score-range-slider",
                                            min=df["score"].min(),
                                            max=df["score"].max(),
                                            value=[
                                                df["score"].min(),
                                                df["score"].max(),
                                            ],
                                            marks={
                                                float(
                                                    df["score"].min()
                                                ): f"{df['score'].min():.3f}",
                                                float(
                                                    df["score"].max()
                                                ): f"{df['score'].max():.3f}",
                                            },
                                            tooltip={
                                                "placement": "bottom",
                                                "always_visible": False,
                                            },
                                            step=0.001,
                                        ),
                                        html.Br(),
                                        html.Label("Cost Range ($):"),
                                        dcc.RangeSlider(
                                            id="cost-range-slider",
                                            min=df["dollar_cost"].min(),
                                            max=df["dollar_cost"].max(),
                                            value=[
                                                df["dollar_cost"].min(),
                                                df["dollar_cost"].max(),
                                            ],
                                            marks={
                                                float(
                                                    df["dollar_cost"].min()
                                                ): f"${df['dollar_cost'].min():.2f}",
                                                float(
                                                    df["dollar_cost"].max()
                                                ): f"${df['dollar_cost'].max():.2f}",
                                            },
                                            tooltip={
                                                "placement": "bottom",
                                                "always_visible": False,
                                            },
                                            step=0.01,
                                        ),
                                        html.Br(),
                                        dbc.Button(
                                            "Reset Filters",
                                            id="reset-button",
                                            color="secondary",
                                            size="sm",
                                            className="mt-2",
                                        ),
                                    ]
                                )
                            ]
                        )
                    ],
                    width=12,
                )
            ],
            className="mb-4",
        ),
        # Main Visualizations
        dbc.Row(
            [
                dbc.Col(
                    [
                        dbc.Card(
                            [
                                dbc.CardBody(
                                    [
                                        dcc.Loading(
                                            id="loading-cost-score",
                                            type="default",
                                            children=[
                                                dcc.Graph(id="cost-vs-score-plot")
                                            ],
                                        )
                                    ]
                                )
                            ]
                        )
                    ],
                    width=6,
                ),
                dbc.Col(
                    [
                        dbc.Card(
                            [
                                dbc.CardBody(
                                    [
                                        dcc.Loading(
                                            id="loading-param-importance",
                                            type="default",
                                            children=[
                                                dcc.Graph(
                                                    id="parameter-importance-plot"
                                                )
                                            ],
                                        )
                                    ]
                                )
                            ]
                        )
                    ],
                    width=6,
                ),
            ],
            className="mb-4",
        ),
        dbc.Row(
            [
                dbc.Col(
                    [
                        dbc.Card(
                            [
                                dbc.CardBody(
                                    [
                                        dcc.Loading(
                                            id="loading-timeline",
                                            type="default",
                                            children=[dcc.Graph(id="timeline-plot")],
                                        )
                                    ]
                                )
                            ]
                        )
                    ],
                    width=6,
                ),
                dbc.Col(
                    [
                        dbc.Card(
                            [
                                dbc.CardBody(
                                    [
                                        dcc.Loading(
                                            id="loading-pareto",
                                            type="default",
                                            children=[dcc.Graph(id="pareto-plot")],
                                        )
                                    ]
                                )
                            ]
                        )
                    ],
                    width=6,
                ),
            ],
            className="mb-4",
        ),
        # Distribution Plots
        dbc.Row(
            [
                dbc.Col(
                    [
                        dbc.Card(
                            [
                                dbc.CardBody(
                                    [
                                        dcc.Loading(
                                            id="loading-score-dist",
                                            type="default",
                                            children=[
                                                dcc.Graph(id="score-distribution-plot")
                                            ],
                                        )
                                    ]
                                )
                            ]
                        )
                    ],
                    width=6,
                ),
                dbc.Col(
                    [
                        dbc.Card(
                            [
                                dbc.CardBody(
                                    [
                                        dcc.Loading(
                                            id="loading-cost-dist",
                                            type="default",
                                            children=[
                                                dcc.Graph(id="cost-distribution-plot")
                                            ],
                                        )
                                    ]
                                )
                            ]
                        )
                    ],
                    width=6,
                ),
            ],
            className="mb-4",
        ),
        # Parameter Correlation Plots
        dbc.Row(
            [
                dbc.Col(
                    [
                        dbc.Card(
                            [
                                dbc.CardBody(
                                    [
                                        dcc.Loading(
                                            id="loading-correlations",
                                            type="default",
                                            children=[
                                                dcc.Graph(
                                                    id="parameter-correlation-plots"
                                                )
                                            ],
                                        )
                                    ]
                                )
                            ]
                        )
                    ],
                    width=12,
                ),
            ],
            className="mb-4",
        ),
        # Selected Run Details
        dbc.Row(
            [
                dbc.Col(
                    [
                        dbc.Card(
                            [
                                dbc.CardBody(
                                    [
                                        html.H5(
                                            "Selected Run Details",
                                            className="card-title",
                                        ),
                                        html.Div(
                                            id="selected-run-details",
                                            children="Click on a point in any plot to see run details",
                                        ),
                                    ]
                                )
                            ]
                        )
                    ],
                    width=12,
                )
            ],
            className="mb-4",
        ),
        # Store for filtered data
        dcc.Store(id="filtered-data"),
        dcc.Store(id="logs-visible", data=False),
        # Modal for cancel confirmation
        dbc.Modal(
            [
                dbc.ModalHeader(dbc.ModalTitle("Confirm Job Cancellation")),
                dbc.ModalBody(id="cancel-modal-body"),
                dbc.ModalFooter(
                    [
                        dbc.Button(
                            "Cancel Job",
                            id="confirm-cancel-btn",
                            color="danger",
                            className="ms-auto",
                        ),
                        dbc.Button("Close", id="close-cancel-modal", color="secondary"),
                    ]
                ),
            ],
            id="cancel-modal",
            is_open=False,
        ),
        dcc.Store(id="job-to-cancel"),
        # Interval for delayed refresh after cancel
        dcc.Interval(
            id="cancel-refresh-interval",
            interval=3000,  # 3 seconds
            n_intervals=0,
            max_intervals=1,  # Only run once
            disabled=True,
        ),
        # Interval for delayed refresh after starting worker
        dcc.Interval(
            id="worker-refresh-interval",
            interval=5000,  # 5 seconds
            n_intervals=0,
            max_intervals=3,  # Check 3 times
            disabled=True,
        ),
    ],
    fluid=True,
    style={"backgroundColor": COLORS["background"]},
)


# Add callbacks for interactivity
@app.callback(
    Output("filtered-data", "data"),
    [
        Input("score-range-slider", "value"),
        Input("cost-range-slider", "value"),
        Input("reset-button", "n_clicks"),
    ],
)
def filter_data(score_range, cost_range, reset_clicks):
    """Filter data based on slider values."""
    ctx = callback_context

    if ctx.triggered and ctx.triggered[0]["prop_id"] == "reset-button.n_clicks":
        # Reset filters
        filtered_df = df
    else:
        # Apply filters
        filtered_df = df[
            (df["score"] >= score_range[0])
            & (df["score"] <= score_range[1])
            & (df["dollar_cost"] >= cost_range[0])
            & (df["dollar_cost"] <= cost_range[1])
        ]

    return filtered_df.to_dict("records")


@app.callback(
    [
        Output("cost-vs-score-plot", "figure"),
        Output("timeline-plot", "figure"),
        Output("parameter-importance-plot", "figure"),
        Output("pareto-plot", "figure"),
        Output("score-distribution-plot", "figure"),
        Output("cost-distribution-plot", "figure"),
        Output("parameter-correlation-plots", "figure"),
    ],
    [Input("filtered-data", "data")],
)
def update_plots(filtered_data):
    """Update all plots based on filtered data."""
    filtered_df = pd.DataFrame(filtered_data)

    # Check if this is the dummy dataframe (no real data)
    if filtered_df.empty or (
        len(filtered_df) == 1 and filtered_df.iloc[0]["run_name"] == ""
    ):
        # Return empty figures if no data
        empty_fig = go.Figure()
        empty_fig.add_annotation(
            text="No sweep runs completed yet. Waiting for data...",
            xref="paper",
            yref="paper",
            x=0.5,
            y=0.5,
            showarrow=False,
        )
        return (
            empty_fig,
            empty_fig,
            empty_fig,
            empty_fig,
            empty_fig,
            empty_fig,
            empty_fig,
        )

    # Create updated plots
    cost_vs_score = create_cost_vs_score_plot(filtered_df)
    timeline = create_timeline_plot(filtered_df)
    param_importance = create_parameter_importance_plot(filtered_df)
    pareto = create_pareto_frontier_plot(filtered_df)
    score_dist = create_score_distribution_plot(filtered_df)
    cost_dist = create_cost_distribution_plot(filtered_df)
    param_correlations = create_parameter_correlation_plots(filtered_df)

    return (
        cost_vs_score,
        timeline,
        param_importance,
        pareto,
        score_dist,
        cost_dist,
        param_correlations,
    )


@app.callback(
    Output("selected-run-details", "children"),
    [Input("cost-vs-score-plot", "clickData"), Input("timeline-plot", "clickData")],
    [State("filtered-data", "data")],
)
def display_run_details(cost_click, timeline_click, filtered_data):
    """Display detailed information about selected run."""
    ctx = callback_context

    if not ctx.triggered:
        return "Click on a point in any plot to see run details"

    # Get the clicked data
    trigger_id = ctx.triggered[0]["prop_id"].split(".")[0]

    if trigger_id == "cost-vs-score-plot" and cost_click:
        point_index = cost_click["points"][0]["pointIndex"]
    elif trigger_id == "timeline-plot" and timeline_click:
        point_index = timeline_click["points"][0]["pointIndex"]
    else:
        return "Click on a point in any plot to see run details"

    # Get the selected run
    filtered_df = pd.DataFrame(filtered_data)
    if point_index >= len(filtered_df):
        return "Error: Invalid point selection"

    run = filtered_df.iloc[point_index]

    # Create detailed view
    details = [
        html.H6(f"Run: {run.get('run_name', 'Unknown')}", className="mb-3"),
        dbc.Row(
            [
                dbc.Col(
                    [
                        html.Strong("Metrics:"),
                        html.Ul(
                            [
                                html.Li(f"Score: {run.get('score', 'N/A'):.4f}"),
                                html.Li(f"Cost: ${run.get('dollar_cost', 'N/A'):.4f}"),
                                html.Li(f"Runtime: {run.get('runtime', 'N/A'):.1f}s"),
                                html.Li(f"Timestamp: {run.get('timestamp', 'N/A')}"),
                            ]
                        ),
                    ],
                    width=4,
                ),
                dbc.Col(
                    [
                        html.Strong("Key Hyperparameters:"),
                        html.Ul(
                            [
                                html.Li(
                                    f"{param.replace('trainer.', '')}: {run.get(param, 'N/A'):.4f}"
                                    if isinstance(run.get(param), (int, float))
                                    else f"{param.replace('trainer.', '')}: {run.get(param, 'N/A')}"
                                )
                                for param in [
                                    "trainer.optimizer.learning_rate",
                                    "trainer.ppo.clip_coef",
                                    "trainer.ppo.vf_coef",
                                    "trainer.total_timesteps",
                                    "trainer.update_epochs",
                                ]
                                if param in run
                            ]
                        ),
                    ],
                    width=8,
                ),
            ]
        ),
    ]

    return details


# Callback for Sky Jobs monitoring
@app.callback(
    [
        Output("sky-jobs-output", "children"),
        Output("loading-jobs-trigger", "children"),
    ],
    [
        Input("refresh-jobs-button", "n_clicks"),
        Input("auto-refresh-interval", "n_intervals"),
        Input("toggle-logs-button", "n_clicks"),
    ],
    [State("logs-visible", "data")],
)
def update_sky_jobs(refresh_clicks, auto_intervals, toggle_clicks, logs_visible):
    """Fetch and display Sky jobs status and optionally logs."""
    ctx = callback_context

    if not ctx.triggered:
        return (
            html.Div(
                "Click 'Refresh Jobs Status' to load Sky jobs information",
                style={"color": "#6c757d", "fontStyle": "italic"},
            ),
            "",  # No loading trigger
        )

    trigger_id = ctx.triggered[0]["prop_id"].split(".")[0]

    # Handle toggle button
    if trigger_id == "toggle-logs-button":
        logs_visible = not logs_visible

    try:
        # Get Sky jobs status
        result = subprocess.run(
            ["sky", "jobs", "queue", "-s"],
            capture_output=True,
            text=True,
            timeout=10,
        )

        if result.returncode != 0:
            return (
                html.Div(
                    f"Error fetching Sky jobs: {result.stderr}",
                    style={"color": "#dc3545"},
                ),
                "",
            )

        # Parse the output into a structured format
        lines = result.stdout.strip().split("\n")

        # Find the header line and data lines
        header_line = None
        data_lines = []
        in_progress_count = 0

        for i, line in enumerate(lines):
            if "In progress tasks:" in line:
                # Extract the count of running tasks
                parts = line.split(":")
                if len(parts) > 1:
                    count_part = parts[1].strip().split()[0]
                    in_progress_count = count_part
            elif line.startswith("ID"):
                header_line = line
            elif header_line and line.strip() and line[0].isdigit():
                data_lines.append(line)

        # Create the formatted output
        output_components = []

        # Add status badge
        output_components.append(
            dbc.Alert(
                [
                    html.H6("Sky Jobs Status", className="alert-heading mb-2"),
                    html.P(f"In progress tasks: {in_progress_count}", className="mb-0"),
                ],
                color="info",
                className="mb-3",
            )
        )

        # Parse and create table if we have data
        if header_line and data_lines:
            # Parse header
            header_line.split()  # Headers are known, no need to store

            # Parse data rows
            rows = []
            for line in data_lines:
                # Split by multiple spaces to handle column alignment
                parts = line.split()
                if len(parts) >= 10:  # Ensure we have enough columns
                    job_id = parts[0]
                    # task = parts[1]  # Not used, skipping
                    name = parts[2]

                    # Parse resources (handle brackets)
                    resource_start = 3
                    resource_parts = []
                    for i in range(resource_start, len(parts)):
                        resource_parts.append(parts[i])
                        if "]" in parts[i]:
                            resource_end = i + 1
                            break
                    else:
                        resource_end = resource_start + 1
                    requested = " ".join(resource_parts)

                    # Parse submitted time
                    submitted_idx = resource_end
                    submitted_parts = []
                    # Look for time pattern (e.g., "1 hr ago")
                    for i in range(submitted_idx, min(submitted_idx + 3, len(parts))):
                        submitted_parts.append(parts[i])
                        if parts[i] == "ago":
                            submitted_idx = i + 1
                            break
                    else:
                        submitted_idx = min(submitted_idx + 2, len(parts))
                    submitted = " ".join(submitted_parts)

                    # Parse durations - they're in format like "1h 2m" or "12s 55m"
                    # Total duration
                    tot_duration_idx = submitted_idx
                    tot_duration_parts = []
                    for i in range(
                        tot_duration_idx, min(tot_duration_idx + 2, len(parts))
                    ):
                        if any(c in parts[i] for c in ["h", "m", "s"]):
                            tot_duration_parts.append(parts[i])
                        else:
                            break
                    tot_duration = (
                        " ".join(tot_duration_parts) if tot_duration_parts else "N/A"
                    )

                    # Job duration
                    job_duration_idx = tot_duration_idx + len(tot_duration_parts)
                    job_duration_parts = []
                    for i in range(
                        job_duration_idx, min(job_duration_idx + 2, len(parts))
                    ):
                        if i < len(parts) and any(
                            c in parts[i] for c in ["h", "m", "s"]
                        ):
                            job_duration_parts.append(parts[i])
                        else:
                            break
                    job_duration = (
                        " ".join(job_duration_parts) if job_duration_parts else "N/A"
                    )

                    # Status is usually the last or second to last item
                    status = "UNKNOWN"
                    for part in reversed(parts):
                        if part in [
                            "RUNNING",
                            "SUCCEEDED",
                            "FAILED",
                            "PENDING",
                            "CANCELLED",
                        ]:
                            status = part
                            break

                    # Create status badge
                    status_color = "success" if status == "RUNNING" else "warning"

                    rows.append(
                        html.Tr(
                            [
                                html.Td(job_id, style={"fontFamily": "monospace"}),
                                html.Td(name[:30] + "..." if len(name) > 30 else name),
                                html.Td(requested),
                                html.Td(submitted),
                                html.Td(tot_duration),
                                html.Td(job_duration),
                                html.Td(
                                    dbc.Badge(status, color=status_color, pill=True)
                                ),
                                html.Td(
                                    dbc.Button(
                                        "Cancel",
                                        id={"type": "cancel-job-btn", "index": job_id},
                                        color="danger",
                                        size="sm",
                                        outline=True,
                                        disabled=status != "RUNNING",
                                    )
                                    if status == "RUNNING"
                                    else "",
                                    style={"textAlign": "center"},
                                ),
                            ]
                        )
                    )

            # Create table
            table = dbc.Table(
                [
                    html.Thead(
                        [
                            html.Tr(
                                [
                                    html.Th("ID"),
                                    html.Th("Name"),
                                    html.Th("Resources"),
                                    html.Th("Submitted"),
                                    html.Th("Total Duration"),
                                    html.Th("Job Duration"),
                                    html.Th("Status"),
                                    html.Th("Actions", style={"textAlign": "center"}),
                                ]
                            )
                        ]
                    ),
                    html.Tbody(rows),
                ],
                striped=True,
                bordered=True,
                hover=True,
                responsive=True,
                size="sm",
            )
            output_components.append(table)

        # Add logs if visible
        if logs_visible and data_lines:
            # Get the first job ID for logs
            first_line_parts = data_lines[0].split()
            if first_line_parts and first_line_parts[0].isdigit():
                job_id = first_line_parts[0]
                log_result = subprocess.run(
                    ["sky", "jobs", "logs", job_id, "--tail", "50"],
                    capture_output=True,
                    text=True,
                    timeout=10,
                )
                if log_result.returncode == 0:
                    output_components.append(
                        html.Div(
                            [
                                html.Hr(),
                                html.H6(
                                    f"Recent Logs - Job {job_id}", className="mb-2"
                                ),
                                html.Pre(
                                    log_result.stdout,
                                    style={
                                        "backgroundColor": "#f8f9fa",
                                        "padding": "10px",
                                        "borderRadius": "5px",
                                        "maxHeight": "300px",
                                        "overflowY": "auto",
                                        "fontSize": "11px",
                                        "fontFamily": "monospace",
                                    },
                                ),
                            ]
                        )
                    )

        return html.Div(output_components), ""

    except subprocess.TimeoutExpired:
        return (
            html.Div(
                "Timeout: Sky command took too long to respond",
                style={"color": "#ffc107"},
            ),
            "",
        )
    except FileNotFoundError:
        return (
            html.Div(
                "Sky CLI not found. Make sure 'sky' is installed and in PATH",
                style={"color": "#dc3545"},
            ),
            "",
        )
    except Exception as e:
        return html.Div(f"Error: {str(e)}", style={"color": "#dc3545"}), ""


# Callback to toggle logs visibility
@app.callback(
    Output("logs-visible", "data"),
    [Input("toggle-logs-button", "n_clicks")],
    [State("logs-visible", "data")],
)
def toggle_logs_visibility(n_clicks, current_state):
    """Toggle the visibility state for logs."""
    if n_clicks:
        return not current_state
    return current_state


# Callback for auto-refresh toggle
@app.callback(
    Output("auto-refresh-interval", "disabled"),
    [Input("auto-refresh-toggle", "value")],
)
def toggle_auto_refresh(value):
    """Enable/disable auto-refresh based on checkbox."""
    return len(value) == 0  # Disabled when checkbox is unchecked


# Callback to trigger refresh after cancel delay
@app.callback(
    [
        Output("refresh-jobs-button", "n_clicks"),
        Output("cancel-refresh-interval", "disabled", allow_duplicate=True),
    ],
    [Input("cancel-refresh-interval", "n_intervals")],
    [State("refresh-jobs-button", "n_clicks")],
    prevent_initial_call=True,
)
def delayed_refresh_after_cancel(n_intervals, current_clicks):
    """Trigger a refresh after the cancel delay."""
    if n_intervals and n_intervals > 0:
        # Trigger refresh and disable the interval again
        return (current_clicks or 0) + 1, True
    return current_clicks or 0, True


# Callback to start a new worker
@app.callback(
    [
        Output("worker-status-message", "children"),
        Output("worker-refresh-interval", "disabled"),
    ],
    [Input("start-worker-button", "n_clicks")],
    prevent_initial_call=True,
)
def start_new_worker(n_clicks):
    """Start a new sweep worker using skypilot."""
    if not n_clicks:
        raise dash.exceptions.PreventUpdate

    # Show starting message immediately
    starting_message = dbc.Alert(
        [
            dbc.Spinner(size="sm", spinner_class_name="me-2"),
            f"Starting new worker for {WANDB_SWEEP_NAME}...",
        ],
        color="info",
        dismissable=True,
        duration=15000,  # Show for 15 seconds
    )

    # Run the launch command in a background thread
    def run_launch():
        try:
            # Get the metta root directory (two levels up from this script)
            metta_root = os.path.abspath(
                os.path.join(os.path.dirname(__file__), "../..")
            )

            # Build the command with venv activation [[memory:4662310]]
            cmd = (
                f"source {metta_root}/.venv/bin/activate && "
                f"./devops/skypilot/launch.py --no-spot sweep run={WANDB_SWEEP_NAME}"
            )

            print(f"Executing: {cmd}")
            print(f"Working directory: {metta_root}")

            result = subprocess.run(
                cmd,
                shell=True,  # Need shell=True for source command
                capture_output=True,
                text=True,
                timeout=60,  # 60 second timeout for launch (increased)
                cwd=metta_root,  # Ensure we're in the metta root directory
                executable="/bin/bash",  # Use bash for source command
            )

            if result.returncode == 0:
                print(f"Successfully started new worker for {WANDB_SWEEP_NAME}")
                print(f"Output: {result.stdout}")
            else:
                print(f"Failed to start worker: {result.stderr}")

        except subprocess.TimeoutExpired:
            print("Launch command timed out - worker may still be starting")
        except Exception as e:
            print(f"Error starting worker: {str(e)}")

    # Start launch in background thread
    launch_thread = threading.Thread(target=run_launch)
    launch_thread.daemon = True
    launch_thread.start()

    # Return status message and enable periodic refresh
    return starting_message, False  # Enable the worker refresh interval


# Callback to trigger refresh after starting worker
@app.callback(
    [
        Output("refresh-jobs-button", "n_clicks", allow_duplicate=True),
        Output("worker-refresh-interval", "disabled", allow_duplicate=True),
    ],
    [Input("worker-refresh-interval", "n_intervals")],
    [State("refresh-jobs-button", "n_clicks")],
    prevent_initial_call=True,
)
def refresh_after_worker_start(n_intervals, current_clicks):
    """Trigger refreshes after starting a worker."""
    if n_intervals and n_intervals > 0:
        # Trigger refresh
        new_clicks = (current_clicks or 0) + 1
        # Disable after max intervals (handled by max_intervals setting)
        should_disable = n_intervals >= 3
        return new_clicks, should_disable
    return current_clicks or 0, False


# Callback to open cancel modal
@app.callback(
    [
        Output("cancel-modal", "is_open"),
        Output("cancel-modal-body", "children"),
        Output("job-to-cancel", "data"),
    ],
    [
        Input({"type": "cancel-job-btn", "index": ALL}, "n_clicks"),
        Input("close-cancel-modal", "n_clicks"),
    ],
    [State("cancel-modal", "is_open")],
)
def toggle_cancel_modal(cancel_clicks, close_click, is_open):
    """Open/close the cancel confirmation modal."""
    ctx = callback_context

    if not ctx.triggered:
        return False, "", None

    trigger_id = ctx.triggered[0]["prop_id"]

    # Handle close button
    if "close-cancel-modal" in trigger_id:
        return False, "", None

    # Handle cancel button clicks
    if "cancel-job-btn" in trigger_id:
        # Check if any button was actually clicked
        if any(cancel_clicks):
            # Find which button was clicked
            button_id = ctx.triggered[0]["prop_id"].split(".")[0]
            import json

            button_dict = json.loads(button_id)
            job_id = button_dict["index"]

            modal_body = f"Are you sure you want to cancel job {job_id}? This action cannot be undone."
            return True, modal_body, job_id

    return False, "", None


# Callback to handle cancel confirmation and close modal immediately
@app.callback(
    [
        Output("cancel-modal", "is_open", allow_duplicate=True),
        Output("sky-jobs-output", "children", allow_duplicate=True),
        Output("cancel-refresh-interval", "disabled"),
    ],
    [Input("confirm-cancel-btn", "n_clicks")],
    [State("job-to-cancel", "data")],
    prevent_initial_call=True,
)
def cancel_job(confirm_clicks, job_id):
    """Handle job cancellation with immediate UI feedback."""
    if not confirm_clicks or not job_id:
        raise dash.exceptions.PreventUpdate

    # Immediately close modal and show cancelling status
    cancelling_message = html.Div(
        [
            dbc.Alert(
                [
                    dbc.Spinner(size="sm", spinner_class_name="me-2"),
                    f"Cancelling job {job_id}...",
                ],
                color="warning",
                className="mb-3",
            ),
            html.P(
                "The job list will refresh automatically in a few seconds.",
                style={"color": "#6c757d", "fontStyle": "italic"},
            ),
        ]
    )

    # Start the cancel command in a separate thread to avoid blocking
    def run_cancel():
        try:
            result = subprocess.run(
                ["sky", "jobs", "cancel", "-y", str(job_id)],
                capture_output=True,
                text=True,
                timeout=10,
            )
            if result.returncode == 0:
                print(f"Successfully cancelled job {job_id}")
            else:
                print(f"Failed to cancel job {job_id}: {result.stderr}")
        except Exception as e:
            print(f"Error cancelling job {job_id}: {str(e)}")

    # Run cancel in background thread
    cancel_thread = threading.Thread(target=run_cancel)
    cancel_thread.daemon = True
    cancel_thread.start()

    # Immediately return with closed modal, status message, and enable delayed refresh
    return False, cancelling_message, False  # Enable the interval timer


# Run the dashboard
if __name__ == "__main__":
    print("\n" + "=" * 50)
    print("Starting Interactive Dashboard...")
    print("=" * 50)
    print("\nDashboard Statistics:")
    print(f"  - Total Runs: {len(df)}")
    print(f"  - Best Score: {df['score'].max():.4f}")
    print(f"  - Total Cost: ${df['dollar_cost'].sum():.2f}")
    print(f"  - Parameters: {len(param_cols)}")
    print("\n" + "=" * 50)

    # Run the app as a web server
    print("\n" + "=" * 50)
    print("Dashboard is running at: http://127.0.0.1:8050/")
    print("Press Ctrl+C to stop the server")
    print("=" * 50 + "\n")
    print("Usage examples:")
    print(f"  python sweep_dashboard.py --sweep-name {WANDB_SWEEP_NAME}")
    print(
        f"  python sweep_dashboard.py --sweep-name {WANDB_SWEEP_NAME} --entity {WANDB_ENTITY} --project {WANDB_PROJECT}"
    )
    print(
        f"  python sweep_dashboard.py --sweep-name {WANDB_SWEEP_NAME} --max-observations 500 --hourly-cost 5.0"
    )

    app.run(host="127.0.0.1", port=8050, debug=False)
