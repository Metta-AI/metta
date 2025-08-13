#!/usr/bin/env python
# coding: utf-8

"""
Interactive Sweep Analysis Dashboard - Plotly Dash
This creates a professional, interactive dashboard for sweep analysis using Plotly Dash.
The dashboard provides WandB-quality visualizations with full interactivity and detailed hover information.
"""

# Import required libraries
import os
import sys
import warnings

# Add the metta module to path before importing from it
sys.path.append(os.path.abspath("../.."))

import dash
import dash_bootstrap_components as dbc
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from dash import Input, Output, State, callback_context, dcc, html
from metta.sweep.wandb_utils import deep_clean, get_sweep_runs

warnings.filterwarnings("ignore")

# Configuration
WANDB_ENTITY = "metta-research"
WANDB_PROJECT = "metta"
WANDB_SWEEP_NAME = "axel.arena_phased_812.v1"
MAX_OBSERVATIONS = 1000
HOURLY_COST = 4.6  # Dollar cost per hour for instance


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
                                            len(df), className="card-title text-primary"
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
                                            f"{df['score'].max():.4f}",
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
                                            f"${df['dollar_cost'].sum():.2f}",
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
                                            f"{df['runtime'].mean():.1f}s",
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
                    [dbc.Card([dbc.CardBody([dcc.Graph(id="cost-vs-score-plot")])])],
                    width=6,
                ),
                dbc.Col(
                    [
                        dbc.Card(
                            [dbc.CardBody([dcc.Graph(id="parameter-importance-plot")])]
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
                    [dbc.Card([dbc.CardBody([dcc.Graph(id="timeline-plot")])])], width=6
                ),
                dbc.Col(
                    [dbc.Card([dbc.CardBody([dcc.Graph(id="pareto-plot")])])], width=6
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
                            [dbc.CardBody([dcc.Graph(id="score-distribution-plot")])]
                        )
                    ],
                    width=6,
                ),
                dbc.Col(
                    [
                        dbc.Card(
                            [dbc.CardBody([dcc.Graph(id="cost-distribution-plot")])]
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
                                    [dcc.Graph(id="parameter-correlation-plots")]
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

    if filtered_df.empty:
        # Return empty figures if no data
        empty_fig = go.Figure()
        empty_fig.add_annotation(
            text="No data matches the current filters",
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

    app.run(host="127.0.0.1", port=8050, debug=False)
