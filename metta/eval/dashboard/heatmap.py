"""
Return an embeddable HTML snippet that shows a policy‑evaluation heat‑map.

Features
--------
* Native Plotly hover box on every cell contains basic data.
* A dedicated map viewer panel shows maps when hovering over cells or x-axis labels.
* The 'Overall' column is excluded from map viewer behavior.
* All IDs are namespaced; multiple snippets can coexist in the same page.

External boiler‑plate (include once per page)
---------------------------------------------
    <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
"""

from __future__ import annotations

import json
import logging
import uuid
from typing import Dict, List, Literal, Tuple

import pandas as pd
import plotly.graph_objects as go
from plotly.utils import PlotlyJSONEncoder

from metta.eval.dashboard.mapviewer import create_map_viewer_html, get_map_viewer_js_functions
from metta.eval.eval_stats_db import EvalStatsDB


# --------------------------------------------------------------------------- #
# helpers                                                                    #
# --------------------------------------------------------------------------- #
def _format_metric(metric: str) -> str:
    return metric.replace("_", " ").capitalize()


def _wandb_url(uri: str, entity: str = "metta-research", project: str = "metta") -> str:
    if uri.startswith("wandb://run/"):
        uri = uri[len("wandb://run/") :]
    if ":v" in uri:
        uri = uri.split(":v")[0]
    return f"https://wandb.ai/{entity}/{project}/runs/{uri}"


# --------------------------------------------------------------------------- #
# core figure builder                                                         #
# --------------------------------------------------------------------------- #
def _build_figure(
    matrix: pd.DataFrame,
    metric: str,
    *,
    colorscale,
    score_range: Tuple[float, float],
    height: int,
    width: int,
) -> go.Figure:
    eval_names: List[str] = matrix.columns.tolist()
    # Remove the replay_url_map column if it exists
    if "replay_url_map" in matrix.columns:
        matrix = matrix.drop(columns=["replay_url_map"])
    policy_rows: List[str] = (
        matrix.pop("policy_uri").tolist() if "policy_uri" in matrix.columns else matrix.index.tolist()
    )

    # Create mapping of full paths to short display names
    short_names = [name.split("/")[-1] if "/" in name and name != "Overall" else name for name in eval_names]

    # Create image map with short names as keys
    img_base = "https://softmax-public.s3.amazonaws.com/policydash/evals/img"
    img_map = {short_name: f"{img_base}/{short_name.lower()}.png" for short_name in short_names}
    img_map["Overall"] = ""  # no tooltip image

    z = [matrix.mean().tolist(), matrix.max().tolist()] + matrix.values.tolist()
    y_labels = ["Mean", "Max"] + policy_rows

    # customdata for each cell (same for every row of a given column)
    customdata = []
    for _ in y_labels:
        row_data = []
        for i, name in enumerate(eval_names):
            short_name = short_names[i]
            row_data.append({"img": img_map[short_name], "shortName": short_name, "fullName": name})
        customdata.append(row_data)

    # ---------------------------------------------------------------- chart
    fig = go.Figure(
        go.Heatmap(
            z=z,
            customdata=customdata,
            x=short_names,  # Use short names for display on x-axis
            y=y_labels,
            colorscale=colorscale,
            zmin=score_range[0],
            zmax=score_range[1],
            colorbar=dict(title="Score"),
            hovertemplate=(
                "<b>Policy:</b> %{y}<br>"
                + "<b>Evaluation:</b> %{customdata.fullName}<br>"
                + "<b>Score:</b> %{z:.2f}<extra></extra>"
            ),
        )
    )

    ticktext = [
        f"<b>{lbl}</b>" if i < 2 else f'<a href="{_wandb_url(lbl)}">{lbl}</a>' for i, lbl in enumerate(y_labels)
    ]
    fig.update_layout(
        yaxis=dict(tickmode="array", tickvals=list(range(len(y_labels))), ticktext=ticktext),
        xaxis=dict(tickangle=-45),
        title=f"{_format_metric(metric)} Policy‑Evaluation Matrix",
        height=height + 50,
        width=width,
        margin=dict(l=50, r=50, t=50, b=100),
        plot_bgcolor="white",
        showlegend=False,
        meta=dict(eval_image_map=img_map, short_names=short_names, full_names=eval_names),
    )

    # dashed rectangle around aggregates
    fig.add_shape(
        type="rect",
        x0=-0.5,
        x1=len(eval_names) - 0.5,
        y0=-0.5,
        y1=1.5,
        line=dict(color="rgba(0,0,0,0.45)", width=2, dash="dash"),
        fillcolor="rgba(0,0,0,0)",
        layer="above",
    )
    return fig


# --------------------------------------------------------------------------- #
# public API                                                                  #
# --------------------------------------------------------------------------- #
def create_heatmap_html_snippet(
    matrix: pd.DataFrame,
    metric: str,
    *,
    height: int = 600,
    width: int = 900,
) -> str:
    """
    Return an HTML fragment containing the heat‑map `<div>` plus scoped JS.

    Includes a dedicated map viewer panel that displays environment maps
    when hovering over cells or x-axis labels.

    Caller must embed Plotly + the CSS styles (imported from mapviewer module).
    """
    if matrix.empty:
        return "<p>No data available</p>"

    score_range = (0, 1)
    colorscale = [
        # Red
        [0.0, "rgb(235, 40, 40)"],
        [0.5, "rgb(235, 40, 40)"],
        # Yellow
        [0.8, "rgb(225,210,80)"],
        # Light Green
        [0.95, "rgb(195,230,80)"],
        # Green
        [1.0, "rgb(20, 230, 80)"],
    ]

    fig = _build_figure(matrix, metric, colorscale=colorscale, score_range=score_range, height=height, width=width)
    uid = f"heat_{uuid.uuid4().hex[:8]}"
    fig_json = json.dumps(fig, cls=PlotlyJSONEncoder)

    # Get policy names from index
    policy_rows = matrix.index.tolist()
    policy_rows_json = json.dumps(policy_rows)

    # Get the replay_url_map from the matrix if it exists, otherwise use an empty dict
    replay_url_map = {}
    if hasattr(matrix, "replay_url_map") and isinstance(matrix.replay_url_map, dict):
        replay_url_map = matrix.replay_url_map

    replay_url_map_json = json.dumps(replay_url_map)

    # ---------------------------------------------------------------- HTML
    return f"""
<!-- enable pointer events on axis labels and cells -->
<style>
  .xaxislayer-above .xtick text,
  .yaxislayer-above .ytick text,
  .heatmap .nsewdrag {{
      pointer-events: all !important;
      cursor: pointer;
  }}
  .clickable-cell {{
      cursor: pointer;
  }}
</style>

<div class="heatmap-wrapper">
  <div id="{uid}"></div>
  
  {create_map_viewer_html(uid)}
</div>

<script>
(function() {{
  const fig = {fig_json};
  const el = document.getElementById("{uid}");
  const imgs = fig.layout.meta.eval_image_map;
  const shortNames = fig.layout.meta.short_names;
  const fullNames = fig.layout.meta.full_names;
  const policyRows = {policy_rows_json};
  const replayUrlMap = {replay_url_map_json};
  
  // Create mapping from short names to full paths
  const shortToFullPath = {{}};
  for (let i = 0; i < shortNames.length; i++) {{
    shortToFullPath[shortNames[i]] = fullNames[i];
  }}
  
  // Track double-click timing
  let lastClickTime = 0;
  const doubleClickThreshold = 300; // ms
  
  // Track if the mouse is over the heatmap
  let isMouseOverHeatmap = false;
  el.addEventListener('mouseenter', function() {{
    isMouseOverHeatmap = true;
  }});
  el.addEventListener('mouseleave', function() {{
    isMouseOverHeatmap = false;
    setTimeout(() => {{
      if (!isMouseOverHeatmap && !isMouseOverMap && !isViewLocked) {{
        hideMap();
      }}
    }}, 100);
  }});

  {get_map_viewer_js_functions(uid)}

  Plotly.newPlot(el, fig.data, fig.layout)
        .then(() => {{
          setTimeout(attachAxisHover, 500); // Ensure DOM is fully rendered
          attachHeatmapHover();
        }});

  /* ------------------------------------------------ axis‑label hover ----- */
  function attachAxisHover() {{
    // Get all x-axis tick labels
    const ticks = el.querySelectorAll(".xaxislayer-above .xtick text");
    
    // Enhanced event binding for axis labels
    ticks.forEach((tick, i) => {{
        if (i >= shortNames.length) return; // Safety check
        const shortName = shortNames[i];
        
        if (shortName.toLowerCase() === "overall") return;  // skip aggregate
        
        // Make sure these elements have proper cursor and pointer events
        tick.style.pointerEvents = "all";
        tick.style.cursor = "pointer";
        
        // Add multiple event handlers for redundancy
        tick.addEventListener('click', () => {{
            showMap(shortName);
        }});
        tick.addEventListener('mouseenter', () => {{
            showMap(shortName);
        }});
        tick.addEventListener('mouseover', () => {{
            showMap(shortName);
        }});
        
        // Add data attributes for easier debugging
        tick.setAttribute('data-short-name', shortName);
    }});
    
    // Add a click handler to the entire axis as a fallback
    const xAxis = el.querySelector('.xaxislayer-above');
    if (xAxis) {{
      xAxis.addEventListener('click', function(e) {{
        // Find the closest tick text element
        const target = e.target.closest('.xtick text');
        if (target && target.hasAttribute('data-short-name')) {{
          const shortName = target.getAttribute('data-short-name');
          showMap(shortName);
        }}
      }});
    }}
  }}

  /* ------------------------------------------------ heatmap cell hover --- */
  function attachHeatmapHover() {{
    el.on('plotly_hover', function(data) {{
      const pts = data.points[0];
      const shortName = pts.x;
      if (shortName.toLowerCase() === "overall") return;  // skip aggregate
      
      const yIndex = pts.pointIndex[0];
      
      // Skip the first two rows (Mean and Max)
      if (yIndex < 2) return;
      
      // Adjust index to account for Mean and Max rows
      const policyIndex = yIndex - 2;
      if (policyIndex >= policyRows.length) return;
      
      const policyName = policyRows[policyIndex];
      
      // Get the full path for the eval
      const fullPath = shortToFullPath[shortName] || shortName;
      
      // Use the specific replay URL from the map if available
      let replayUrl;
      const mapKey = `${{policyName}}|${{fullPath}}`;
      
      if (replayUrlMap && replayUrlMap[mapKey]) {{
        replayUrl = replayUrlMap[mapKey];
      }} else {{
        // Fallback to constructing a URL if not in the map
        replayUrl = `https://metta-ai.github.io/metta/?replayUrl=${{policyName}}/${{fullPath}}/replay.json.z`;
      }}
      
      // Show map with replay URL
      showMap(shortName, replayUrl);
    }});
    
    // Handle clicks on cells to toggle lock and detect double-clicks
    el.on('plotly_click', function(data) {{
      const now = new Date().getTime();
      const pts = data.points[0];
      const shortName = pts.x;
      const yIndex = pts.pointIndex[0];
      
      // Skip the first two rows (Mean and Max) and the Overall column
      if (yIndex < 2 || shortName.toLowerCase() === "overall") return;
      
      // First get the policy name
      const policyIndex = yIndex - 2;
      if (policyIndex >= policyRows.length) return;
      
      const policyName = policyRows[policyIndex];
      
      // Get full path for this eval name
      const fullPath = shortToFullPath[shortName] || shortName;
      
      // Get the replay URL from the map if available
      let replayUrl;
      const mapKey = `${{policyName}}|${{fullPath}}`;
      
      if (replayUrlMap && replayUrlMap[mapKey]) {{
        replayUrl = replayUrlMap[mapKey];
      }}
      
      // Handle single vs double click
      if (now - lastClickTime < doubleClickThreshold && replayUrl) {{
        // This is a double-click - open replay in new tab
        window.open(replayUrl, '_blank');
      }} else {{
        // This is a single click - toggle lock and update map
        toggleLock();
        
        // Force show the map for this cell regardless of lock state
        const wasLocked = isViewLocked;
        isViewLocked = false;
        showMap(shortName, replayUrl);
        isViewLocked = wasLocked;
      }}
      
      // Update last click time
      lastClickTime = now;
    }});
    
    // Only trigger unhover when we're sure we've left both the heatmap and the map viewer
    el.on('plotly_unhover', function() {{
      setTimeout(() => {{
        if (!isMouseOverHeatmap && !isMouseOverMap && !isViewLocked) {{
          hideMap();
        }}
      }}, 100);
    }});
    
    // Add visual indicators for clickable cells
    el.on('plotly_afterplot', function() {{
      const cells = el.querySelectorAll('.heatmap .nsewdrag');
      cells.forEach(cell => {{
        cell.classList.add('clickable-cell');
        cell.style.transition = 'opacity 0.2s';
        cell.title = "Click to lock view; double-click to open replay";
      }});
    }});
  }}
}})();
</script>
"""


def get_heatmap_matrix(
    stats_db: EvalStatsDB,
    metric: str,
    suite: str | None = None,
    num_output_policies: int | str = "all",
) -> pd.DataFrame:
    """
    Get matrix data for the specified metric from a StatsDB.

    Args:
        stats_db: EvalStatsDB instance
        metric: The metric to get data for
        num_output_policies: Optional number of policies to output
        suite: Optional suite name to filter evaluations

    Returns:
        DataFrame with policies as rows and evaluations as columns
    """
    logger = logging.getLogger(__name__)
    logger.info(f"Building heatmap matrix for metric {metric}")

    # Base SQL query using the policy_simulation_agent_aggregates view
    base_sql = """
    SELECT
        policy_key,
        policy_version,
        sim_env as eval_name,
        mean as value
    FROM policy_simulation_agent_aggregates
    WHERE metric = ?
    """
    params = [metric]
    # Add suite filtering if specified
    if suite is not None:
        logger.info(f"Adding suite filter for: {suite}")
        base_sql += " AND sim_suite = ?"
        params.append(suite)

    # Log the SQL query for debugging
    logger.info(f"Executing SQL query: {sql} with params {params}")

    # Execute the query
    df = stats_db.con.execute(sql, params).fetchdf()
    logger.info(f"Query returned {len(df)} rows")

    if len(df) == 0:
        logger.warning(f"No data found for metric {metric}")
        return pd.DataFrame()

    # Create the policy_uri column by combining policy_key and policy_version
    df["policy_uri"] = df["policy_key"] + ":" + df["policy_version"].astype(str)
    logger.info(f"Created policy_uri column. Sample: {df['policy_uri'].head(3).tolist() if len(df) > 0 else []}")

    # Get replay URLs for each (policy, eval_name) pair
    replay_url_map = _get_replay_urls_map(stats_db, df)
    logger.info(f"Retrieved {len(replay_url_map)} replay URLs")

    # Process data into matrix format
    policies = df["policy_uri"].unique()
    eval_names = df["eval_name"].unique()
    logger.info(f"Found {len(policies)} unique policies and {len(eval_names)} unique evaluation names")

    # Create a dictionary to map (policy_uri, eval_name) to value
    data_map = {}
    for _, row in df.iterrows():
        data_map[(row["policy_uri"], row["eval_name"])] = row["value"]

    # Calculate overall scores for each policy
    overall_scores = {}
    for policy in policies:
        policy_df = df[df["policy_uri"] == policy]
        overall_scores[policy] = policy_df["value"].mean()

    # Create the matrix data
    matrix_data = []
    for policy in policies:
        row_data = {"policy_uri": policy, "Overall": overall_scores[policy]}
        for eval_name in eval_names:
            if (policy, eval_name) in data_map:
                row_data[eval_name] = data_map[(policy, eval_name)]
        matrix_data.append(row_data)

    # Convert to DataFrame and set index
    matrix = pd.DataFrame(matrix_data)
    if len(matrix) > 0:
        matrix = matrix.set_index("policy_uri")

        # Always sort by overall score (lowest first)
        sorted_policies = sorted(policies, key=lambda p: overall_scores[p])
        matrix = matrix.reindex(sorted_policies)

        # Limit the number of policies
        if num_output_policies != "all" and isinstance(num_output_policies, int):
            matrix = matrix.tail(num_output_policies)

        # Attach the replay URL map as an attribute on the DataFrame
        matrix.replay_url_map = replay_url_map

    logger.info(f"Final matrix shape: {matrix.shape}")
    return matrix


def _get_replay_urls_map(stats_db: EvalStatsDB, data_df: pd.DataFrame) -> Dict[str, str]:
    """
    Get replay URLs for each (policy, eval_name) combination in the data frame.

    Returns a dictionary mapping "policy_uri|eval_name" to the replay URL.
    """
    if data_df.empty:
        return {}

    logger = logging.getLogger(__name__)

    # Get unique (policy_key, policy_version) combinations
    policy_pairs = data_df[["policy_key", "policy_version"]].drop_duplicates().values.tolist()

    # Map to store replay URLs: "policy_uri|eval_name" -> replay_url
    replay_url_map = {}

    for policy_key, policy_version in policy_pairs:
        # Get all eval_names for this policy
        eval_names = data_df[(data_df["policy_key"] == policy_key) & (data_df["policy_version"] == policy_version)][
            "eval_name"
        ].unique()

        # Get replay URLs for this policy
        for eval_name in eval_names:
            # Skip "Overall" as it's not a real evaluation
            if eval_name == "Overall":
                continue

            # Get replay URLs for this specific environment
            replay_urls = stats_db.get_replay_urls(policy_key=policy_key, policy_version=policy_version, env=eval_name)

            if replay_urls:
                # Use the first URL for this combination
                key = f"{policy_key}|{policy_version}|{eval_name}"
                replay_url_map[key] = replay_urls[0]
                logger.debug(f"Found replay URL for {key}: {replay_urls[0]}")
            else:
                logger.debug(f"No replay URLs found for policy {policy_uri}, env {eval_name}")

    return replay_url_map
