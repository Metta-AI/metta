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
import uuid
from typing import List, Tuple

import pandas as pd
import plotly.graph_objects as go
from plotly.utils import PlotlyJSONEncoder

from metta.eval.mapviewer import create_map_viewer_html, get_map_viewer_js_functions


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
    replay_base_url: str = "https://softmax-public.s3.us-east-1.amazonaws.com/replays/evals",
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
  const replayBaseUrl = "{replay_base_url}";
  
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
      
      // Get full path for replay URL
      const fullPath = shortToFullPath[shortName] || shortName;
      
      // Construct the replay URL
      const replayUrl = `https://metta-ai.github.io/metta/?replayUrl=${{replayBaseUrl}}/${{policyName}}/${{fullPath}}/replay.json.z`;
      
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
      
      // Get full path for this eval name (for replay URL)
      const fullPath = shortToFullPath[shortName] || shortName;
      
      // Construct the replay URL
      const replayUrl = `https://metta-ai.github.io/metta/?replayUrl=${{replayBaseUrl}}/${{policyName}}/${{fullPath}}/replay.json.z`;
      
      // Handle single vs double click
      if (now - lastClickTime < doubleClickThreshold) {{
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
