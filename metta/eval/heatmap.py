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

    z = [matrix.mean().tolist(), matrix.max().tolist()] + matrix.values.tolist()
    y_labels = ["Mean", "Max"] + policy_rows

    # ---------------------------------------------------------------- images
    img_base = "https://softmax-public.s3.amazonaws.com/policydash/evals/img"
    img_map = {e: f"{img_base}/{e.lower()}.png" for e in eval_names}
    img_map["Overall"] = ""  # no tooltip image

    # customdata for each cell (same for every row of a given column)
    customdata = [[img_map[ev] for ev in eval_names] for _ in y_labels]

    # ---------------------------------------------------------------- chart
    fig = go.Figure(
        go.Heatmap(
            z=z,
            customdata=customdata,
            x=eval_names,
            y=y_labels,
            colorscale=colorscale,
            zmin=score_range[0],
            zmax=score_range[1],
            colorbar=dict(title="Score"),
            hovertemplate=("<b>Policy:</b> %{y}<br><b>Evaluation:</b> %{x}<br><b>Score:</b> %{z:.2f}<extra></extra>"),
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
        meta=dict(eval_image_map=img_map),
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
    eval_names_json = json.dumps(matrix.columns.tolist())

    # ---------------------------------------------------------------- HTML
    return f"""
<!-- enable pointer events on axis labels and cells -->
<style>
  .xaxislayer-above .xtick text,
  .yaxislayer-above .ytick text {{
      pointer-events: all !important;
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
  const evalNames = {eval_names_json};
  
  // Track if the mouse is over the heatmap
  let isMouseOverHeatmap = false;
  el.addEventListener('mouseenter', function() {{
    isMouseOverHeatmap = true;
  }});
  el.addEventListener('mouseleave', function() {{
    isMouseOverHeatmap = false;
    setTimeout(() => {{
      if (!isMouseOverHeatmap && !isMouseOverMap) {{
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
        if (i >= evalNames.length) return; // Safety check
        const evalName = evalNames[i];
        if (evalName.toLowerCase() === "overall") return;  // skip aggregate
        
        // Make sure these elements have proper cursor and pointer events
        tick.style.pointerEvents = "all";
        tick.style.cursor = "pointer";
        
        // Add multiple event handlers for redundancy
        tick.addEventListener('click', () => showMap(evalName));
        tick.addEventListener('mouseenter', () => showMap(evalName));
        tick.addEventListener('mouseover', () => showMap(evalName));
        
        // Add a data attribute for easier debugging
        tick.setAttribute('data-eval-name', evalName);
    }});
    
    // Add a click handler to the entire axis as a fallback
    const xAxis = el.querySelector('.xaxislayer-above');
    if (xAxis) {{
      xAxis.addEventListener('click', function(e) {{
        // Find the closest tick text element
        const target = e.target.closest('.xtick text');
        if (target && target.hasAttribute('data-eval-name')) {{
          showMap(target.getAttribute('data-eval-name'));
        }}
      }});
    }}
  }}

  /* ------------------------------------------------ heatmap cell hover --- */
  function attachHeatmapHover() {{
    el.on('plotly_hover', function(data) {{
      const pts = data.points[0];
      const evalName = pts.x;
      if (evalName.toLowerCase() === "overall") return;  // skip aggregate
      showMap(evalName);
    }});
    
    // Only trigger unhover when we're sure we've left both the heatmap and the map viewer
    el.on('plotly_unhover', function() {{
      setTimeout(() => {{
        if (!isMouseOverHeatmap && !isMouseOverMap) {{
          hideMap();
        }}
      }}, 100);
    }});
  }}
}})();
</script>
"""
