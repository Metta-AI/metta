"""
Generate a self‑contained HTML *snippet* with an interactive policy‑evaluation
heat‑map.  The snippet relies on two global resources that the caller (e.g.
`report.py`) must include exactly once per page:

    <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
    <style> … the .popover styles shown in report.py … </style>

Only one public helper is exposed:

    create_heatmap_html_snippet(matrix: pd.DataFrame, metric: str, ...)

It returns an HTML string ready to be embedded in a larger report.
"""

from __future__ import annotations

import json
import uuid
from typing import List, Tuple

import pandas as pd
import plotly.graph_objects as go
from plotly.utils import PlotlyJSONEncoder


# --------------------------------------------------------------------------- #
# internal helpers
# --------------------------------------------------------------------------- #
def _format_metric(metric: str) -> str:
    return metric.replace("_", " ").capitalize()


def _wandb_url(uri: str, entity: str = "metta-research", project: str = "metta") -> str:
    if uri.startswith("wandb://run/"):
        uri = uri[len("wandb://run/") :]
    if ":v" in uri:
        uri = uri.split(":v")[0]
    return f"https://wandb.ai/{entity}/{project}/runs/{uri}"


def _build_plotly_figure(
    matrix: pd.DataFrame,
    metric: str,
    colorscale: List[Tuple[float, str]],
    score_range: Tuple[float, float],
    *,
    height: int,
    width: int,
) -> go.Figure:
    # ----------------------------------------------- rows / columns / values
    eval_names: List[str] = matrix.columns.tolist()
    policy_rows: List[str] = (
        matrix.pop("policy_uri").tolist() if "policy_uri" in matrix.columns else matrix.index.tolist()
    )

    z = [matrix.mean().tolist(), matrix.max().tolist()] + matrix.values.tolist()
    y_labels = ["Mean", "Max"] + policy_rows

    fig = go.Figure(
        go.Heatmap(
            z=z,
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
    )

    # draw dashed box around aggregates
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

    # store eval‑>image map in meta
    img_base = "https://softmax-public.s3.amazonaws.com/policydash/evals/img"
    fig.update_layout(meta=dict(eval_image_map={e: f"{img_base}/{e.lower()}.png" for e in eval_names}))

    return fig


# --------------------------------------------------------------------------- #
# public API
# --------------------------------------------------------------------------- #
def create_heatmap_html_snippet(
    matrix: pd.DataFrame,
    metric: str,
    *,
    height: int = 600,
    width: int = 900,
) -> str:
    """
    Return an HTML fragment (``<div>…</div><script>…</script>``) that renders
    an interactive heat‑map with image tool‑tips.

    The caller is responsible for injecting Plotly’s JS bundle and the shared
    `.popover` CSS *once* per page.
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
        # Green
        [1.0, "rgb(20, 230, 80)"],
    ]

    fig = _build_plotly_figure(
        matrix,
        metric,
        colorscale=colorscale,
        score_range=score_range,
        height=height,
        width=width,
    )

    uid = f"heatmap_{uuid.uuid4().hex[:8]}"
    p_id = f"popover_{uid}"
    t_id = f"popover_title_{uid}"
    i_id = f"popover_img_{uid}"

    fig_json = json.dumps(fig, cls=PlotlyJSONEncoder)

    # ---- HTML fragment ----------------------------------------------------
    return f"""
<div class="heatmap-wrapper">
  <div id="{uid}"></div>
  <div class="popover" id="{p_id}">
      <div class="popover-title" id="{t_id}"></div>
      <img class="popover-img" id="{i_id}" alt="Evaluation map">
  </div>
</div>

<script>
(function() {{
  const fig      = {fig_json};
  const el       = document.getElementById("{uid}");
  const pop      = document.getElementById("{p_id}");
  const popTitle = document.getElementById("{t_id}");
  const popImg   = document.getElementById("{i_id}");
  const imgMap   = fig.layout.meta.eval_image_map;

  let lastMouse = {{clientX:0, clientY:0}};
  window.addEventListener("mousemove", e => (lastMouse = e));

  Plotly.newPlot(el, fig.data, fig.layout).then(() => attachHover());

  function attachHover() {{
    Plotly.d3.selectAll(el.querySelectorAll(".xtick text"))
      .on("mouseenter", function(_, i) {{
          const name = this.textContent.trim();
          show(name, d3.event);
      }})
      .on("mouseleave", hide);

    el.on("plotly_hover", d => {{
      if (!d.points?.length) return;
      show(d.points[0].x, d.event || lastMouse);
    }});
    el.on("plotly_unhover", hide);
  }}

  function show(name, ev) {{
    const img = imgMap[name] || "";
    popTitle.textContent = name;
    popImg.src = img;
    popImg.style.display = img ? "block" : "none";

    const x = Math.min(ev.clientX + 15, innerWidth  - pop.offsetWidth  - 10);
    const y = Math.min(ev.clientY + 15, innerHeight - pop.offsetHeight - 10);
    pop.style.left = x + "px";
    pop.style.top  = y + "px";
    pop.style.opacity = "1";
  }}
  function hide() {{ pop.style.opacity = "0"; }}
}})();
</script>
"""
