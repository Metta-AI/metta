"""
Return an embeddable HTML snippet that shows a policy‑evaluation heat‑map.

Features
--------
* Native Plotly hover box on every cell contains the evaluation‑map picture.
* An extra pop‑over appears when you hover (or click) an x‑axis label.
* The 'Overall' column is excluded from both image behaviours.
* All IDs are namespaced; multiple snippets can coexist in the same page.

External boiler‑plate (include once per page)
---------------------------------------------
    <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
    <style>
        .popover{position:fixed;z-index:1000;background:#fff;border:1px solid #ddd;
                 border-radius:5px;padding:10px;box-shadow:0 2px 8px rgba(0,0,0,.3);
                 pointer-events:none;opacity:0;transition:opacity .2s}
        .popover-title{font-weight:bold;text-align:center;margin-bottom:8px;
                       border-bottom:1px solid #eee;padding-bottom:5px}
        .popover-img{max-width:100%;max-height:250px;display:block;margin:0 auto}
    </style>
"""

from __future__ import annotations

import json
import uuid
from typing import List, Tuple

import pandas as pd
import plotly.graph_objects as go
from plotly.utils import PlotlyJSONEncoder


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
            hovertemplate=(
                "<b>Policy:</b> %{y}"
                "<br><b>Evaluation:</b> %{x}"
                "<br><b>Score:</b> %{z:.2f}"
                "<br><img src='%{customdata}' "
                "style='width:140px; margin-top:6px;'>"
                "<extra></extra>"
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
# public API                                                                  #
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

    Caller must embed Plotly + the `.popover` CSS (shown in module docstring).
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

    fig = _build_figure(matrix, metric, colorscale=colorscale, score_range=score_range, height=height, width=width)
    uid = f"heat_{uuid.uuid4().hex[:8]}"
    pop_id = f"{uid}_pop"
    tit_id = f"{uid}_title"
    img_id = f"{uid}_img"
    fig_json = json.dumps(fig, cls=PlotlyJSONEncoder)

    # ---------------------------------------------------------------- HTML
    return f"""
<div class="heatmap-wrapper">
  <div id="{uid}"></div>
  <div class="popover" id="{pop_id}">
      <div class="popover-title" id="{tit_id}"></div>
      <img class="popover-img" id="{img_id}" alt="Evaluation map">
  </div>
</div>

<script>
(function() {{
  const fig   = {fig_json};
  const el    = document.getElementById("{uid}");
  const pop   = document.getElementById("{pop_id}");
  const pT    = document.getElementById("{tit_id}");
  const pImg  = document.getElementById("{img_id}");
  const imgs  = fig.layout.meta.eval_image_map;

  let lastMouse = {{clientX:0, clientY:0}};
  window.addEventListener("mousemove", e => (lastMouse = e));

  Plotly.newPlot(el, fig.data, fig.layout).then(() => attachAxisHover());

  /* ------------------------------------------------ axis‑label hover ----- */
  function attachAxisHover() {{
    const ticks = el.querySelectorAll(".xtick text");
    ticks.forEach((tick, i) => {{
        const evalName = fig.data[0].x[i];
        if (evalName === "Overall") return;       // skip
        tick.style.cursor = "pointer";
        tick.onmouseenter = ev => show(evalName, ev);
        tick.onmouseleave = hide;
        tick.onclick      = ev => show(evalName, ev);
    }});
  }}

  /* ------------------------------------------------ cell hover ----------- */
  el.on("plotly_hover", d => {{
      if (!d.points?.length) return;
      const evalName = d.points[0].x;
      if (evalName === "Overall") return;
      show(evalName, d.event || lastMouse);
  }});
  el.on("plotly_unhover", hide);

  /* ------------------------------------------------ helpers -------------- */
  function show(name, ev) {{
    const url = imgs[name] || "";
    pT.textContent = name;
    pImg.src       = url;
    pImg.style.display = url ? "block" : "none";

    const x = Math.min(ev.clientX+15, innerWidth  - pop.offsetWidth  - 10);
    const y = Math.min(ev.clientY+15, innerHeight - pop.offsetHeight - 10);
    pop.style.left = x + "px";
    pop.style.top  = y + "px";
    pop.style.opacity = "1";
  }}
  function hide() {{ pop.style.opacity = "0"; }}
}})();
</script>
"""
