"""
High‑level report generator.

At the moment we produce a single heat‑map, but the structure anticipates
multiple chart types – simply append more HTML snippets to `graphs_html`.
"""

from __future__ import annotations

import logging
from typing import List, Literal

from metta.eval.dashboard.dashboard_config import DashboardConfig
from metta.eval.dashboard.heatmap import create_heatmap_html_snippet, get_heatmap_matrix
from metta.eval.dashboard.mapviewer import MAP_VIEWER_CSS
from metta.eval.eval_stats_db import EvalStatsDB
from metta.util.file import write_data

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
# report generator
# --------------------------------------------------------------------------- #
_BODY_CSS = """
body{
  font-family:Arial, sans-serif;margin:0;padding:20px;background:#f8f9fa;
}
.container{
  max-width:1200px;margin:0 auto;background:#fff;padding:20px;border-radius:5px;
  box-shadow:0 2px 4px rgba(0,0,0,.1);
}
h1{color:#333;border-bottom:1px solid #ddd;padding-bottom:10px}
"""


def _assemble_page(title: str, graphs: List[str]) -> str:
    graphs_html = "\n".join(graphs)
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <title>{title}</title>
  <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
  <style>{_BODY_CSS}{MAP_VIEWER_CSS}</style>
</head>
<body>
  <div class="container">
    <h1>{title}</h1>
    {graphs_html}
  </div>
</body>
</html>"""


def generate_dashboard_html(db: EvalStatsDB, dashboard_cfg: DashboardConfig) -> str:
    logger = logging.getLogger(__name__)

    logger.info(f"Analyzer config: {dashboard_cfg}")

    metric = dashboard_cfg.metric
    num_output_policies: int | Literal["all"] = dashboard_cfg.num_output_policies
    suite = dashboard_cfg.suite

    matrix = get_heatmap_matrix(db, metric, suite, num_output_policies)

    if matrix.empty:
        return "<html><body><h1>No data available</h1></body></html>"

    heatmap_html = create_heatmap_html_snippet(
        matrix,
        metric,
        height=600,
        width=900,
    )

    title = f"Policy Evaluation Report: {metric}"
    return _assemble_page(title, [heatmap_html])


def generate_dashboard(dashboard_cfg: DashboardConfig):
    with EvalStatsDB.from_uri(dashboard_cfg.eval_db_uri) as db:
        html_content = generate_dashboard_html(db, dashboard_cfg)
    output_path = dashboard_cfg.output_path
    write_data(output_path, html_content, content_type="text/html")

    return html_content, output_path
