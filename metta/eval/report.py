"""
High‑level report generator.

At the moment we produce a single heat‑map, but the structure anticipates
multiple chart types – simply append more HTML snippets to `graphs_html`.
"""

from __future__ import annotations

import logging
import os
import shutil
from typing import List, Literal

from omegaconf import DictConfig

from metta.eval.analysis_config import AnalyzerConfig
from metta.eval.heatmap import create_heatmap_html_snippet
from metta.eval.mapviewer import MAP_VIEWER_CSS
from metta.sim.stats_db import StatsDB
from metta.util.file import local_copy, write_data
from metta.util.wandb.wandb_context import WandbContext

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


def generate_report_html(analyzer_cfg: AnalyzerConfig, omegaconf_cfg: DictConfig) -> str:
    metric = analyzer_cfg.metric
    view_type = analyzer_cfg.view_type
    policy_uri = analyzer_cfg.policy_uri

    num_output_policies: int | Literal["all"] = analyzer_cfg.num_output_policies

    with local_copy(analyzer_cfg.eval_db_uri) as db_path:
        db = StatsDB(db_path)
        db.materialize_policy_simulations_view(metric)
        matrix = db.policy_simulations_view(
            metric, view_type=view_type, policy_uri=policy_uri, num_output_policies=num_output_policies
        )

    if matrix.empty:
        return "<html><body><h1>No data available</h1></body></html>"

    heatmap_html = create_heatmap_html_snippet(
        matrix,
        metric,
        replay_base_url="https://softmax-public.s3.us-east-1.amazonaws.com/replays/evals",
        height=600,
        width=900,
    )

    title = f"Policy Evaluation Report: {metric}"
    if view_type == "policy_versions" and policy_uri:
        title += f" – All versions of {policy_uri}"

    return _assemble_page(title, [heatmap_html])


def generate_report(analyzer_cfg: AnalyzerConfig, omegaconf_cfg: DictConfig):
    html_content = generate_report_html(analyzer_cfg, omegaconf_cfg)
    output_path = analyzer_cfg.output_path

    # Handle per-policy filename tweak
    if analyzer_cfg.view_type == "policy_versions" and analyzer_cfg.policy_uri:
        base, ext = os.path.splitext(output_path)
        safe_name = analyzer_cfg.policy_uri.split("/")[-1].replace(":", "_")
        output_path = f"{base}_{safe_name}{ext}"

    write_data(output_path, html_content, content_type="text/html")

    return html_content, output_path


def dump_stats(analyzer_cfg: AnalyzerConfig, omegaconf_cfg: DictConfig):  # TODO: remove omegaconf_cfg
    assert analyzer_cfg.eval_db_uri is not None, "eval_db_uri is required"

    logger.info(f"Importing data from {analyzer_cfg.eval_db_uri}")
    with WandbContext(omegaconf_cfg) as wandb_run:
        eval_stats_db = EvalStatsDB.from_uri(analyzer_cfg.eval_db_uri, omegaconf_cfg.run_dir, wandb_run)

    analyzer = EvalStatsAnalyzer(eval_stats_db, analyzer_cfg.analysis, analyzer_cfg.policy_uri)
    dfs, _ = analyzer.analyze(include_policy_fitness=False)
    for df in dfs:
        print(df)
