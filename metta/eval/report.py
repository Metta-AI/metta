"""
High‑level report generator.

At the moment we produce a single heat‑map, but the structure anticipates
multiple chart types – simply append more HTML snippets to `graphs_html`.
"""

from __future__ import annotations

import logging
import os
import shutil
import tempfile
from typing import List

import boto3
import hydra
from botocore.exceptions import NoCredentialsError
from omegaconf import DictConfig

from metta.eval.db import PolicyEvalDB
from metta.eval.heatmap import create_heatmap_html_snippet
from metta.eval.mapviewer import MAP_VIEWER_CSS
from metta.sim.eval_stats_db import EvalStatsDB
from metta.util.wandb.wandb_context import WandbContext

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
# S3 util
# --------------------------------------------------------------------------- #
def _upload_to_s3(html: str, s3_path: str):
    if not s3_path.startswith("s3://"):
        raise ValueError("S3 path must start with s3://")

    bucket, key = s3_path[5:].split("/", 1)
    try:
        boto3.client("s3").put_object(Body=html, Bucket=bucket, Key=key, ContentType="text/html")
    except NoCredentialsError as e:
        logger.error("AWS credentials not found; run setup_sso.py")
        raise e


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


def generate_report_html(cfg: DictConfig) -> str:
    metric = cfg.analyzer.metric
    view_type = cfg.analyzer.view_type
    policy_uri = cfg.analyzer.policy_uri

    tmp_dir = tempfile.mkdtemp()
    db_path = os.path.join(tmp_dir, "policy_metrics.sqlite")
    logger.info("Working db path: %s", db_path)

    try:
        db = PolicyEvalDB(db_path)
        db.import_from_eval_stats(cfg)

        matrix = db.get_matrix_data(metric, view_type=view_type, policy_uri=policy_uri)
        if matrix.empty:
            return "<html><body><h1>No data available</h1></body></html>"

        # create heat‑map snippet
        heatmap_html = create_heatmap_html_snippet(
            matrix,
            metric,
            height=600,
            width=900,
        )

        title = f"Policy Evaluation Report: {metric}"
        if view_type == "policy_versions" and policy_uri:
            title += f" – All versions of {policy_uri}"

        return _assemble_page(title, [heatmap_html])

    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


def generate_report(cfg: DictConfig):
    html_content = generate_report_html(cfg)
    output_path = cfg.analyzer.output_path

    # handle per‑policy filename tweak
    if cfg.analyzer.view_type == "policy_versions" and cfg.analyzer.policy_uri:
        base, ext = os.path.splitext(output_path)
        safe_name = cfg.analyzer.policy_uri.split("/")[-1].replace(":", "_")
        output_path = f"{base}_{safe_name}{ext}"

    if output_path.startswith("s3://"):
        _upload_to_s3(html_content, output_path)
        logger.info("Report uploaded to %s", output_path)
    else:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, "w") as fh:
            fh.write(html_content)
        logger.info("Report written to %s", output_path)

    return html_content, output_path


def dump_stats(cfg: DictConfig):
    logger.info(f"Importing data from {cfg.eval.eval_db_uri}")
    with WandbContext(cfg) as wandb_run:
        eval_stats_db = EvalStatsDB.from_uri(cfg.eval.eval_db_uri, cfg.run_dir, wandb_run)

    analyzer = hydra.utils.instantiate(cfg.analyzer, eval_stats_db)
    dfs, _ = analyzer.analyze(include_policy_fitness=False)
    for df in dfs:
        print(df)
