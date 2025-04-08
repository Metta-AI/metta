"""
Generate reports from policy evaluation metrics.
"""

import os
import logging
from typing import Optional
from omegaconf import DictConfig
from eval.db import PolicyEvalDB
from eval.heatmap import create_matrix_visualization, save_heatmap_to_html

logger = logging.getLogger(__name__)

def generate_report(
    cfg: DictConfig,
    output_dir: str = ".",
    metric: str = "episode_reward",
    db_path: Optional[str] = None,
) -> str:
    output_dir = os.path.abspath(output_dir)
    os.makedirs(output_dir, exist_ok=True)
    
    # Use temporary database if not specified
    if db_path is None:
        db_path = os.path.join(output_dir, "policy_metrics.sqlite")
    
    # Initialize database and import data
    logger.info(f"Initializing database at {db_path}")
    db = PolicyEvalDB(db_path)
    
    db.import_from_eval_stats(cfg)
    
    # Get matrix data for visualization
    logger.info(f"Generating matrix visualization for metric: {metric}")
    matrix_data = db.get_matrix_data(metric)
    
    if matrix_data.empty:
        logger.warning(f"No data found for metric: {metric}")
        return None
    
    # Create visualization
    fig = create_matrix_visualization(
        matrix_data=matrix_data,
            metric="episode_reward",
        colorscale='RdYlGn',
        score_range=(0, 3),  # Typical range for episode_reward
        height=600,
        width=900
    )
    
    # Save to HTML
    report_name = f"policy_evaluation.html"
    report_path = os.path.join(output_dir, report_name)
    
    logger.info(f"Saving report to {report_path}")
    save_heatmap_to_html(
        fig=fig,
        output_path=report_path,
        title=f"Policy Evaluation: {metric}"
    )
    
    return report_path