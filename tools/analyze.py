#!/usr/bin/env -S uv run

"""Analysis tool for MettaGrid evaluation results using typer."""

import logging
from typing import Optional

import typer
from pydantic import Field

from metta.common.util.typed_config import ConfigWithBuilder
from metta.common.wandb.wandb_config import WandbConfig
from metta.eval.analysis import analyze
from metta.eval.analysis_config import AnalysisConfig
from tools.utils import get_policy_store_from_cfg

logger = logging.getLogger(__name__)
app = typer.Typer()


class AnalyzeConfig(ConfigWithBuilder):
    """Configuration for the analyze script."""

    # Run configuration
    run: str = Field(description="Run name/identifier")

    # Analysis configuration
    analysis: AnalysisConfig = Field(description="Analysis configuration")

    # WandB configuration (optional)
    wandb: WandbConfig | None = Field(default=None, description="WandB configuration")

    # Device configuration
    device: str = Field(default="cuda", description="Device to use (cuda/cpu)")


@app.command()
def main(
    config: str = typer.Option(..., "--config", help="Path to YAML configuration file"),
    policy_uri: Optional[str] = typer.Option(None, "--policy-uri", help="Override policy URI from config"),
    eval_db_uri: Optional[str] = typer.Option(
        None, "--eval-db-uri", help="Override evaluation database URI from config"
    ),
    device: str = typer.Option("cuda", "--device", help="Device to use (cuda/cpu)"),
):
    """Analyze Metta AI evaluation results."""
    # Load configuration from file
    cfg = AnalyzeConfig.from_file(config)

    # Apply command line overrides
    if policy_uri:
        cfg.analysis.policy_uri = policy_uri
    if eval_db_uri:
        cfg.analysis.eval_db_uri = eval_db_uri
    cfg.device = device

    # Setup logging
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")

    logger.info(f"Starting analysis with run: {cfg.run}")
    logger.info(f"Policy URI: {cfg.analysis.policy_uri}")
    logger.info(f"Eval DB URI: {cfg.analysis.eval_db_uri}")
    logger.info(f"Device: {cfg.device}")

    try:
        # Get policy store - create config dict with device and wandb settings
        config_dict = {"device": cfg.device}
        if cfg.wandb:
            config_dict["wandb"] = cfg.wandb.model_dump()

        # Add analysis config to the dict
        config_dict["analysis"] = cfg.analysis.model_dump()

        policy_store = get_policy_store_from_cfg(config_dict)

        # Get policy record
        policy_pr = policy_store.policy_record(
            cfg.analysis.policy_uri,
            cfg.analysis.policy_selector.type,
            metric=cfg.analysis.policy_selector.metric,
        )

        # Run analysis
        logger.info("Running analysis...")
        analyze(policy_pr, cfg.analysis)

        logger.info("Analysis completed successfully")

    except Exception as e:
        logger.error(f"Analysis failed with error: {e}")
        raise

    return 0


if __name__ == "__main__":
    app()
