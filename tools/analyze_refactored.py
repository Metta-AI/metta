#!/usr/bin/env -S uv run

"""Analysis tool for MettaGrid evaluation results - Refactored example."""

import argparse
import logging
import sys
from typing import Optional

from pydantic import Field

from metta.common.util.typed_config import ConfigWithBuilder
from metta.common.wandb.wandb_config import WandbConfig
from metta.eval.analysis import analyze
from metta.eval.analysis_config import AnalysisConfig
from tools.utils import get_policy_store_from_cfg

logger = logging.getLogger(__name__)


class AnalyzeToolConfig(ConfigWithBuilder):
    """Configuration for the analyze tool - ONLY contains other configs."""

    # Config objects only - no runtime data
    analysis: AnalysisConfig = Field(description="Analysis configuration")
    wandb: Optional[WandbConfig] = Field(default=None, description="WandB configuration")


def main():
    """Main analysis function."""
    parser = argparse.ArgumentParser(description="Analyze Metta AI evaluation results")

    # Configuration file (contains only Config objects)
    parser.add_argument("--config", type=str, required=True, help="Path to YAML configuration file")

    # Runtime parameters (not stored in config file)
    parser.add_argument("--run", type=str, required=True, help="Run name/identifier")
    parser.add_argument("--policy-uri", type=str, required=True, help="Policy URI to analyze")
    parser.add_argument("--eval-db-uri", type=str, required=True, help="Evaluation database URI")
    parser.add_argument("--device", type=str, default="cuda", help="Device to use (cuda/cpu)")

    # Runtime overrides for specific config fields
    parser.add_argument(
        "--policy-selector-type", type=str, default=None, help="Override policy selector type (best/latest/specific)"
    )
    parser.add_argument("--policy-selector-metric", type=str, default=None, help="Override policy selector metric")

    args = parser.parse_args()

    # Load configuration from file (only Config objects)
    cfg = AnalyzeToolConfig.from_file(args.config)

    # Apply runtime overrides to config objects
    if args.policy_selector_type:
        cfg.analysis.policy_selector.type = args.policy_selector_type

    if args.policy_selector_metric:
        cfg.analysis.policy_selector.metric = args.policy_selector_metric

    # Set runtime values on analysis config
    # Note: In a cleaner design, these would be passed separately to analyze()
    cfg.analysis.policy_uri = args.policy_uri
    cfg.analysis.eval_db_uri = args.eval_db_uri

    # Setup logging
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")

    logger.info(f"Starting analysis with run: {args.run}")
    logger.info(f"Policy URI: {args.policy_uri}")
    logger.info(f"Eval DB URI: {args.eval_db_uri}")
    logger.info(f"Device: {args.device}")
    logger.info(f"Config loaded from: {args.config}")

    try:
        # Get policy store - create runtime context dict
        runtime_context = {
            "device": args.device,
            "wandb": cfg.wandb.model_dump() if cfg.wandb else None,
        }

        policy_store = get_policy_store_from_cfg(runtime_context)

        # Get policy record
        policy_pr = policy_store.policy_record(
            args.policy_uri, cfg.analysis.policy_selector.type, metric=cfg.analysis.policy_selector.metric
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
    sys.exit(main())
