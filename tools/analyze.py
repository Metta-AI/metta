#!/usr/bin/env -S uv run

"""Analysis tool for MettaGrid evaluation results using argparse."""

import argparse
import logging
import sys
import yaml
from pathlib import Path
from typing import Any

from pydantic import Field

from metta.common.util.typed_config import BaseModelWithForbidExtra
from metta.eval.analysis import analyze
from metta.eval.analysis_config import AnalysisConfig
from tools.utils import get_policy_store_from_cfg


logger = logging.getLogger(__name__)


class AnalyzeConfig(BaseModelWithForbidExtra):
    """Configuration for the analyze script."""
    
    # Run configuration
    run: str = Field(description="Run name/identifier")
    
    # Analysis configuration
    analysis: AnalysisConfig = Field(description="Analysis configuration")
    
    # Other configurations
    device: str = Field(default="cuda", description="Device to use")
    wandb: dict[str, Any] = Field(default_factory=dict, description="WandB configuration")


def load_config_from_yaml(config_path: str) -> AnalyzeConfig:
    """Load configuration from YAML file."""
    config_path = Path(config_path)
    
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    with open(config_path, 'r') as f:
        config_data = yaml.safe_load(f)
    
    return AnalyzeConfig.model_validate(config_data)


def main():
    """Main analysis function."""
    parser = argparse.ArgumentParser(description="Analyze Metta AI evaluation results")
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to YAML configuration file"
    )
    parser.add_argument(
        "--policy-uri",
        type=str,
        default=None,
        help="Override policy URI"
    )
    parser.add_argument(
        "--eval-db-uri",
        type=str,
        default=None,
        help="Override evaluation database URI"
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Override device (cuda/cpu)"
    )
    
    args = parser.parse_args()
    
    # Load configuration from file
    cfg = load_config_from_yaml(args.config)
    
    # Apply command line overrides
    if args.policy_uri:
        cfg.analysis.policy_uri = args.policy_uri
        
    if args.eval_db_uri:
        cfg.analysis.eval_db_uri = args.eval_db_uri
        
    if args.device:
        cfg.device = args.device
    
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    logger.info(f"Starting analysis with config: {cfg.run}")
    logger.info(f"Policy URI: {cfg.analysis.policy_uri}")
    logger.info(f"Eval DB URI: {cfg.analysis.eval_db_uri}")
    
    try:
        # Get policy store
        config_dict = {
            "device": cfg.device,
            **cfg.model_dump()
        }
        policy_store = get_policy_store_from_cfg(config_dict)
        
        # Get policy record
        policy_pr = policy_store.policy_record(
            cfg.analysis.policy_uri, 
            cfg.analysis.policy_selector.type, 
            metric=cfg.analysis.policy_selector.metric
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