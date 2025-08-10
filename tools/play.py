#!/usr/bin/env -S uv run

"""Interactive play script for Metta AI using argparse instead of Hydra."""

import argparse
import logging
import sys
import yaml
from pathlib import Path
from typing import Any

from pydantic import Field

import mettascope.server as server
from metta.common.util.constants import DEV_METTASCOPE_FRONTEND_URL
from metta.common.util.typed_config import BaseModelWithForbidExtra


logger = logging.getLogger(__name__)


class ReplayJobConfig(BaseModelWithForbidExtra):
    """Configuration for replay/play job."""
    
    open_browser_on_start: bool = Field(default=True, description="Open browser automatically")
    policy_uri: str | None = Field(default=None, description="Policy URI to load")
    env: str | None = Field(default=None, description="Environment config path")
    device: str = Field(default="cpu", description="Device to use")
    

class PlayConfig(BaseModelWithForbidExtra):
    """Configuration for the play script."""
    
    # Run configuration
    run: str = Field(description="Run name/identifier")
    
    # Replay job configuration
    replay_job: ReplayJobConfig = Field(default_factory=ReplayJobConfig)
    
    # WandB config
    wandb: dict[str, Any] = Field(default_factory=dict, description="WandB configuration")


def load_config_from_yaml(config_path: str) -> PlayConfig:
    """Load configuration from YAML file."""
    config_path = Path(config_path)
    
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    with open(config_path, 'r') as f:
        config_data = yaml.safe_load(f)
    
    return PlayConfig.model_validate(config_data)


def main():
    """Main play function."""
    parser = argparse.ArgumentParser(description="Play/interact with Metta AI agent")
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
        "--no-browser",
        action="store_true",
        help="Don't open browser automatically"
    )
    
    args = parser.parse_args()
    
    # Load configuration from file
    cfg = load_config_from_yaml(args.config)
    
    # Apply command line overrides
    if args.policy_uri:
        cfg.replay_job.policy_uri = args.policy_uri
        
    if args.no_browser:
        cfg.replay_job.open_browser_on_start = False
    
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    logger.info(f"Starting play session with config: {cfg.run}")
    logger.info(f"Policy URI: {cfg.replay_job.policy_uri}")
    logger.info(f"Open browser: {cfg.replay_job.open_browser_on_start}")
    
    # Convert to dict for server compatibility
    cfg_dict = cfg.model_dump()
    
    # Start the server
    open_browser = cfg.replay_job.open_browser_on_start
    ws_url = "%2Fws"
    
    if open_browser:
        server.run(cfg_dict, open_url=f"?wsUrl={ws_url}")
    else:
        logger.info(f"Enter MettaGrid @ {DEV_METTASCOPE_FRONTEND_URL}?wsUrl={ws_url}")
        server.run(cfg_dict)
    
    return 0


if __name__ == "__main__":
    sys.exit(main())