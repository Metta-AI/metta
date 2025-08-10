#!/usr/bin/env -S uv run

"""Generate a replay file for visualization in MettaScope using argparse."""

import argparse
import logging
import platform
import sys
import yaml
from pathlib import Path
from urllib.parse import quote
from typing import Any

from pydantic import Field

import mettascope.server as server
from metta.agent.mocks import MockPolicyRecord
from metta.common.util.constants import DEV_METTASCOPE_FRONTEND_URL
from metta.common.util.typed_config import BaseModelWithForbidExtra
from metta.common.wandb.wandb_context import WandbContext
from metta.rl.env_config import create_env_config, EnvConfig
from metta.sim.simulation import Simulation
from metta.sim.simulation_config import SingleEnvSimulationConfig
from tools.utils import get_policy_store_from_cfg


logger = logging.getLogger(__name__)


class ReplayJobConfig(BaseModelWithForbidExtra):
    """Configuration for replay job."""
    
    sim: SingleEnvSimulationConfig = Field(description="Single environment simulation config")
    policy_uri: str | None = Field(default=None, description="Policy URI")
    selector_type: str = Field(default="top", description="Policy selector type")
    replay_dir: str = Field(description="Replay directory")
    stats_dir: str = Field(description="Stats directory")
    open_browser_on_start: bool = Field(default=True, description="Open browser automatically")


class ReplayConfig(BaseModelWithForbidExtra):
    """Configuration for the replay script."""
    
    # Run configuration
    run: str = Field(description="Run name/identifier")
    
    # Replay job configuration
    replay_job: ReplayJobConfig = Field(description="Replay job config")
    
    # Environment configuration
    env: EnvConfig | None = Field(default=None, description="Environment configuration")
    device: str = Field(default="cuda", description="Device to use")
    
    # WandB configuration
    wandb: dict[str, Any] = Field(default_factory=dict, description="WandB configuration")


def load_config_from_yaml(config_path: str) -> ReplayConfig:
    """Load configuration from YAML file."""
    config_path = Path(config_path)
    
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    with open(config_path, 'r') as f:
        config_data = yaml.safe_load(f)
    
    return ReplayConfig.model_validate(config_data)


def main():
    """Main replay function."""
    parser = argparse.ArgumentParser(description="Generate replay for Metta AI visualization")
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
        "--device",
        type=str,
        default=None,
        help="Override device (cuda/cpu)"
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
        
    if args.device:
        cfg.device = args.device
        
    if args.no_browser:
        cfg.replay_job.open_browser_on_start = False
    
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    logger.info(f"Starting replay generation with config: {cfg.run}")
    logger.info(f"Policy URI: {cfg.replay_job.policy_uri}")
    logger.info(f"Device: {cfg.device}")
    
    # Set up directories based on run name
    replay_dir = Path("train_dir") / cfg.run / "replays"
    stats_dir = Path("train_dir") / cfg.run / "stats"
    
    replay_dir.mkdir(parents=True, exist_ok=True)
    stats_dir.mkdir(parents=True, exist_ok=True)
    
    cfg.replay_job.replay_dir = str(replay_dir)
    cfg.replay_job.stats_dir = str(stats_dir)
    
    # Create env config
    if cfg.env:
        env_cfg = cfg.env
    else:
        env_cfg = EnvConfig(device=cfg.device)
    
    # Setup WandB context
    wandb_context = None
    if cfg.wandb:
        from metta.common.wandb.wandb_context import WandbRun
        wandb_run = WandbRun(
            project=cfg.wandb.get("project", "metta"),
            name=cfg.run,
            config=cfg.model_dump(),
            tags=cfg.wandb.get("tags", [])
        )
        wandb_context = WandbContext(wandb_run)
    
    try:
        with wandb_context if wandb_context else nullcontext():
            # Get policy store
            config_dict = {
                "device": cfg.device,
                **cfg.model_dump()
            }
            policy_store = get_policy_store_from_cfg(config_dict, wandb_context)
            
            # Get policy record
            if cfg.replay_job.policy_uri is not None:
                policy_record = policy_store.policy_record(cfg.replay_job.policy_uri)
            else:
                policy_record = MockPolicyRecord(run_name="replay_run", uri=None)
            
            # Create simulation
            sim_config = cfg.replay_job.sim
            simulation = Simulation(
                simulation_config=sim_config,
                policy_record=policy_record,
                env_config=env_cfg
            )
            
            logger.info("Running simulation to generate replay...")
            
            # Run simulation
            result = simulation.run(
                stats_dir=Path(cfg.replay_job.stats_dir),
                replay_dir=Path(cfg.replay_job.replay_dir)
            )
            
            logger.info(f"Replay generated successfully")
            logger.info(f"Stats saved to: {cfg.replay_job.stats_dir}")
            logger.info(f"Replays saved to: {cfg.replay_job.replay_dir}")
            
            # Start server if requested
            if cfg.replay_job.open_browser_on_start:
                logger.info("Starting MettaScope server...")
                
                # Convert config to dict for server
                cfg_dict = cfg.model_dump()
                
                # Start server with replay
                ws_url = "%2Fws"
                server.run(cfg_dict, open_url=f"?wsUrl={ws_url}")
            else:
                logger.info(f"Replay ready. View at: {DEV_METTASCOPE_FRONTEND_URL}")
    
    except Exception as e:
        logger.error(f"Failed to generate replay: {e}")
        raise
    
    return 0


# Context manager for optional wandb
from contextlib import nullcontext

if __name__ == "__main__":
    sys.exit(main())