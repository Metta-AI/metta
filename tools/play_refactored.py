#!/usr/bin/env -S uv run

"""Interactive play script for Metta AI - Refactored example."""

import argparse
import logging
import sys
from types import SimpleNamespace
from typing import Optional

from pydantic import Field

import mettascope.server as server
from metta.common.util.constants import DEV_METTASCOPE_FRONTEND_URL
from metta.common.util.typed_config import ConfigWithBuilder
from metta.common.wandb.wandb_config import WandbConfig
from metta.rl.env_config import EnvConfig

logger = logging.getLogger(__name__)


class PlayToolConfig(ConfigWithBuilder):
    """Configuration for the play tool - ONLY contains other configs."""

    # Config objects only - no runtime data
    env: Optional[EnvConfig] = Field(default=None, description="Environment configuration")
    wandb: Optional[WandbConfig] = Field(
        default_factory=lambda: WandbConfig(enabled=False), description="WandB configuration (disabled by default)"
    )


def main():
    """Main play function."""
    parser = argparse.ArgumentParser(description="Play/interact with Metta AI agent")

    # Configuration file (optional for play mode)
    parser.add_argument("--config", type=str, default=None, help="Path to YAML configuration file (optional)")

    # Runtime parameters (not stored in config file)
    parser.add_argument("--policy-uri", type=str, default=None, help="Policy URI to load")
    parser.add_argument("--no-browser", action="store_true", help="Don't open browser automatically")
    parser.add_argument("--device", type=str, default="cpu", help="Device to use (cpu/cuda)")
    parser.add_argument("--port", type=int, default=8000, help="Server port (default: 8000)")
    parser.add_argument("--host", type=str, default="localhost", help="Server host (default: localhost)")

    # Runtime overrides for specific env config fields
    parser.add_argument("--seed", type=int, default=None, help="Override environment seed")
    parser.add_argument(
        "--vectorization",
        type=str,
        default=None,
        choices=["serial", "multiprocessing"],
        help="Override vectorization mode",
    )
    parser.add_argument("--torch-deterministic", action="store_true", help="Enable torch deterministic mode")

    args = parser.parse_args()

    # Load configuration from file if provided, otherwise use defaults
    if args.config:
        cfg = PlayToolConfig.from_file(args.config)
        logger.info(f"Config loaded from: {args.config}")
    else:
        cfg = PlayToolConfig()
        logger.info("Using default configuration")

    # Create environment config if not provided in config file
    if not cfg.env:
        cfg.env = EnvConfig(device=args.device)
    else:
        # Override device from command line
        cfg.env.device = args.device

    # Apply runtime overrides to env config
    if args.seed is not None:
        cfg.env.seed = args.seed

    if args.vectorization:
        cfg.env.vectorization = args.vectorization

    if args.torch_deterministic:
        cfg.env.torch_deterministic = True

    # Setup logging
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")

    logger.info("Starting play session")
    logger.info(f"Policy URI: {args.policy_uri if args.policy_uri else 'None (manual control)'}")
    logger.info(f"Server: {args.host}:{args.port}")
    logger.info(f"Open browser: {not args.no_browser}")
    logger.info(f"Device: {cfg.env.device}")

    if cfg.env.seed is not None:
        logger.info(f"Seed: {cfg.env.seed}")

    if cfg.env.vectorization:
        logger.info(f"Vectorization: {cfg.env.vectorization}")

    # Create runtime context for the server
    # Server expects certain fields that are now command-line arguments
    server_runtime = {
        # Runtime parameters
        "policy_uri": args.policy_uri,
        "open_browser_on_start": not args.no_browser,
        "host": args.host,
        "port": args.port,
        # Config objects (serialized)
        "device": cfg.env.device,
        "env": cfg.env.model_dump() if cfg.env else None,
        "wandb": cfg.wandb.model_dump() if cfg.wandb else {"enabled": False},
    }

    # Convert to SimpleNamespace for dot notation access
    server_cfg = SimpleNamespace(**server_runtime)

    # Start the server
    ws_url = "%2Fws"

    try:
        if not args.no_browser:
            logger.info(f"Opening browser at {DEV_METTASCOPE_FRONTEND_URL}?wsUrl={ws_url}")
            server.run(server_cfg, open_url=f"?wsUrl={ws_url}")
        else:
            logger.info(f"Server running. Enter MettaGrid @ {DEV_METTASCOPE_FRONTEND_URL}?wsUrl={ws_url}")
            server.run(server_cfg)
    except KeyboardInterrupt:
        logger.info("Play session terminated by user")
    except Exception as e:
        logger.error(f"Server error: {e}")
        raise

    return 0


if __name__ == "__main__":
    sys.exit(main())
