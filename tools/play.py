#!/usr/bin/env -S uv run

"""Interactive play script for Metta AI using argparse instead of Hydra."""

import argparse
import logging
import sys
from typing import Any

from pydantic import Field

import mettascope.server as server
from metta.common.util.constants import DEV_METTASCOPE_FRONTEND_URL
from metta.common.util.typed_config import ConfigWithBuilder

logger = logging.getLogger(__name__)


class PlayConfig(ConfigWithBuilder):
    """Configuration for the play script."""

    # Play configuration fields
    open_browser_on_start: bool = Field(default=True, description="Open browser automatically")
    policy_uri: str | None = Field(default=None, description="Policy URI to load")
    env: str | None = Field(default=None, description="Environment config path")
    device: str = Field(default="cpu", description="Device to use")

    # WandB config
    wandb: dict[str, Any] = Field(default_factory=dict, description="WandB configuration")


def main():
    """Main play function."""
    parser = argparse.ArgumentParser(description="Play/interact with Metta AI agent")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML configuration file")
    parser.add_argument("--policy-uri", type=str, default=None, help="Override policy URI")
    parser.add_argument("--no-browser", action="store_true", help="Don't open browser automatically")

    args = parser.parse_args()

    # Load configuration from file
    cfg = PlayConfig.from_file(args.config)

    # Apply command line overrides
    if args.policy_uri:
        cfg.policy_uri = args.policy_uri

    if args.no_browser:
        cfg.open_browser_on_start = False

    # Setup logging
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")

    logger.info("Starting play session")
    logger.info(f"Policy URI: {cfg.policy_uri}")
    logger.info(f"Open browser: {cfg.open_browser_on_start}")

    # Start the server (pass Pydantic config directly)
    open_browser = cfg.open_browser_on_start
    ws_url = "%2Fws"

    if open_browser:
        server.run(cfg, open_url=f"?wsUrl={ws_url}")
    else:
        logger.info(f"Enter MettaGrid @ {DEV_METTASCOPE_FRONTEND_URL}?wsUrl={ws_url}")
        server.run(cfg)

    return 0


if __name__ == "__main__":
    sys.exit(main())
