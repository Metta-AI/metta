#!/usr/bin/env -S uv run

"""Interactive play script for Metta AI using typer."""

import logging
from types import SimpleNamespace
from typing import Optional

import typer
from pydantic import Field

import mettascope.server as server
from metta.common.util.constants import DEV_METTASCOPE_FRONTEND_URL
from metta.common.util.typed_config import ConfigWithBuilder
from metta.common.wandb.wandb_config import WandbConfig
from metta.rl.env_config import EnvConfig

logger = logging.getLogger(__name__)
app = typer.Typer()


class PlayConfig(ConfigWithBuilder):
    """Configuration for the play script."""

    # Environment configuration (optional)
    env: EnvConfig | None = Field(default=None, description="Environment configuration")

    # WandB configuration (optional)
    wandb: WandbConfig | None = Field(default=None, description="WandB configuration")

    # Direct arguments
    policy_uri: str | None = Field(default=None, description="Policy URI to load")
    no_browser: bool = Field(default=False, description="Don't open browser automatically")
    device: str = Field(default="cpu", description="Device to use (cpu/cuda)")
    env_path: str | None = Field(default=None, description="Environment config path for server")


@app.command()
def main(
    config: Optional[str] = typer.Option(None, "--config", help="Path to YAML configuration file (optional)"),
    policy_uri: Optional[str] = typer.Option(None, "--policy-uri", help="Policy URI to load"),
    no_browser: bool = typer.Option(False, "--no-browser", help="Don't open browser automatically"),
    device: str = typer.Option("cpu", "--device", help="Device to use (cpu/cuda)"),
    env_path: Optional[str] = typer.Option(None, "--env-path", help="Environment config path for server"),
):
    """Play/interact with Metta AI agent."""
    # Load configuration from file if provided, otherwise create default
    if config:
        cfg = PlayConfig.from_file(config)
    else:
        cfg = PlayConfig()

    # Apply command line overrides
    if policy_uri:
        cfg.policy_uri = policy_uri
    cfg.no_browser = no_browser
    cfg.device = device
    if env_path:
        cfg.env_path = env_path

    # Create environment config if not provided
    if not cfg.env:
        cfg.env = EnvConfig(device=cfg.device)
    else:
        # Override device if specified on command line
        cfg.env.device = cfg.device

    # Setup logging
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")

    logger.info("Starting play session")
    logger.info(f"Policy URI: {cfg.policy_uri}")
    logger.info(f"Open browser: {not cfg.no_browser}")
    logger.info(f"Device: {cfg.env.device}")

    # Create a config-like object for the server that includes all necessary fields
    server_config = {
        "policy_uri": cfg.policy_uri,
        "open_browser_on_start": not cfg.no_browser,
        "device": cfg.env.device,
        "env": cfg.env_path,
        "wandb": cfg.wandb.model_dump() if cfg.wandb else {"enabled": False},
    }

    # Convert to a simple namespace object that can be accessed with dot notation
    server_cfg = SimpleNamespace(**server_config)

    # Start the server
    ws_url = "%2Fws"

    if not cfg.no_browser:
        server.run(server_cfg, open_url=f"?wsUrl={ws_url}")
    else:
        logger.info(f"Enter MettaGrid @ {DEV_METTASCOPE_FRONTEND_URL}?wsUrl={ws_url}")
        server.run(server_cfg)

    return 0


if __name__ == "__main__":
    app()
