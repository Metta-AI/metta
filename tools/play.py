#!/usr/bin/env -S uv run
# Starts a websocket server that allows you to play as a metta agent.

import logging

from omegaconf import DictConfig, OmegaConf

import mettascope.server as server
from metta.util.metta_script import metta_script


def main(cfg: DictConfig):
    logger = logging.getLogger("tools.play")
    logger.info(f"tools.play job config:\n{OmegaConf.to_yaml(cfg, resolve=True)}")

    open_browser = OmegaConf.select(cfg, "replay_job.open_browser_on_start", default=True)

    ws_url = "%2Fws"

    if open_browser:
        server.run(cfg, open_url=f"?wsUrl={ws_url}")
    else:
        logger.info(f"Enter MettaGrid @ http://localhost:8000?wsUrl={ws_url}")
        server.run(cfg)


metta_script(main, "replay_job")
