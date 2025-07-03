#!/usr/bin/env -S uv run
# Starts a websocket server that allows you to play as a metta agent.

import hydra
from omegaconf import OmegaConf

import mettascope.server as server
from metta.common.util.script_decorators import get_metta_logger, metta_script


@hydra.main(version_base=None, config_path="../configs", config_name="replay_job")
@metta_script
def main(cfg):
    logger = get_metta_logger()
    logger.info(f"tools.play job config:\n{OmegaConf.to_yaml(cfg, resolve=True)}")

    open_browser = OmegaConf.select(cfg, "replay_job.open_browser_on_start", default=True)

    ws_url = "%2Fws"

    if open_browser:
        server.run(cfg, open_url=f"?wsUrl={ws_url}")
    else:
        logger.info(f"Enter MettaGrid @ http://localhost:8000?wsUrl={ws_url}")
        server.run(cfg)


if __name__ == "__main__":
    main()
