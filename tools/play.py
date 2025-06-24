#!/usr/bin/env -S uv run
# Starts a websocket server that allows you to play as a metta agent.

import hydra

import mettascope.server as server


@hydra.main(version_base=None, config_path="../configs", config_name="replay_job")
def main(cfg):
    server.run(cfg, open_url="?wsUrl=%2Fws")


if __name__ == "__main__":
    main()
