import os
import signal  # Aggressively exit on ctrl+c

import hydra
from mettagrid.mettagrid_env import MettaGridEnv
from mettagrid.renderer.raylib.raylib_renderer import MettaGridRaylibRenderer

signal.signal(signal.SIGINT, lambda sig, frame: os._exit(0))

@hydra.main(version_base=None, config_path="../configs", config_name="simple")
def main(cfg):
    env = MettaGridEnv(cfg, render_mode="human")
    renderer = MettaGridRaylibRenderer(env._c_env, env._env_cfg.game)

    while True:
        renderer.render_and_wait()


if __name__ == "__main__":
    main()
