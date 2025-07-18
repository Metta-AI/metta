from typing import Literal

import hydra
import numpy as np
from omegaconf import DictConfig
from omegaconf.omegaconf import OmegaConf

import mettascope.server
from metta.map.utils.storable_map import StorableMap, grid_to_lines
from metta.mettagrid.curriculum import single_task
from metta.mettagrid.level_builder import Level
from metta.mettagrid.mettagrid_env import MettaGridEnv
from metta.sim.map_preview import write_local_map_preview

ShowMode = Literal["mettascope", "ascii", "ascii_border", "none"]


def show_map(storable_map: StorableMap, mode: ShowMode | None):
    if not mode or mode == "none":
        return

    if mode == "mettascope":
        num_agents = np.count_nonzero(np.char.startswith(storable_map.grid, "agent"))

        env_cfg = OmegaConf.load("./configs/env/mettagrid/full.yaml")
        env_cfg.game.num_agents = int(num_agents)
        OmegaConf.resolve(env_cfg)
        assert isinstance(env_cfg, DictConfig)

        level = Level(storable_map.grid, [])
        env = MettaGridEnv(single_task("show_map", env_cfg), level=level, render_mode="none")

        file_path = write_local_map_preview(env)

        with hydra.initialize(version_base=None, config_path="../../../configs"):
            cfg = hydra.compose(config_name="replay_job")
            mettascope.server.run(cfg, open_url=f"?replayUrl=local/{file_path}")

    elif mode == "ascii":
        ascii_lines = grid_to_lines(storable_map.grid)
        print("\n".join(ascii_lines))

    elif mode == "ascii_border":
        ascii_lines = grid_to_lines(storable_map.grid, border=True)
        print("\n".join(ascii_lines))

    else:
        raise ValueError(f"Invalid show mode: {mode}")
