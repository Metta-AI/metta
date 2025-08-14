from typing import Literal

import hydra
import numpy as np

import mettascope.server
from metta.map.utils.storable_map import StorableMap, grid_to_lines
from metta.mettagrid.config.builder import make_arena
from metta.mettagrid.mettagrid_env import MettaGridEnv
from metta.sim.map_preview import write_local_map_preview

ShowMode = Literal["mettascope", "ascii", "ascii_border", "none"]


def show_map(storable_map: StorableMap, mode: ShowMode | None):
    if not mode or mode == "none":
        return

    if mode == "mettascope":
        num_agents = np.count_nonzero(np.char.startswith(storable_map.grid, "agent"))
        env_cfg = make_arena(num_agents=num_agents)
        env = MettaGridEnv(env_cfg, render_mode="rgb_array")

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
