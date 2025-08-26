from typing import Literal

import numpy as np

import mettascope.server
from metta.mettagrid.config.envs import make_arena
from metta.mettagrid.mapgen.utils.storable_map import StorableMap, grid_to_lines
from metta.mettagrid.mettagrid_env import MettaGridEnv
from metta.sim.map_preview import write_local_map_preview
from metta.sim.simulation_config import SimulationConfig
from metta.tools.play import PlayTool

ShowMode = Literal["mettascope", "ascii", "ascii_border", "none"]


def show_map(storable_map: StorableMap, mode: ShowMode | None):
    if not mode or mode == "none":
        return

    if mode == "mettascope":
        num_agents = np.count_nonzero(np.char.startswith(storable_map.grid, "agent"))
        env_cfg = make_arena(num_agents=num_agents)
        env_cfg = env_cfg.with_ascii_map(map_data=[list(line) for line in grid_to_lines(storable_map.grid)])
        env = MettaGridEnv(env_cfg, render_mode="rgb_array")

        file_path = write_local_map_preview(env)

        play = PlayTool(
            sim=SimulationConfig(
                env=env_cfg,
                name="map.utils.show",
            ),
            open_browser_on_start=True,
        )
        mettascope.server.run(play, open_url=f"?replayUrl=local/{file_path}")

    elif mode == "ascii":
        ascii_lines = grid_to_lines(storable_map.grid)
        print("\n".join(ascii_lines))

    elif mode == "ascii_border":
        ascii_lines = grid_to_lines(storable_map.grid, border=True)
        print("\n".join(ascii_lines))

    else:
        raise ValueError(f"Invalid show mode: {mode}")
