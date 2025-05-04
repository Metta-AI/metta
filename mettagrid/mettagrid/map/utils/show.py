from typing import Any, Literal, cast

import numpy as np

from mettagrid.config.utils import get_cfg
from mettagrid.map.utils.storable_map import StorableMap, grid_to_ascii
from mettagrid.mettagrid_env import MettaGridEnv

ShowMode = Literal["raylib", "ascii", "ascii_border", "none"]


def show_map(storable_map: StorableMap, mode: ShowMode | None):
    if not mode or mode == "none":
        return

    if mode == "raylib":
        num_agents = np.count_nonzero(np.char.startswith(storable_map.grid, "agent"))

        env_cfg = get_cfg("show_map")
        env_cfg.game.num_agents = num_agents

        env = MettaGridEnv(cast(Any, env_cfg), env_map=storable_map.grid, render_mode="none")

        from mettagrid.renderer.raylib.raylib_renderer import MettaGridRaylibRenderer

        renderer = MettaGridRaylibRenderer(env._c_env, env._env_cfg.game)
        while True:
            renderer.render_and_wait()

    elif mode == "ascii":
        ascii_lines = grid_to_ascii(storable_map.grid)
        print("\n".join(ascii_lines))

    elif mode == "ascii_border":
        ascii_lines = grid_to_ascii(storable_map.grid, border=True)
        print("\n".join(ascii_lines))

    else:
        raise ValueError(f"Invalid show mode: {mode}")
