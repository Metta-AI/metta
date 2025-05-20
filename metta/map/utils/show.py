from typing import Any, Literal, cast

import hydra
import numpy as np
from omegaconf.omegaconf import OmegaConf

<<<<<<< HEAD:metta/map/utils/show.py
from metta.map.utils.storable_map import StorableMap, grid_to_ascii
=======
import mettascope.server
from metta.map.utils.storable_map import StorableMap, grid_to_ascii
from metta.sim.map_preview import write_local_map_preview
>>>>>>> 13c12a2fdf120e435aa056c95de09aa7ccaa5a87:deps/mettagrid/mettagrid/map/utils/show.py
from mettagrid.mettagrid_env import MettaGridEnv

ShowMode = Literal["mettascope", "ascii", "ascii_border", "none"]


def show_map(storable_map: StorableMap, mode: ShowMode | None):
    if not mode or mode == "none":
        return

    if mode == "mettascope":
        num_agents = np.count_nonzero(np.char.startswith(storable_map.grid, "agent"))

        env_cfg = OmegaConf.load("./configs/env/mettagrid/mettagrid.yaml")
        env_cfg.game.num_agents = num_agents

        env = MettaGridEnv(cast(Any, env_cfg), env_map=storable_map.grid, render_mode="none")

        file_path = write_local_map_preview(env)
        url_path = file_path.split("mettascope/")[-1]

        with hydra.initialize(version_base=None, config_path="../../../configs"):
            cfg = hydra.compose(config_name="replay_job")
            mettascope.server.run(cfg, open_url=f"?replayUrl={url_path}")

    elif mode == "ascii":
        ascii_lines = grid_to_ascii(storable_map.grid)
        print("\n".join(ascii_lines))

    elif mode == "ascii_border":
        ascii_lines = grid_to_ascii(storable_map.grid, border=True)
        print("\n".join(ascii_lines))

    else:
        raise ValueError(f"Invalid show mode: {mode}")
