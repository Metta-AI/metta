import os
import random
import signal  # Aggressively exit on ctrl+c
import string
import time
from datetime import datetime

import hydra
from omegaconf import OmegaConf

from mettagrid.mettagrid_env import MettaGridEnv

signal.signal(signal.SIGINT, lambda sig, frame: os._exit(0))


def env_to_ascii(env):
    grid = env._c_env.render_ascii()
    # convert to strings
    return ["".join(row) for row in grid]


def save_env_map(env, target_file, gen_time):
    ascii_grid = env_to_ascii(env)

    resolved_config = env._env_cfg.game.map_builder
    config = env._cfg_template.game.map_builder
    metadata = {
        **env._env_cfg.mapgen.metadata,
        "gen_time": gen_time,
        "timestamp": datetime.now().isoformat(),
    }

    with open(target_file, "w") as f:
        # Note: OmegaConf messes up multiline strings (adds extra newlines).
        # But we take care of it in the mettamap viewer.
        frontmatter = OmegaConf.to_yaml(
            {
                "metadata": metadata,
                "config": config,
                "resolved_config": resolved_config,
            }
        )
        f.write(frontmatter)
        f.write("\n---\n")
        f.write("\n".join(ascii_grid) + "\n")

    print(f"Saved map to {target_file}")


@hydra.main(version_base=None, config_path="../configs", config_name="mapgen")
def main(cfg):
    start = time.time()
    env = MettaGridEnv(cfg, render_mode="human")
    gen_time = time.time() - start
    print(f"Time taken to create env: {gen_time} seconds")

    if cfg.mapgen.save:
        target_name = cfg.mapgen.target.get("name", None)
        if target_name is None:
            random_suffix = "".join(random.choices(string.ascii_lowercase + string.digits, k=8))
            target_name = f"map_{random_suffix}.yaml"

        target_file = os.path.join(cfg.mapgen.target.dir, target_name)
        save_env_map(env, target_file, gen_time=gen_time)

    show = cfg.mapgen.show
    if show == "raylib":
        from mettagrid.renderer.raylib.raylib_renderer import MettaGridRaylibRenderer

        renderer = MettaGridRaylibRenderer(env._c_env, env._env_cfg.game)
        while True:
            renderer.render_and_wait()
    elif show == "ascii":
        ascii_grid = env_to_ascii(env)
        print("\n".join(ascii_grid))
    elif show == "ascii_border":
        ascii_grid = env_to_ascii(env)
        # Useful for generating examples for docstrings in code.
        width = len(ascii_grid[0])
        lines = ["┌" + "─" * width + "┐"]
        for row in ascii_grid:
            lines.append("│" + row + "│")
        lines.append("└" + "─" * width + "┘")
        print("\n".join(lines))
    elif show == "none":
        pass
    else:
        raise ValueError(f"Invalid show mode: {show}")


if __name__ == "__main__":
    main()
