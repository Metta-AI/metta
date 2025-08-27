import argparse
from pathlib import Path

import numpy as np

from metta.mettagrid.config import building
from metta.mettagrid.map_builder.ascii import AsciiMapBuilder
from metta.mettagrid.mettagrid_config import ActionConfig, ActionsConfig, EnvConfig, GameConfig
from metta.mettagrid.mettagrid_env import MettaGridEnv
from metta.mettagrid.replay_writer import ReplayWriter


def parse_args():
    p = argparse.ArgumentParser(description="Run a basic_plant demo and write a replay")
    p.add_argument("--steps", type=int, default=60, help="Number of steps to simulate")
    p.add_argument("--out", type=str, default="outputs/replays/basic_plant_demo.json.z", help="Replay output path")
    p.add_argument(
        "--map", type=str, default=str(Path(__file__).parent.parent / "mettagrid" / "configs" / "basic_plant_map.txt")
    )
    return p.parse_args()


def main():
    args = parse_args()

    # Build a tiny env config that includes basic_plant and minimal actions
    actions = ActionsConfig(
        noop=ActionConfig(),
        get_items=ActionConfig(),
    )

    # minimal object set: walls and basic_plant from default building presets
    objects = {
        "wall": building.wall,
        "basic_plant": building.basic_plant,
    }

    # ASCII map
    ascii_cfg = AsciiMapBuilder.Config.from_uri(args.map)

    env_cfg = EnvConfig(
        label="basic_plant_demo",
        game=GameConfig(
            num_agents=1,
            actions=actions,
            objects=objects,
            map_builder=ascii_cfg,
            max_steps=1000,
        ),
    )

    env = MettaGridEnv(env_cfg=env_cfg)
    writer = ReplayWriter(replay_dir=str(Path(args.out).parent))
    episode_id = "basic_plant_demo"
    writer.start_episode(episode_id, env)

    obs, info = env.reset()
    # No-op actions (two-argument action space expected; zeros ok)
    for _ in range(args.steps):
        actions = np.zeros((env.num_agents, 2), dtype=np.int32)
        obs, rewards, terminals, truncations, infos = env.step(actions)
        writer.log_step(episode_id, actions, rewards)
        if terminals.all() or truncations.all():
            break

    url = writer.write_replay(episode_id)
    print("Replay written:", url or args.out)


if __name__ == "__main__":
    main()
