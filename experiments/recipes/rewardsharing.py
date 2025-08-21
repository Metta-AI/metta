from typing import Optional

import metta.cogworks.curriculum as cc
from metta.map.mapgen import MapGen
from metta.map.scene import ChildrenAction
from metta.map.scenes.random import Random
from metta.map.scenes.room_grid import RoomGrid
from metta.map.types import AreaWhere
from metta.mettagrid.config.envs import make_arena
from metta.mettagrid.mettagrid_config import ActionConfig, ActionsConfig, EnvConfig
from metta.sim.simulation_config import SimulationConfig


def make_env() -> EnvConfig:
    # 2 agents, no combat
    env = make_arena(num_agents=2, combat=False)

    # Minimal actions for pickup/drop
    env.game.actions = ActionsConfig(
        move=ActionConfig(),
        rotate=ActionConfig(),
        get_items=ActionConfig(),
        put_items=ActionConfig(),
    )

    # 100% team reward sharing
    env.game.groups["agent"].group_reward_pct = 1.0

    # Grid split into: top room | divider band | bottom room
    # We set RoomGrid border_width=0 so we fully control the divider with our own walls.
    scene = RoomGrid.factory(
        RoomGrid.Params(rows=3, columns=1, border_width=0, border_object="wall"),
        children_actions=[
            # Top room: 1 agent + 1 mine
            ChildrenAction(
                scene=Random.factory(Random.Params(objects={"mine_red": 1}, agents=1)),
                where=AreaWhere(tags=["room_0_0"]),
            ),
            # Divider band: place generator in empty band first...
            ChildrenAction(
                scene=Random.factory(
                    Random.Params(objects={"generator_red": 1}, agents=0)
                ),
                where=AreaWhere(tags=["room_1_0"]),
            ),
            # ...then fill the rest of the divider with walls so it acts like a wall band.
            # Using a large count; Random will cap to available empty cells.
            ChildrenAction(
                scene=Random.factory(Random.Params(objects={"wall": 10_000}, agents=0)),
                where=AreaWhere(tags=["room_1_0"]),
            ),
            # Bottom room: 1 agent + 1 altar
            ChildrenAction(
                scene=Random.factory(Random.Params(objects={"altar": 1}, agents=1)),
                where=AreaWhere(tags=["room_2_0"]),
            ),
        ],
    )

    # Map size: adjust as you like; seed=None randomizes each episode
    env.game.map_builder = MapGen.Config(
        width=11,
        height=11,
        border_width=0,  # no outer border for simplicity on small maps
        instance_border_width=0,
        root=scene,
        seed=None,
    )

    return env


def make_curriculum(env: Optional[EnvConfig] = None):
    base_env = env or make_env()
    tasks = cc.bucketed(base_env)
    # Train on sizes 3,4,5,6 only
    tasks.add_bucket("game.map_builder.width", [3, 4, 5, 6])
    tasks.add_bucket("game.map_builder.height", [3, 4, 5, 6])
    return tasks.to_curriculum()


def make_evals(env: Optional[EnvConfig] = None) -> list[SimulationConfig]:
    base_env = env or make_env()
    evals: list[SimulationConfig] = []
    for s in [3, 4, 5, 6, 7]:
        e = base_env.model_copy(deep=True)
        e.game.map_builder.width = s
        e.game.map_builder.height = s
        evals.append(SimulationConfig(name=f"eval_size_{s}", env=e))
    return evals
