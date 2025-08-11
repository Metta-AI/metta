import numpy as np

from metta.mettagrid.level_builder import LevelMap
from metta.mettagrid.mettagrid_config import (
    ActionConfig,
    ActionsConfig,
    AgentConfig,
    AttackActionConfig,
    EnvConfig,
    GameConfig,
    GroupConfig,
    WallConfig,
)
from metta.mettagrid.mettagrid_env import MettaGridEnv


def test_env_map():
    # Create a level map directly with the expected dimensions (7x8 including borders)
    grid = np.full((8, 7), "empty", dtype="<U50")
    # Add walls around perimeter
    grid[0, :] = "wall"
    grid[-1, :] = "wall"
    grid[:, 0] = "wall"
    grid[:, -1] = "wall"
    # Place agent
    grid[3, 3] = "agent.agent"

    level_map = LevelMap(grid=grid, labels=[])

    actions_config = ActionsConfig(
        noop=ActionConfig(),
        move=ActionConfig(),
        rotate=ActionConfig(),
        put_items=ActionConfig(),
        get_items=ActionConfig(),
        attack=AttackActionConfig(),
        swap=ActionConfig(),
    )

    game_config = GameConfig(
        num_agents=1,
        max_steps=1000,
        obs_width=11,
        obs_height=11,
        num_observation_tokens=200,
        agent=AgentConfig(),
        groups={"agent": GroupConfig(id=0)},
        actions=actions_config,
        objects={"wall": WallConfig(type_id=1, swappable=False)},
        level_map=level_map,
    )

    env_config = EnvConfig(game=game_config, desync_episodes=True)

    env = MettaGridEnv(env_config, render_mode="human")

    # Expected dimensions:
    # - Base dimensions: 5x6
    # - MapGen border_width=1 adds 2 to each dimension → 7x8
    assert env.map_width == 7  # 5 + 2*1 (border)
    assert env.map_height == 8  # 6 + 2*1 (border)
