#!/usr/bin/env -S uv run

"""
Demo showing how to create an EnvConfig and build a game map using the map builder.
"""

from metta.mettagrid.map_builder.random import RandomMapBuilder
from metta.mettagrid.mettagrid_config import ActionConfig, ActionsConfig, EnvConfig

env_config = EnvConfig()
env_config.game.num_agents = 24

env_config.game.map_builder = RandomMapBuilder.Config(
    agents=24, width=10, height=10, objects={"wall": 10, "altar": 1}, border_width=1, border_object="wall"
)

env_config.game.actions = ActionsConfig(
    move_8way=ActionConfig(enabled=True),
    rotate=ActionConfig(enabled=True),
)

print("=== env_config ===")
print(env_config.model_dump_json(indent=2))

map_builder = env_config.game.map_builder.create()
game_map = map_builder.build()

print("=== game_map ===")
print(game_map.grid)
