#!/usr/bin/env -S uv run

"""
Demo showing how to create an MettaGridConfig and build a game map using the map builder.
"""

from mettagrid.config.mettagrid_config import ActionConfig, ActionsConfig, MettaGridConfig
from mettagrid.map_builder.random import RandomMapBuilder


def main():
    mg_config = MettaGridConfig()
    mg_config.game.num_agents = 24

    mg_config.game.map_builder = RandomMapBuilder.Config(
        agents=24, width=10, height=10, objects={"wall": 10, "altar": 1}, border_width=1, border_object="wall"
    )

    mg_config.game.actions = ActionsConfig(
        move=ActionConfig(),
        rotate=ActionConfig(),
    )

    print("=== mg_config ===")
    print(mg_config.model_dump_json(indent=2))

    map_builder = mg_config.game.map_builder.create()
    game_map = map_builder.build()

    print("=== game_map ===")
    print(game_map.grid)


if __name__ == "__main__":
    main()
