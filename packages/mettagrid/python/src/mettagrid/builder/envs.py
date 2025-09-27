from typing import Optional

import mettagrid.mapgen.scenes.random

# Local import moved to factory usage to avoid forbidden cross-package dependency at import time
from mettagrid.config.mettagrid_config import (
    ActionConfig,
    ActionsConfig,
    AgentConfig,
    AgentRewards,
    AttackActionConfig,
    GameConfig,
    MettaGridConfig,
)
from mettagrid.map_builder.assembler_map_builder import AssemblerMapBuilder
from mettagrid.map_builder.map_builder import MapBuilderConfig
from mettagrid.map_builder.perimeter_incontext import PerimeterInContextMapBuilder
from mettagrid.map_builder.random import RandomMapBuilder
from mettagrid.mapgen.mapgen import MapGen

from . import building, empty_converters


def make_arena(
    num_agents: int,
    combat: bool = True,
    map_builder: MapBuilderConfig | None = None,  # custom map builder; must match num_agents
) -> MettaGridConfig:
    objects = {
        "wall": building.wall,
        "altar": building.altar,
        "mine_red": building.mine_red,
        "generator_red": building.generator_red,
        "lasery": building.lasery,
        "armory": building.armory,
    }

    actions = ActionsConfig(
        noop=ActionConfig(),
        move=ActionConfig(),
        rotate=ActionConfig(enabled=False),  # Disabled for unified movement system
        put_items=ActionConfig(),
        get_items=ActionConfig(),
        attack=AttackActionConfig(
            consumed_resources={
                "laser": 1,
            },
            defense_resources={
                "armor": 1,
            },
        ),
        swap=ActionConfig(enabled=False),
        change_color=ActionConfig(enabled=False),
    )

    if not combat:
        actions.attack.consumed_resources = {"laser": 100}

    if map_builder is None:
        map_builder = MapGen.Config(
            num_agents=num_agents,
            width=25,
            height=25,
            border_width=6,
            instance_border_width=0,
            root=mettagrid.mapgen.scenes.random.Random.factory(
                params=mettagrid.mapgen.scenes.random.Random.Params(
                    agents=6,
                    objects={
                        "wall": 10,
                        "altar": 5,
                        "mine_red": 10,
                        "generator_red": 5,
                        "lasery": 1,
                        "armory": 1,
                    },
                ),
            ),
        )

    return MettaGridConfig(
        label="arena" + (".combat" if combat else ""),
        game=GameConfig(
            num_agents=num_agents,
            actions=actions,
            objects=objects,
            agent=AgentConfig(
                default_resource_limit=50,
                resource_limits={
                    "heart": 255,
                },
                rewards=AgentRewards(
                    inventory={
                        "heart": 1,
                    },
                ),
            ),
            map_builder=map_builder,
        ),
    )


def make_navigation(num_agents: int) -> MettaGridConfig:
    altar = empty_converters.altar.model_copy()
    altar.cooldown = 255  # Maximum cooldown
    altar.initial_resource_count = 1
    altar.max_conversions = 0
    altar.input_resources = {}
    altar.output_resources = {"heart": 1}
    cfg = MettaGridConfig(
        game=GameConfig(
            num_agents=num_agents,
            objects={
                "altar": altar,
                "wall": building.wall,
            },
            resource_names=["heart"],
            actions=ActionsConfig(
                move=ActionConfig(),
                rotate=ActionConfig(enabled=False),
                get_items=ActionConfig(),
            ),
            agent=AgentConfig(
                rewards=AgentRewards(
                    inventory={
                        "heart": 1,
                    },
                ),
            ),
            # Always provide a concrete map builder config so tests can set width/height
            map_builder=RandomMapBuilder.Config(agents=num_agents),
        )
    )
    return cfg


def make_navigation_sequence(num_agents: int) -> MettaGridConfig:
    altar = building.altar.model_copy()
    altar.input_resources = {"battery_red": 1}
    altar.cooldown = 15
    mine = building.mine_red.model_copy()
    mine.cooldown = 15
    generator = building.generator_red.model_copy()
    generator.cooldown = 15
    cfg = MettaGridConfig(
        game=GameConfig(
            num_agents=num_agents,
            objects={
                "altar": altar,
                "wall": building.wall,
                "mine_red": mine,
                "generator_red": generator,
            },
            resource_names=["heart", "ore_red", "battery_red"],
            actions=ActionsConfig(
                move=ActionConfig(),
                rotate=ActionConfig(enabled=False),
                get_items=ActionConfig(),
            ),
            agent=AgentConfig(
                rewards=AgentRewards(
                    inventory={
                        "heart": 1,
                        "ore_red": 0.001,
                        "battery_red": 0.01,
                    },
                ),
                default_resource_limit=1,
                resource_limits={
                    "heart": 100,
                },
            ),
            # Always provide a concrete map builder config so tests can set width/height
            map_builder=RandomMapBuilder.Config(agents=num_agents),
        )
    )
    return cfg


def make_in_context_chains(
    num_agents: int,
    max_steps,
    game_objects: dict,
    map_builder_objects: dict,
    width: int = 6,
    height: int = 6,
    obstacle_type: Optional[str] = None,
    density: Optional[str] = None,
    chain_length: int = 2,
    num_sinks: int = 0,
    dir: Optional[str] = None,
) -> MettaGridConfig:
    game_objects["wall"] = empty_converters.wall
    cfg = MettaGridConfig(
        desync_episodes=False,
        game=GameConfig(
            max_steps=max_steps,
            num_agents=num_agents,
            objects=game_objects,
            map_builder=MapGen.Config(
                instances=num_agents,
                instance_map=PerimeterInContextMapBuilder.Config(
                    agents=1,
                    width=width,
                    height=height,
                    objects=map_builder_objects,
                    obstacle_type=obstacle_type,
                    density=density,
                    chain_length=chain_length,
                    num_sinks=num_sinks,
                    dir=dir,
                ),
            ),
            actions=ActionsConfig(
                move=ActionConfig(),
                rotate=ActionConfig(enabled=False),  # Disabled for unified movement system
                get_items=ActionConfig(),
                put_items=ActionConfig(),
            ),
            agent=AgentConfig(
                rewards=AgentRewards(
                    inventory={
                        "heart": 1,
                    },
                ),
                default_resource_limit=1,
                resource_limits={"heart": 15},
            ),
        ),
    )
    return cfg


def make_icl_assembler(
    num_agents: int,
    num_instances: int,
    max_steps,
    game_objects: dict,
    map_builder_objects: dict,
    width: int = 6,
    height: int = 6,
    terrain: str = "",
) -> MettaGridConfig:
    game_objects["wall"] = empty_converters.wall
    cfg = MettaGridConfig(
        desync_episodes=False,
        game=GameConfig(
            max_steps=max_steps,
            num_agents=num_agents * num_instances,
            objects=game_objects,
            map_builder=MapGen.Config(
                instances=num_instances,
                instance_map=AssemblerMapBuilder.Config(
                    agents=num_agents,
                    width=width,
                    height=height,
                    objects=map_builder_objects,
                    terrain=terrain,
                ),
            ),
            actions=ActionsConfig(
                move=ActionConfig(),
                rotate=ActionConfig(enabled=False),  # Disabled for unified movement system
                get_items=ActionConfig(),
            ),
            agent=AgentConfig(
                rewards=AgentRewards(
                    inventory={
                        "heart": 1,
                    },
                ),
                default_resource_limit=1,
                resource_limits={"heart": 15},
            ),
        ),
    )
    return cfg


def make_icl_with_numpy(
    num_agents: int,
    num_instances: int,
    max_steps,
    game_objects: dict,
    instance_map: MapBuilderConfig,
) -> MettaGridConfig:
    game_objects["wall"] = empty_converters.wall
    cfg = MettaGridConfig(
        desync_episodes=False,
        game=GameConfig(
            max_steps=max_steps,
            num_agents=num_agents * num_instances,
            objects=game_objects,
            map_builder=MapGen.Config(
                instances=num_instances,
                instance_map=instance_map,
            ),
            actions=ActionsConfig(
                move=ActionConfig(),
                rotate=ActionConfig(enabled=False),  # Disabled for unified movement system
                get_items=ActionConfig(),
                put_items=ActionConfig(),
            ),
            agent=AgentConfig(
                rewards=AgentRewards(
                    inventory={
                        "heart": 1,
                    },
                ),
                default_resource_limit=1,
                resource_limits={"heart": 15},
            ),
        ),
    )

    return cfg
