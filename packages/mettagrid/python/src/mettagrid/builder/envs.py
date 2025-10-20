from typing import Optional

import mettagrid.mapgen.scenes.random
from mettagrid.builder import building, empty_assemblers

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
from mettagrid.map_builder.map_builder import MapBuilderConfig
from mettagrid.map_builder.perimeter_incontext import PerimeterInContextMapBuilder
from mettagrid.map_builder.random import RandomMapBuilder
from mettagrid.mapgen.mapgen import MapGen


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
            instance=mettagrid.mapgen.scenes.random.Random.Config(
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
    nav_assembler = building.AssemblerConfig(
        name="altar",
        type_id=8,
        map_char="_",
        render_symbol="🛣️",
        recipes=[([], building.RecipeConfig(input_resources={}, output_resources={"heart": 1}, cooldown=255))],
    )
    cfg = MettaGridConfig(
        game=GameConfig(
            num_agents=num_agents,
            objects={
                "altar": nav_assembler,
                "wall": building.wall,
            },
            resource_names=["heart"],
            actions=ActionsConfig(
                move=ActionConfig(enabled=True),
                noop=ActionConfig(enabled=True),
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


def make_assembly_lines(
    num_agents: int,
    max_steps,
    game_objects: dict,
    map_builder_objects: dict,
    width: int = 6,
    height: int = 6,
    terrain: str = "no-terrain",
    chain_length: int = 2,
    num_sinks: int = 0,
    dir: Optional[str] = None,
) -> MettaGridConfig:
    game_objects["wall"] = empty_assemblers.wall
    cfg = MettaGridConfig(
        desync_episodes=False,
        game=GameConfig(
            max_steps=max_steps,
            num_agents=num_agents,
            objects=game_objects,
            map_builder=MapGen.Config(
                instances=num_agents,
                instance=PerimeterInContextMapBuilder.Config(
                    agents=1,
                    width=width,
                    height=height,
                    objects=map_builder_objects,
                    density=terrain,
                    chain_length=chain_length,
                    num_sinks=num_sinks,
                    dir=dir,
                ),
            ),
            actions=ActionsConfig(
                move=ActionConfig(),
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
