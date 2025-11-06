import typing

import mettagrid.builder

# Local import moved to factory usage to avoid forbidden cross-package dependency at import time
import mettagrid.config.mettagrid_config
import mettagrid.map_builder.map_builder
import mettagrid.map_builder.perimeter_incontext
import mettagrid.map_builder.random
import mettagrid.mapgen.mapgen
import mettagrid.mapgen.scenes.random


def make_arena(
    num_agents: int,
    combat: bool = True,
    map_builder: mettagrid.map_builder.map_builder.MapBuilderConfig
    | None = None,  # custom map builder; must match num_agents
) -> mettagrid.config.mettagrid_config.MettaGridConfig:
    objects = {
        "wall": mettagrid.builder.building.wall,
        "altar": mettagrid.builder.building.assembler_altar,
        "mine_red": mettagrid.builder.building.assembler_mine_red,
        "generator_red": mettagrid.builder.building.assembler_generator_red,
        "lasery": mettagrid.builder.building.assembler_lasery,
        "armory": mettagrid.builder.building.assembler_armory,
    }

    actions = mettagrid.config.mettagrid_config.ActionsConfig(
        noop=mettagrid.config.mettagrid_config.NoopActionConfig(),
        move=mettagrid.config.mettagrid_config.MoveActionConfig(),
        attack=mettagrid.config.mettagrid_config.AttackActionConfig(
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
        map_builder = mettagrid.mapgen.mapgen.MapGen.Config(
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

    return mettagrid.config.mettagrid_config.MettaGridConfig(
        label="arena" + (".combat" if combat else ""),
        game=mettagrid.config.mettagrid_config.GameConfig(
            num_agents=num_agents,
            actions=actions,
            objects=objects,
            agent=mettagrid.config.mettagrid_config.AgentConfig(
                default_resource_limit=50,
                resource_limits={
                    "heart": 255,
                },
                rewards=mettagrid.config.mettagrid_config.AgentRewards(
                    inventory={
                        "heart": 1,
                    },
                ),
            ),
            map_builder=map_builder,
        ),
    )


def make_navigation(num_agents: int) -> mettagrid.config.mettagrid_config.MettaGridConfig:
    nav_assembler = mettagrid.builder.building.AssemblerConfig(
        name="altar",
        type_id=8,
        map_char="_",
        render_symbol="🛣️",
        protocols=[
            mettagrid.builder.building.ProtocolConfig(input_resources={}, output_resources={"heart": 1}, cooldown=255)
        ],
    )
    cfg = mettagrid.config.mettagrid_config.MettaGridConfig(
        game=mettagrid.config.mettagrid_config.GameConfig(
            num_agents=num_agents,
            objects={
                "altar": nav_assembler,
                "wall": mettagrid.builder.building.wall,
            },
            resource_names=["heart"],
            actions=mettagrid.config.mettagrid_config.ActionsConfig(
                move=mettagrid.config.mettagrid_config.MoveActionConfig(enabled=True),
                noop=mettagrid.config.mettagrid_config.NoopActionConfig(enabled=True),
            ),
            agent=mettagrid.config.mettagrid_config.AgentConfig(
                rewards=mettagrid.config.mettagrid_config.AgentRewards(
                    inventory={
                        "heart": 1,
                    },
                ),
            ),
            # Always provide a concrete map builder config so tests can set width/height
            map_builder=mettagrid.map_builder.random.RandomMapBuilder.Config(agents=num_agents),
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
    dir: typing.Optional[str] = None,
) -> mettagrid.config.mettagrid_config.MettaGridConfig:
    game_objects["wall"] = mettagrid.builder.empty_assemblers.wall
    cfg = mettagrid.config.mettagrid_config.MettaGridConfig(
        desync_episodes=False,
        game=mettagrid.config.mettagrid_config.GameConfig(
            max_steps=max_steps,
            num_agents=num_agents,
            objects=game_objects,
            map_builder=mettagrid.mapgen.mapgen.MapGen.Config(
                instances=num_agents,
                instance=mettagrid.map_builder.perimeter_incontext.PerimeterInContextMapBuilder.Config(
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
            actions=mettagrid.config.mettagrid_config.ActionsConfig(
                noop=mettagrid.config.mettagrid_config.NoopActionConfig(),
                move=mettagrid.config.mettagrid_config.MoveActionConfig(),
            ),
            agent=mettagrid.config.mettagrid_config.AgentConfig(
                rewards=mettagrid.config.mettagrid_config.AgentRewards(
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
