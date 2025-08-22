import metta.map.scenes.random
from metta.map.mapgen import MapGen
from metta.mettagrid.config import building, empty_converters
from metta.mettagrid.map_builder.map_builder import MapBuilderConfig
from metta.mettagrid.map_builder.random import RandomMapBuilder
from metta.mettagrid.mettagrid_config import (
    ActionConfig,
    ActionsConfig,
    AgentConfig,
    AgentRewards,
    AttackActionConfig,
    EnvConfig,
    GameConfig,
    GroupConfig,
    InventoryRewards,
)


def make_arena(
    num_agents: int,
    combat: bool = True,
    map_builder: MapBuilderConfig | None = None,  # custom map builder; must match num_agents
) -> EnvConfig:
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
        rotate=ActionConfig(),
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
            root=metta.map.scenes.random.Random.factory(
                params=metta.map.scenes.random.Random.Params(
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

    return EnvConfig(
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
                    inventory=InventoryRewards(
                        heart=1,
                    ),
                ),
            ),
            groups={
                "agent": GroupConfig(
                    id=0,
                    sprite=0,
                    props=AgentConfig(),
                ),
            },
            map_builder=map_builder,
        ),
    )


def make_navigation(num_agents: int) -> EnvConfig:
    altar = building.altar.model_copy()
    altar.cooldown = 255  # Maximum cooldown
    altar.initial_resource_count = 1
    cfg = EnvConfig(
        game=GameConfig(
            num_agents=num_agents,
            objects={
                "altar": altar,
                "wall": building.wall,
            },
            actions=ActionsConfig(
                move=ActionConfig(),
                rotate=ActionConfig(),
                get_items=ActionConfig(),
            ),
            agent=AgentConfig(
                rewards=AgentRewards(
                    inventory=InventoryRewards(
                        heart=1,
                    ),
                ),
            ),
            # Always provide a concrete map builder config so tests can set width/height
            map_builder=RandomMapBuilder.Config(agents=num_agents),
        )
    )
    return cfg


def make_memory_sequence(num_agents: int = 1) -> EnvConfig:
    altar = building.altar.model_copy()
    altar.input_resources = {"battery_red": 1}
    altar.output_resources = {"heart": 1}
    altar.max_output = 1
    altar.conversion_ticks = 1
    altar.cooldown = 255
    altar.initial_resource_count = 0

    mine_red = building.mine_red.model_copy()
    mine_red.output_resources = {"ore_red": 1}
    mine_red.max_output = 1
    mine_red.conversion_ticks = 1
    mine_red.cooldown = 10
    mine_red.initial_resource_count = 1
    mine_red.color = 0

    generator_red = building.generator_red.model_copy()
    generator_red.input_resources = {"ore_red": 1}
    generator_red.output_resources = {"battery_red": 1}
    generator_red.max_output = 1
    generator_red.conversion_ticks = 1
    generator_red.cooldown = 10
    generator_red.initial_resource_count = 0
    generator_red.color = 0

    return EnvConfig(
        game=GameConfig(
            num_agents=num_agents,
            max_steps=150,
            objects={
                "altar": altar,
                "mine_red": mine_red,
                "mine_green": building.mine_green,
                "generator_red": generator_red,
                "wall": building.wall,
            },
            actions=ActionsConfig(
                move=ActionConfig(),
                rotate=ActionConfig(),
                put_items=ActionConfig(),
                get_items=ActionConfig(),
            ),
            agent=AgentConfig(
                rewards=AgentRewards(
                    inventory=InventoryRewards(heart=1, ore_red=0, battery_red=0),
                ),
            ),
        )
    )
