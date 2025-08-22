import metta.map.scenes.random
from metta.map.mapgen import MapGen
from metta.mettagrid.config import building, empty_converters
from metta.mettagrid.map_builder.map_builder import MapBuilderConfig
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
            # always override the map builder
            map_builder={},
        )
    )
    return cfg
