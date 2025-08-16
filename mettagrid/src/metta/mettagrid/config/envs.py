from metta.map.mapgen import MapGenConfig
from metta.mettagrid.config import building
from metta.mettagrid.map_builder.random import RandomMapBuilderConfig
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
        move_8way=ActionConfig(),
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

    return EnvConfig(
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
            map_builder=MapGenConfig(
                num_agents=num_agents,
                width=25,
                height=25,
                instances=num_agents // 6,
                border_width=6,
                instance_border_width=0,
                root={
                    "type": "metta.map.scenes.random.Random",
                    "params": {
                        "agents": 6,
                        "objects": {
                            "wall": 20,
                            "altar": 5,
                            "mine_red": 10,
                            "generator_red": 5,
                            "lasery": 1,
                            "armory": 1,
                        },
                    },
                },
            ),
        )
    )


def make_nav(num_agents: int) -> EnvConfig:
    altar = building.altar
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
                        heart=0.333,
                    ),
                ),
            ),
            map_builder=RandomMapBuilderConfig(
                agents=num_agents,
                width=25,
                height=25,
                border_object="wall",
                border_width=1,
            ),
        )
    )
    return cfg
