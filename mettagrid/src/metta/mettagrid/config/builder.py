from metta.mettagrid.config import object
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


def arena(
    num_agents: int,
    combat: bool = False,
) -> EnvConfig:
    objects = {
        "wall": object.wall,
        "altar": object.altar,
        "mine_red": object.mine_red,
        "generator_red": object.generator_red,
        "lasery": object.lasery,
        "armory": object.armory,
    }

    actions = ActionsConfig(
        noop=ActionConfig(),
        move=ActionConfig(),
        move_8way=ActionConfig(),
        move_cardinal=ActionConfig(),
        rotate=ActionConfig(),
        put_items=ActionConfig(),
        get_items=ActionConfig(),
        attack=AttackActionConfig(
            required_resources={
                "laser": 1,
            },
            defense_resources={
                "armor": 1,
            },
        ),
        swap=ActionConfig(),
        change_color=ActionConfig(),
    )

    if not combat:
        actions.attack.required_resources = {"laser": 100}

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
        )
    )
