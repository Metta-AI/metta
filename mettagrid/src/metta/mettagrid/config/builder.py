from metta.mettagrid.config import object
from metta.mettagrid.mettagrid_config import (
    EnvConfig,
    PyActionConfig,
    PyActionsConfig,
    PyAgentConfig,
    PyAgentRewards,
    PyAttackActionConfig,
    PyGameConfig,
    PyGroupConfig,
    PyInventoryRewards,
)


def arena(
    num_agents: int,
    combat: bool = False,
) -> EnvConfig:
    objects = {
        "altar": object.altar,
        "mine_red": object.mine_red,
        "generator_red": object.generator_red,
    }

    actions = PyActionsConfig(
        noop=PyActionConfig(),
        move=PyActionConfig(),
        move_8way=PyActionConfig(),
        move_cardinal=PyActionConfig(),
        rotate=PyActionConfig(),
        put_items=PyActionConfig(),
        get_items=PyActionConfig(),
    )

    if combat:
        objects["lasery"] = object.lasery
        objects["armory"] = object.armory

        actions.attack = PyAttackActionConfig(
            required_resources={
                "laser": 1,
            },
            defense_resources={
                "armor": 1,
            },
        )

    return EnvConfig(
        game=PyGameConfig(
            num_agents=num_agents,
            actions=actions,
            objects=objects,
            agent=PyAgentConfig(
                default_resource_limit=50,
                resource_limits={
                    "heart": 255,
                },
                rewards=PyAgentRewards(
                    inventory=PyInventoryRewards(
                        heart=1,
                    ),
                ),
            ),
            groups={
                "agent": PyGroupConfig(
                    id=0,
                    sprite=0,
                    props=PyAgentConfig(),
                ),
            },
        )
    )
