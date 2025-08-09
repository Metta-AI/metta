from metta.mettagrid.config import object
from metta.mettagrid.mettagrid_config import (
    EnvConfig,
    PyActionConfig,
    PyActionsConfig,
    PyAgentConfig,
    PyAgentRewards,
    PyGameConfig,
    PyGroupConfig,
    PyInventoryRewards,
)

resources = [
    "ore_red",
    "ore_blue",
    "ore_green",
    "battery_red",
    "battery_blue",
    "battery_green",
    "heart",
    "armor",
    "laser",
    "blueprint",
]

objects = {
    "altar": object.altar,
    "mine_red": object.mine_red,
    "mine_blue": object.mine_blue,
    "mine_green": object.mine_green,
    "generator_red": object.generator_red,
    "generator_blue": object.generator_blue,
    "generator_green": object.generator_green,
}


def arena(
    num_agents: int,
) -> EnvConfig:
    return EnvConfig(
        game=PyGameConfig(
            num_agents=num_agents,
            actions=PyActionsConfig(
                noop=PyActionConfig(
                    enabled=True,
                )
            ),
            inventory_item_names=resources,
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
                "solo": PyGroupConfig(
                    id=0,
                    sprite=0,
                    props=PyAgentConfig(),
                ),
            },
        )
    )
