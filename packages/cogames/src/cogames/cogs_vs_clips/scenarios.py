from mettagrid.config.mettagrid_config import (
    ActionConfig,
    ActionsConfig,
    AgentConfig,
    AgentRewards,
    GameConfig,
    MettaGridConfig,
    WallConfig,
)
from mettagrid.map_builder.random import RandomMapBuilder


def make_cogs_vs_clips_scenario() -> MettaGridConfig:
    return MettaGridConfig(
        game=GameConfig(
            num_agents=2,
            actions=ActionsConfig(
                move=ActionConfig(),
                noop=ActionConfig(),
                rotate=ActionConfig(),
            ),
            objects={"wall": WallConfig(type_id=1)},
            map_builder=RandomMapBuilder.Config(
                width=10,
                height=10,
                agents=2,
                seed=42,
            ),
            agent=AgentConfig(
                default_resource_limit=10,
                resource_limits={"heart": 10},
                rewards=AgentRewards(
                    inventory={},
                ),
            ),
        )
    )
