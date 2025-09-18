"""Minimal recipe example - just define default_recipe_mettagrid() and the system creates defaults for all tools."""

import metta.mettagrid.builder.envs as eb
from metta.mettagrid.mettagrid_config import MettaGridConfig


def default_recipe_mettagrid(num_agents: int = 4) -> MettaGridConfig:
    """The only required function - creates the base environment.

    From just this, the system can automatically create:
    - train: Creates env-only curriculum and trainer
    - play/replay: Wraps in SimulationConfig
    - sim/evaluate: Wraps in SimulationConfig list
    """
    env = eb.make_arena(num_agents=num_agents)

    # Simple customization
    env.game.max_steps = 500
    env.game.agent.rewards.inventory["heart"] = 1

    return env
