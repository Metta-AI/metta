"""Simple recipe that only defines mettagrid_recipe.

This demonstrates that a recipe can define just a mettagrid_recipe function
and the system will automatically convert it to the appropriate tool config.
"""

from mettagrid.builder.envs import make_arena
from mettagrid.config.mettagrid_config import MettaGridConfig


def mettagrid_recipe() -> MettaGridConfig:
    """Single recipe function that works for all tools (except analyze).

    The system will automatically wrap this in:
    - TrainTool for 'train simple'
    - PlayTool for 'play simple'
    - ReplayTool for 'replay simple'
    - SimTool for 'evaluate simple' or 'sim simple'
    """
    env = make_arena(num_agents=8)
    env.label = "simple"
    return env
