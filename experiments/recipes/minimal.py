"""Minimal recipe example - just define env_recipe() and the system creates defaults for all tools."""

import metta.mettagrid.builder.envs as eb
from metta.mettagrid.mettagrid_config import MettaGridConfig
from metta.eval.analysis_config import AnalysisConfig


def env_recipe(num_agents: int = 4) -> MettaGridConfig:
    """The only required function - creates the base environment.

    From just this, the system can automatically create:
    - train: Creates env-only curriculum and trainer
    - play/replay: Wraps in SimulationConfig
    - evaluate: Wraps in list of SimulationConfig
    """
    env = eb.make_arena(num_agents=num_agents)

    # Simple customization
    env.game.max_steps = 500
    env.game.agent.rewards.inventory["heart"] = 1
    env.label = "minimal"

    return env


def analyze_recipe(eval_db_uri: str, policy_uri: str | None = None) -> AnalysisConfig:
    """Analysis configuration - auto-created from env_recipe."""

    return AnalysisConfig(
        eval_db_uri=eval_db_uri,
        policy_uri=policy_uri,
        metrics=["*"],
    )
