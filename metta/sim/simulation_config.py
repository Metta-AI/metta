# metta/sim/simulation_config.py

from typing import Optional

from pydantic import Field

from metta.common.util.config import Config
from metta.mettagrid import EnvConfig


class SimulationConfig(Config):
    """Configuration for a single simulation run."""

    name: str = Field(description="Name of the simulation")
    env: EnvConfig

    # Core simulation config
    num_episodes: int = Field(default=1, description="Number of episodes to run", ge=1)
    max_time_s: int = Field(default=120, description="Maximum time in seconds to run the simulation", ge=0)

    npc_policy_uri: Optional[str] = Field(default=None, description="URI of the policy to use for NPC agents")
    policy_agents_pct: float = Field(default=1.0, description="pct of agents to be controlled by policies", ge=0, le=1)
