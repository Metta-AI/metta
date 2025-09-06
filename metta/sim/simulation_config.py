# metta/sim/simulation_config.py

from typing import Optional, Union

from pydantic import Field

from metta.mettagrid import MettaGridConfig
from metta.mettagrid.config import Config
from metta.sim.env_config import TribalEnvConfig


class SimulationConfig(Config):
    """Configuration for a single simulation run."""

    name: str = Field(description="Name of the simulation")
    env: Union[MettaGridConfig, TribalEnvConfig]  # Accept both MettaGrid and Tribal configs

    # Core simulation config
    num_episodes: int = Field(default=1, description="Number of episodes to run", ge=1)
    max_time_s: int = Field(default=120, description="Maximum time in seconds to run the simulation", ge=0)

    npc_policy_uri: Optional[str] = Field(default=None, description="URI of the policy to use for NPC agents")
    policy_agents_pct: float = Field(default=1.0, description="pct of agents to be controlled by policies", ge=0, le=1)

    episode_tags: Optional[list[str]] = Field(default=None, description="Tags to add to each episode")
