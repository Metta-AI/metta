# metta/sim/simulation_config.py

from typing import Dict, Literal, Optional

from pydantic import Field

from metta.common.util.config import Config
from metta.mettagrid import EnvConfig


class SimulationConfig(Config):
    """Configuration for a single simulation run."""

    # Core simulation config
    num_episodes: int = Field(default=1, description="Number of episodes to run", ge=1)
    max_time_s: int = Field(default=120, description="Maximum time in seconds to run the simulation", ge=0)

    npc_policy_uri: Optional[str] = Field(default=None, description="URI of the policy to use for NPC agents")
    policy_agents_pct: float = Field(default=1.0, description="pct of agents to be controlled by policies", ge=0, le=1)


class SingleEnvSimulationConfig(SimulationConfig):
    """Configuration for a single simulation run."""

    type: Literal["single"] = Field(default="single", description="Type discriminator for SingleEnvSimulationConfig")
    env: EnvConfig
    name: str = Field(description="Name of the simulation")


class SimulationSuiteConfig(SimulationConfig):
    """A suite of named simulations, with suite-level defaults injected."""

    type: Literal["suite"] = Field(default="suite", description="Type discriminator for SimulationSuiteConfig")
    name: str = Field(description="Name of the simulation suite")
    simulations: Dict[str, SingleEnvSimulationConfig] = Field(description="Simulations to run")
    episode_tags: list[str] = Field(default=[], description="Tags to add to each episode")
