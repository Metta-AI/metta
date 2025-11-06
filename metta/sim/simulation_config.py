# metta/sim/simulation_config.py

import typing

import pydantic

import mettagrid
import mettagrid.base_config


class SimulationConfig(mettagrid.base_config.Config):
    """Configuration for a single simulation run."""

    suite: str = pydantic.Field(description="Name of the simulation suite")
    name: str = pydantic.Field(description="Name of the simulation")
    env: mettagrid.MettaGridConfig

    # Core simulation config
    num_episodes: int = pydantic.Field(default=1, description="Number of episodes to run", ge=1)
    max_time_s: int = pydantic.Field(default=120, description="Maximum time in seconds to run the simulation", ge=0)

    npc_policy_uri: typing.Optional[str] = pydantic.Field(
        default=None, description="URI of the policy to use for NPC agents"
    )
    policy_agents_pct: float = pydantic.Field(
        default=1.0, description="pct of agents to be controlled by policies", ge=0, le=1
    )

    episode_tags: typing.Optional[list[str]] = pydantic.Field(default=None, description="Tags to add to each episode")

    @property
    def full_name(self) -> str:
        return f"{self.suite}/{self.name}"
