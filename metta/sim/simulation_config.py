# metta/sim/simulation_config.py

from typing import Any

from pydantic import Field

from metta.sim.runner import SimulationRunConfig
from mettagrid import MettaGridConfig
from mettagrid.base_config import Config


class SimulationConfig(Config):
    """Configuration for a single simulation run."""

    suite: str = Field(description="Name of the simulation suite")
    name: str = Field(description="Name of the simulation")
    env: MettaGridConfig

    # Core simulation config
    num_episodes: int = Field(default=1, description="Number of episodes to run", ge=1)
    max_time_s: int = Field(default=120, description="Maximum time in seconds to run the simulation", ge=0)

    # Optional NPC/scripted agent settings (used by legacy flows and dual-policy experiments)
    npc_policy_uri: str | None = Field(default=None, description="URI of the policy to use for NPC agents")
    npc_policy_class: str | None = Field(default=None, description="Import path to scripted NPC policy class")
    npc_policy_kwargs: dict[str, Any] = Field(
        default_factory=dict, description="Keyword args for scripted NPC policy"
    )
    policy_agents_pct: float = Field(
        default=1.0, description="pct of agents to be controlled by policies", ge=0, le=1
    )

    # Optional episode metadata
    episode_tags: dict[str, str] | None = Field(
        default=None, description="Tag key/value pairs to attach to each episode"
    )

    @property
    def full_name(self) -> str:
        return f"{self.suite}/{self.name}"

    def to_simulation_run_config(self) -> SimulationRunConfig:
        """Convert to the runtime simulation config used by the new runner."""
        tags = dict(self.episode_tags or {})
        tags.setdefault("name", self.name)
        tags.setdefault("category", self.suite)
        return SimulationRunConfig(
            env=self.env,
            num_episodes=self.num_episodes,
            episode_tags=tags,
            # proportions/max_action_time_ms can be added here when needed
        )
