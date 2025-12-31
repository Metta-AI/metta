# metta/sim/simulation_config.py


from pydantic import Field

from metta.rl.slot_config import PolicySlotConfig
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
    policy_slots: list[PolicySlotConfig] | None = Field(
        default=None, description="Optional list of policy slots for evaluation/simulation."
    )
    agent_slot_map: list[str] | None = Field(
        default=None,
        description="Optional mapping (length=num_agents) assigning each agent index to a policy slot id.",
    )

    @property
    def full_name(self) -> str:
        return f"{self.suite}/{self.name}"

    def to_simulation_run_config(self) -> SimulationRunConfig:
        return SimulationRunConfig(
            env=self.env,
            num_episodes=self.num_episodes,
            episode_tags={"name": self.name, "category": self.suite},
            policy_slots=self.policy_slots,
            agent_slot_map=self.agent_slot_map,
        )
