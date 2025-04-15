from dataclasses import dataclass, field, fields
from typing import Dict, Optional

from omegaconf import MISSING, DictConfig


@dataclass
class SimulationConfig:
    """Configuration for a Metta simulation run."""

    # Required parameters
    run_id: str = MISSING
    env: str = MISSING

    # Optional parameters with defaults
    device: str = "cuda"
    npc_policy_uri: Optional[str] = None
    env_overrides: Optional[DictConfig] = None
    policy_agents_pct: float = 1.0
    num_envs: int = 50
    num_episodes: int = 50
    max_time_s: int = 60
    vectorization: str = "serial"
    eval_db_uri: Optional[str] = None


@dataclass
class SimulationSuiteConfig(SimulationConfig):
    # Named simulation configs
    simulations: Dict[str, SimulationConfig] = field(default_factory=dict)

    def apply_defaults_to_simulations(self):
        """
        Apply suite-wide defaults to simulations that don't specify values.
        Uses inheritance to automatically propagate all SimulationConfig fields.
        """
        # Iterate through all simulation configs
        for _, sim_config in self.simulations.items():
            # For each field in the parent SimulationConfig class
            for field_obj in fields(SimulationConfig):
                field_name = field_obj.name

                # Skip if simulation explicitly sets this value to something other than MISSING
                sim_value = getattr(sim_config, field_name)
                if sim_value is not MISSING:
                    continue

                # Apply suite's value as default
                suite_value = getattr(self, field_name)
                if suite_value is not MISSING:
                    setattr(sim_config, field_name, suite_value)
