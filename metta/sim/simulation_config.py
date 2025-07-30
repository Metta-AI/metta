# metta/sim/simulation_config.py

from typing import Dict, Optional

from pydantic import model_validator

from metta.common.util.config import Config


class SimulationConfig(Config):
    """Configuration for a single simulation run."""

    __init__ = Config.__init__

    # Core simulation config
    num_episodes: int
    max_time_s: int = 120
    env_overrides: dict = {}

    npc_policy_uri: Optional[str] = None
    policy_agents_pct: float = 1.0

    # Doxascope config
    doxascope: Optional[dict] = None


class SingleEnvSimulationConfig(SimulationConfig):
    """Configuration for a single simulation run."""

    __init__ = SimulationConfig.__init__

    env: str
    env_overrides: dict = {}


class SimulationSuiteConfig(SimulationConfig):
    """A suite of named simulations, with suite-level defaults injected."""

    name: str
    simulations: Dict[str, SingleEnvSimulationConfig]
    episode_tags: list[str] = []

    @model_validator(mode="before")
    @classmethod
    def propagate_suite_fields(cls, values: dict) -> dict:
        # collect only fields that were explicitly passed (not defaults)
        # note: in `mode="before"`, `values` is raw user input

        explicitly_provided = {
            k: v
            for k, v in values.items()
            if k in SimulationConfig.model_fields  # only fields simulation children would know
        }
        raw_sims = values.get("simulations", {}) or {}
        merged: Dict[str, dict] = {}

        for name, sim_cfg in raw_sims.items():
            # Handle both dict and SingleEnvSimulationConfig instances
            if isinstance(sim_cfg, dict):
                # Raw dict - merge with suite defaults
                merged[name] = {**explicitly_provided, **sim_cfg}
            elif isinstance(sim_cfg, SingleEnvSimulationConfig):
                # Already instantiated - convert to dict and merge
                sim_dict = sim_cfg.model_dump()
                merged[name] = {**explicitly_provided, **sim_dict}
            else:
                # Pass through as-is and let Pydantic handle validation
                merged[name] = sim_cfg

        values["simulations"] = merged
        return values
