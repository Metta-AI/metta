# metta/sim/simulation_config.py

from typing import Dict, Optional

from pydantic import model_validator

from metta.util.config import Config


class SimulationConfig(Config):
    """Configuration for a single simulation run."""

    # Core simulation config
    num_episodes: int
    max_time_s: int = 120

    npc_policy_uri: Optional[str] = None
    policy_agents_pct: float = 1.0


class SingleEnvSimulationConfig(SimulationConfig):
    """Configuration for a single simulation run."""

    env: str
    env_overrides: Optional[dict] = None


class SimulationSuiteConfig(SimulationConfig):
    """A suite of named simulations, with suite-level defaults injected."""

    name: str
    simulations: Dict[str, SingleEnvSimulationConfig]

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
            # propagate suite values into each child
            merged[name] = {**explicitly_provided, **sim_cfg}

        values["simulations"] = merged
        return values
