# metta/sim/simulation_config.py


from pathlib import Path
from typing import Dict, Optional

from pydantic import model_validator

from metta.util.config import Config


class SimulationConfig(Config):
    """Configuration for a single simulation run."""

    # Core simulation config
    env: str
    device: str
    num_envs: int
    num_episodes: int

    # Where individual stats are written
    # If not provided, stats are written to a temporary directory
    stats_dir: Optional[Path] = None

    npc_policy_uri: Optional[str] = None
    env_overrides: Optional[dict] = None
    policy_agents_pct: float = 1.0
    max_time_s: int = 60
    vectorization: str = "serial"
    eval_db_uri: Optional[str] = None


class SimulationSuiteConfig(SimulationConfig):
    """A suite of named simulations, with suite-level defaults injected."""

    name: str
    run_dir: str
    simulations: Dict[str, SimulationConfig]
    # —— don't need env bc all the simulations will specify ——
    env: Optional[str] = None

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
