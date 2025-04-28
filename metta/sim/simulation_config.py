# metta/sim/simulation_config.py

from typing import Dict, Optional

from pydantic import model_validator

from metta.util.config import Config


class SimulationConfig(Config):
    """Configuration for a single simulation run."""
    env: str
    device: str
    num_envs: int
    num_episodes: int

    npc_policy_uri: Optional[str] = None
    env_overrides: Optional[dict] = None
    policy_agents_pct: float = 1.0
    max_time_s: int = 60
    vectorization: str = "serial"
    eval_db_uri: Optional[str] = None


class SimulationSuiteConfig(SimulationConfig):
    """A suite of named simulations, with suite-level defaults injected."""
    run_dir: str
    simulations: Dict[str, SimulationConfig]

    # —— don't need env bc all the simulations will specify —— 
    env: Optional[str] = None

    @model_validator(mode="before")
    def _propagate_defaults(cls, values: dict) -> dict:
        # collect any suite-level overrides that are present & non-None
        suite_defaults = {
            k: v for k, v in values.items()
            if k in ("env", "device", "num_envs", "num_episodes") and v is not None
        }

        raw_sims = values.get("simulations", {}) or {}
        merged: Dict[str, dict] = {}
        for name, sim_cfg in raw_sims.items():
            # sim_cfg is a dict; override only where sim_cfg provides a key
            merged[name] = {**suite_defaults, **sim_cfg}
        values["simulations"] = merged
        return values


