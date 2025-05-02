# metta/sim/simulation_config.py

import os
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
    # Path to save replay, can be a local path or s3:// URL
    replay_path: Optional[str] = None


class SimulationSuiteConfig(SimulationConfig):
    """A suite of named simulations, with suite-level defaults injected."""

    name: str
    run_dir: str
    simulations: Dict[str, SimulationConfig]
    # —— don't need env bc all the simulations will specify ——
    env: Optional[str] = None
    # If set, we will set up replay paths for all simulations in this directory
    replay_dir: Optional[str] = None

    @model_validator(mode="before")
    def propagate_defaults(cls, values: dict) -> dict:
        # collect any suite-level overrides that are present & non-None
        suite_defaults = {
            k: v for k, v in values.items() if k in ("env", "device", "num_envs", "num_episodes") and v is not None
        }
        raw_sims = values.get("simulations", {}) or {}
        merged: Dict[str, dict] = {}
        for name, sim_cfg in raw_sims.items():
            # sim_cfg is a dict; override only where sim_cfg provides a key
            merged[name] = {**suite_defaults, **sim_cfg}
        values["simulations"] = merged
        return values

    def propagate_replay_paths(self):
        """
        Projects the suite-level replay_dir to individual simulation replay_paths.
        Note that those individual simulation replaypaths also can get converted
        into separate paths for each env and episode.
        """
        if self.replay_dir is None:
            return

        for name, sim_config in self.simulations.items():
            if sim_config.replay_path is None:  # Only set if not already specified
                if self.replay_dir.startswith("s3://"):
                    sim_config.replay_path = f"{self.replay_dir.rstrip('/')}/{name}/replay.json.z"
                else:
                    sim_config.replay_path = os.path.join(self.replay_dir, name, "replay.json.z")
