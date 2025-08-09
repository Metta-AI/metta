# metta/sim/simulation_config.py

import json
from typing import Any, Dict, Optional

from omegaconf import DictConfig, OmegaConf
from pydantic import BaseModel, model_validator

from metta.common.util.config import Config


def _to_jsonable(obj):
    """Convert nested objects (Pydantic, OmegaConf) to plain JSON-safe containers."""
    if isinstance(obj, BaseModel):
        # Dump to python first to avoid JSON serialization on unknown inner types
        python_obj = obj.model_dump(mode="python")
        return _to_jsonable(python_obj)
    if isinstance(obj, DictConfig):
        return OmegaConf.to_container(obj, resolve=True)
    if isinstance(obj, dict):
        return {k: _to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_jsonable(v) for v in obj]
    return obj


class SimulationConfig(Config):
    """Configuration for a single simulation run."""

    __init__ = Config.__init__

    # Core simulation config
    num_episodes: int
    max_time_s: int = 120
    env_overrides: dict = {}

    npc_policy_uri: Optional[str] = None
    policy_agents_pct: float = 1.0


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

    # ------------------------------------------------------------------
    # Serialization helpers for remote execution
    # ------------------------------------------------------------------
    def to_jsonable(self) -> dict[str, Any]:
        return _to_jsonable(self)

    @classmethod
    def from_json(cls, json_str: str) -> "SimulationSuiteConfig":
        """Create a SimulationSuiteConfig from a JSON string."""
        data = json.loads(json_str)
        if not isinstance(data, dict):
            raise ValueError("Expected JSON object for SimulationSuiteConfig")
        return cls.model_validate(data)  # type: ignore[return-value]
