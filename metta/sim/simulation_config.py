from dataclasses import dataclass
from typing import Dict, Optional

from omegaconf import DictConfig

from metta.util.config import propagate_cfg


@dataclass(kw_only=True)
class SimulationConfig:
    """Configuration for a Metta simulation run."""

    # Required parameters
    env: str
    device: str

    # Optional parameters with defaults
    npc_policy_uri: Optional[str] = None
    env_overrides: Optional[DictConfig] = None
    policy_agents_pct: float = 1.0
    num_envs: int = 50
    num_episodes: int = 50
    max_time_s: int = 60
    vectorization: str = "serial"
    eval_db_uri: Optional[str] = None


@dataclass(kw_only=True)
class SimulationSuiteConfig(SimulationConfig):
    simulations: Dict[str, SimulationConfig]
    run_dir: str

    @classmethod
    def __preprocess_dictconfig__(cls, cfg_dict: dict) -> dict:
        """
        Copy all fields defined on `SimulationConfig` from the suite node
        into each entry of `suite.simulations` that doesn’t already have it, allowing for
        suite-wide defaults.
        """
        # parent mapping is cfg_dict itself; children mapping is cfg_dict["simulations"]
        propagate_cfg(cfg_dict, cfg_dict.get("simulations", {}), SimulationConfig)
        if getattr(cfg_dict, "env", None) is None:
            cfg_dict["env"] = ""  # Allow for empty env for the simulation suite
        return cfg_dict
