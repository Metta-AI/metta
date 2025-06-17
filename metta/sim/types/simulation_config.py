from typing import Optional

from metta.util.types.base_config import BaseConfig


class SimulationConfig(BaseConfig):
    """Configuration for a single simulation run."""

    __init__ = BaseConfig.__init__

    num_episodes: int
    max_time_s: int = 120
    env_overrides: dict = {}

    npc_policy_uri: Optional[str] = None
    policy_agents_pct: float = 1.0
