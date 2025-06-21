from typing import Any

from pydantic import Field

from metta.util.types.base_config import BaseConfig, config_registry


class SimulationBaseConfig(BaseConfig):
    """
    Base simulation parameters shared by all simulations.

    This corresponds to the sim.yaml configuration that other configs inherit from.

    Example YAML (sim.yaml):
        num_episodes: 1
        max_time_s: 60
        env_overrides: {}
    """

    __init__ = BaseConfig.__init__  # For proper IDE support

    num_episodes: int = 1
    max_time_s: float = 60.0
    env_overrides: dict[str, Any] = Field(default_factory=dict)


# Register configuration classes with the global registry
config_registry.register("simulation_base", SimulationBaseConfig)
