from typing import Any, Generator, Optional, Tuple

from pydantic import field_validator, model_validator

from metta.sim.types.simulation_base_config import SimulationBaseConfig
from metta.util.types.base_config import BaseConfig, config_registry
from mettagrid.types.environment_config import EnvironmentConfig


class SimulationConfig(SimulationBaseConfig):
    """
    Configuration for simulations.

    Inherits base simulation parameters from SimulationBaseConfig and adds
    suite-specific configuration. All simulations use the same pattern - a named
    suite with a simulations dictionary, even if there's only one simulation.

    Each simulation entry represents environment configuration overrides that will
    be merged with the base environment config loaded from the 'env' path.

    Override hierarchy (later overrides earlier):
    1. Base environment config (from env path)
    2. Global env_overrides (from SimulationConfig)
    3. Simulation-specific overrides

    Example YAML:
        defaults:
          - simulation_base  # loads from simulation_base.yaml
          - _self_

        # overrides to simulation_base
        max_time_s: 30
        env_overrides: # apply to all simulation environments
          game:
            use_observation_tokens: true

        name: my_suite
        simulations:
          my_sim:
            env: env/mettagrid/simple
            # simulation-specific overrides:
            policy_agents_pct: 0.5
            game:
              number_of_agents: 4  # Merges with global game config
    """

    __init__ = BaseConfig.__init__  # For proper IDE support

    # Hydra defaults
    defaults: Optional[list[str]] = None

    # Suite identification
    name: str

    # Simulation definitions - each is a dict of environment config overrides
    simulations: dict[str, dict[str, Any]]

    @field_validator("defaults")
    @classmethod
    def validate_defaults(cls, v: Optional[list[str]]) -> Optional[list[str]]:
        """Validate Hydra defaults list follows best practices."""
        if v is None:
            return v

        if len(v) == 0:
            raise ValueError("defaults list cannot be empty")

        # Check that _self_ is last (Hydra best practice)
        if v[-1] != "_self_":
            raise ValueError(
                "defaults list should end with '_self_' to ensure local config takes precedence over included configs"
            )

        return v

    @field_validator("name")
    @classmethod
    def validate_name(cls, v: str) -> str:
        """Ensure suite name is specified."""
        if v == "???" or not v:
            raise ValueError("Suite name must be specified (got '???')")
        return v

    @field_validator("simulations")
    @classmethod
    def validate_simulations(cls, v: dict[str, dict[str, Any]]) -> dict[str, dict[str, Any]]:
        """Validate simulations dictionary structure."""
        if not v:
            raise ValueError("Simulations dictionary cannot be empty")

        # Validate each simulation entry
        for sim_name, sim_config in v.items():
            if not isinstance(sim_config, dict):
                raise ValueError(f"Simulation '{sim_name}' must be a dictionary")

            # Each simulation must have an 'env' key
            if "env" not in sim_config:
                raise ValueError(f"Simulation '{sim_name}' missing required 'env' field")

            if not sim_config["env"] or sim_config["env"] == "???":
                raise ValueError(f"Simulation '{sim_name}' has invalid env: {sim_config['env']}")

        return v

    @model_validator(mode="after")
    def validate_as_environment_configs(self) -> "SimulationConfig":
        """
        Validate that each simulation entry can form a valid EnvironmentConfig.

        This is a post-validation step that ensures the simulation overrides
        are compatible with EnvironmentConfig schema.
        """
        for sim_name, sim_overrides in self.simulations.items():
            try:
                # Extract just the override fields (not 'env')
                overrides = {k: v for k, v in sim_overrides.items() if k != "env"}

                # Combine with env_overrides for validation
                combined_overrides = {}
                if self.env_overrides:
                    combined_overrides = self._deep_merge(combined_overrides, self.env_overrides)
                if overrides:
                    combined_overrides = self._deep_merge(combined_overrides, overrides)

                # Try to create an EnvironmentConfig with the combined overrides
                # This validates that all override fields are valid for EnvironmentConfig
                _ = EnvironmentConfig(**combined_overrides)
            except Exception as e:
                # If we can't create a valid EnvironmentConfig, provide helpful error
                raise ValueError(f"Simulation '{sim_name}' has invalid environment configuration: {e}") from e

        return self

    @property
    def is_single(self) -> bool:
        """Check if this suite contains only one simulation."""
        return len(self.simulations) == 1

    def get_simulation_names(self) -> list[str]:
        """Get list of simulation names in the suite."""
        return list(self.simulations.keys())

    def get_simulation_env(self, name: str) -> str:
        """
        Get environment path for a simulation.

        Args:
            name: Simulation name

        Returns:
            Environment path string

        Raises:
            KeyError: If simulation name not found
        """
        if name not in self.simulations:
            raise KeyError(f"Simulation '{name}' not found in suite '{self.name}'")
        return self.simulations[name]["env"]

    def get_simulation_overrides(self, name: str) -> dict[str, Any]:
        """
        Get environment config overrides for a simulation.

        Args:
            name: Simulation name

        Returns:
            Dictionary of environment config overrides (excluding 'env' field)
        """
        if name not in self.simulations:
            raise KeyError(f"Simulation '{name}' not found in suite '{self.name}'")

        # Return all fields except 'env' as these are the overrides
        overrides = {k: v for k, v in self.simulations[name].items() if k != "env"}
        return overrides

    def get_combined_overrides(self, name: str) -> dict[str, Any]:
        """
        Get combined overrides for a simulation (env_overrides + simulation-specific).

        Args:
            name: Simulation name

        Returns:
            Dictionary with all overrides merged in proper order
        """
        combined = {}

        # First apply global env_overrides
        if self.env_overrides:
            combined = self._deep_merge(combined, self.env_overrides)

        # Then apply simulation-specific overrides
        sim_overrides = self.get_simulation_overrides(name)
        if sim_overrides:
            combined = self._deep_merge(combined, sim_overrides)

        return combined

    def get_simulation_config(self, name: str) -> dict[str, Any]:
        """
        Get full configuration for a simulation.

        Merges the base simulation parameters with simulation-specific config.

        Args:
            name: Simulation name

        Returns:
            Dictionary with simulation configuration including:
            - env: The environment config path
            - num_episodes, max_time_s, env_overrides from SimulationBase
            - Any additional overrides from the simulation entry
        """
        if name not in self.simulations:
            raise KeyError(f"Simulation '{name}' not found in suite '{self.name}'")

        # Start with base simulation parameters
        sim_config = {
            "env": self.simulations[name]["env"],
            "num_episodes": self.num_episodes,
            "max_time_s": self.max_time_s,
            "env_overrides": {**self.env_overrides},  # Copy to avoid mutation
        }

        # Add all fields from the simulation config (including env config overrides)
        for key, value in self.simulations[name].items():
            if key not in sim_config:
                sim_config[key] = value

        return sim_config

    def build_environment_config(self, name: str) -> EnvironmentConfig:
        """
        Build a complete EnvironmentConfig for a simulation.

        This method applies a three-level override hierarchy:
        1. Base environment config (loaded from 'env' path)
        2. Global env_overrides from SimulationConfig
        3. Simulation-specific overrides

        Args:
            name: Simulation name

        Returns:
            Complete EnvironmentConfig with all overrides applied
        """
        if name not in self.simulations:
            raise KeyError(f"Simulation '{name}' not found in suite '{self.name}'")

        env_path = self.simulations[name]["env"]

        # Get combined overrides (env_overrides + simulation-specific)
        combined_overrides = self.get_combined_overrides(name)

        # Load the base environment config and apply combined overrides
        env_config = EnvironmentConfig.from_hydra_path(env_path, combined_overrides)

        return env_config

    def iter_simulations(self) -> Generator[Tuple[str, dict[str, Any]], None, None]:
        """
        Iterate over all simulations with their full configurations.

        Yields:
            Tuple of (simulation_name, simulation_config)
        """
        for name in self.get_simulation_names():
            yield name, self.get_simulation_config(name)

    def _deep_merge(self, base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
        """
        Deep merge two dictionaries, with override taking precedence.

        Args:
            base: Base dictionary
            override: Override dictionary

        Returns:
            Merged dictionary
        """
        result = base.copy()

        for key, value in override.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                # Recursively merge nested dictionaries
                result[key] = self._deep_merge(result[key], value)
            else:
                # Override the value
                result[key] = value

        return result


# Register configuration class with the global registry
config_registry.register("simulation", SimulationConfig)
