from typing import Optional, Union

import numpy as np
from omegaconf import DictConfig, ListConfig
from omegaconf.omegaconf import OmegaConf

from metta.util.config import config_from_path
from mettagrid.mettagrid_env import MettaGridEnv


class MettaGridEnvSet(MettaGridEnv):
    """
    A wrapper around MettaGridEnv that allows for multiple configurations to be used for training.

    This class overrides the base method "_resolve_original_cfg" to choose from a list of options.

    ex:
        _target_: mettagrid.mettagrid_env.MettaGridEnvSet

        envs:
        - /env/mettagrid/simple
        - /env/mettagrid/bases

        probabilities:
        - 0.5
        - 0.5

    """

    def __init__(
        self,
        cfg: Union[DictConfig, ListConfig],
        render_mode: Optional[str] = None,
        buf=None,
        **kwargs,
    ):
        """
        Initialize a MettaGridEnvSet.

        Args:
            cfg: provided OmegaConf configuration
                - cfg.env should provide sub-configurations

            weights: weights for selecting environments.
                - Will be normalized to sum to 1.
                - If None, uniform distribution will be used.

            render_mode: Mode for rendering the environment
            buf: Buffer for Pufferlib
            **kwargs: Additional arguments passed to parent classes
        """

        self._original_cfg_paths = list(cfg.envs.keys())
        weights = list(cfg.envs.values())

        # Validate that all environments have the same agent count
        first_env_cfg = config_from_path(self._original_cfg_paths[0])
        num_agents = first_env_cfg.game.num_agents
        action_space = first_env_cfg.game.actions

        # Improve error message with specific environment information
        for env_path in self._original_cfg_paths:
            env_cfg = config_from_path(env_path)
            if env_cfg.game.num_agents != num_agents:
                raise ValueError(
                    "For MettaGridEnvSet, the number of agents must be the same in all environments. "
                    f"Environment '{env_path}' has {env_cfg.game.num_agents} agents, but expected {num_agents} "
                    f"(from first environment '{self._original_cfg_paths[0]}')"
                )
            if env_cfg.game.actions != action_space:
                raise ValueError(
                    "For MettaGridEnvSet, the action space must be the same in all environments. "
                    f"Environment '{env_path}' has {env_cfg.game.actions}, but expected {action_space} "
                    f"(from first environment '{self._original_cfg_paths[0]}')"
                )

        # Handle probabilities/weights
        if weights is None:
            # Use uniform distribution if no probabilities provided
            self._probabilities = [1.0 / len(self._original_cfg_paths)] * len(self._original_cfg_paths)
        else:
            # Check that probabilities match the number of environments
            if len(weights) != len(self._original_cfg_paths):
                raise ValueError(
                    f"Number of weights ({len(weights)}) must match "
                    f"number of environments ({len(self._original_cfg_paths)})"
                )

            if any(p < 0 for p in weights):
                raise ValueError("All weights must be non-negative")

            # Normalize weights to probabilities
            total = sum(weights)
            if total == 0:
                raise ValueError("Sum of weights cannot be zero")
            self._probabilities = [p / total for p in weights]

        super().__init__(cfg, render_mode=render_mode, buf=buf, **kwargs)

        # start with a random config from the set
        self.active_cfg = self._resolve_original_cfg()

    def _resolve_original_cfg(self):
        """
        Select a random configuration based on probabilities.

        Returns:
            A resolved environment configuration

        Raises:
            ValueError: If the number of agents in the selected environment
                       doesn't match the global number of agents
        """
        selected_path = np.random.choice(self._original_cfg_paths, p=self._probabilities)
        cfg = config_from_path(selected_path)
        cfg = OmegaConf.create(cfg)

        # Insert stats into the configuration
        cfg = self._insert_progress_into_cfg(cfg)

        OmegaConf.resolve(cfg)
        return cfg
