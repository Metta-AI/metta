import numpy as np
from omegaconf import DictConfig
from omegaconf.omegaconf import OmegaConf

from metta.util.config import config_from_path
from mettagrid.mettagrid_env import MettaGridEnv


class MettaGridEnvSet(MettaGridEnv):
    """
    This is a wrapper around MettaGridEnv that allows for multiple environments to be used for training
    with prioritized environment replay.
    """

    def __init__(
        self,
        env_cfg: DictConfig,
        render_mode: str,
        buf=None,
        alpha: float = 0.6,  # Priority exponent (controls how much prioritization is used)
        beta: float = 0.4,  # Initial importance sampling weight (increases to 1 over time)
        beta_annealing_steps: int = 100000,  # Number of episodes to anneal beta to 1.0
        epsilon: float = 0.01,  # Small constant to avoid zero priority
        **kwargs,
    ):
        self._env_cfgs = env_cfg.envs
        self._num_agents_global = env_cfg.num_agents
        self._num_envs = len(self._env_cfgs)

        # Prioritized replay parameters
        self._alpha = alpha
        self._beta = beta
        self._beta_annealing_steps = beta_annealing_steps
        self._beta_increment = (1.0 - beta) / beta_annealing_steps if beta_annealing_steps > 0 else 0
        self._epsilon = epsilon

        # Initialize tracking variables
        self._episode_count = 0
        self._current_env_idx = None
        self._env_priorities = np.ones(self._num_envs, dtype=np.float32)  # Initial priorities are uniform
        self._env_performance = np.zeros(self._num_envs, dtype=np.float32)  # Track performance for each env
        self._env_visits = np.zeros(self._num_envs, dtype=np.int32)  # Count how many times each env is visited

        # Get initial environment config
        self._env_cfg = self._get_new_env_cfg()

        super().__init__(env_cfg, render_mode, buf=buf, env_map=None, **kwargs)
        self._cfg_template = None  # we don't use this with multiple envs, so we clear it to emphasize that fact

    def _update_priorities(self, env_idx: int, performance: float):
        """
        Update the priority for a specific environment based on agent performance.

        Args:
            env_idx: Index of the environment
            performance: Performance metric (e.g., negative reward or TD error)
        """
        # Update performance tracking
        self._env_performance[env_idx] = performance

        # Update priority with absolute error (raised to power of alpha)
        self._env_priorities[env_idx] = (abs(performance) + self._epsilon) ** self._alpha

    def _get_env_probabilities(self):
        """
        Calculate probabilities for environment selection based on priorities.
        """
        # Normalize priorities to get probabilities
        total_priority = np.sum(self._env_priorities)
        return self._env_priorities / total_priority if total_priority > 0 else np.ones(self._num_envs) / self._num_envs

    def _get_importance_sampling_weight(self, env_idx: int):
        """
        Calculate importance sampling weight for the selected environment.
        """
        # Calculate probability of selecting this environment
        probs = self._get_env_probabilities()

        # Calculate importance sampling weight (to correct for bias)
        weight = (1.0 / (self._num_envs * probs[env_idx])) ** self._beta

        # Normalize weight by max weight to keep values reasonable
        max_weight = (1.0 / (self._num_envs * np.min(probs))) ** self._beta
        return weight / max_weight

    def _update_beta(self):
        """Anneal beta parameter towards 1.0 over time."""
        self._beta = min(1.0, self._beta + self._beta_increment)

    def _get_new_env_cfg(self):
        """
        Select an environment based on prioritized replay probabilities.
        """
        self._env_map = None
        # Get probabilities based on priorities
        probs = self._get_env_probabilities()

        # Select environment based on priorities
        env_idx = np.random.choice(self._num_envs, p=probs)
        self._current_env_idx = env_idx
        self._env_visits[env_idx] += 1

        # Get the environment configuration
        selected_env = self._env_cfgs[env_idx]
        env_cfg = config_from_path(selected_env)

        # Check consistency in number of agents
        if self._num_agents_global != env_cfg.game.num_agents:
            raise ValueError(
                "For MettaGridEnvSet, the number of agents must be the same for all environments. "
                f"Global: {self._num_agents_global}, Env: {env_cfg.game.num_agents}"
            )

        env_cfg = OmegaConf.create(env_cfg)
        OmegaConf.resolve(env_cfg)
        return env_cfg

    def reset(self, seed=None, options=None):
        """
        Reset the environment and update priorities based on performance in the previous episode.
        """
        # Update priorities based on last episode's performance if applicable
        if self._current_env_idx is not None:
            # Use a performance metric from the previous episode (e.g., negative reward)
            # The current implementation assumes we want to prioritize environments where
            # agents are struggling (lower rewards = higher priority)
            if hasattr(self, "_last_episode_reward"):
                # Use negative reward as error (higher error = higher priority)
                performance = -self._last_episode_reward
                self._update_priorities(self._current_env_idx, performance)

        # Increment episode counter and update beta
        self._episode_count += 1
        self._update_beta()

        # Standard reset procedure
        obs, infos = super().reset(seed, options)
        return obs, infos

    def step(self, actions):
        """
        Step the environment and track necessary information for prioritization.
        """
        observations, rewards, terminals, truncations, infos = super().step(actions)

        # Store information needed for prioritization
        if (terminals.all() or truncations.all()) and "episode/reward.mean" in infos:
            self._last_episode_reward = infos["episode/reward.mean"]

        return observations, rewards, terminals, truncations, infos

    def get_env_stats(self):
        """
        Return statistics about environment selection for logging/debugging.
        """
        return {
            "env_priorities": self._env_priorities.copy(),
            "env_probabilities": self._get_env_probabilities(),
            "env_visits": self._env_visits.copy(),
            "env_performance": self._env_performance.copy(),
            "beta": self._beta,
            "episode_count": self._episode_count,
        }


# from typing import Optional, Union

# import numpy as np
# from omegaconf import DictConfig, ListConfig
# from omegaconf.omegaconf import OmegaConf

# from metta.util.config import config_from_path
# from mettagrid.mettagrid_env import MettaGridEnv


# class MettaGridEnvSet(MettaGridEnv):
#     """
#     A wrapper around MettaGridEnv that allows for multiple configurations to be used for training.

#     This class overrides the base method "_resolve_original_cfg" to choose from a list of options.

#     ex:
#         _target_: mettagrid.mettagrid_env.MettaGridEnvSet

#         envs:
#         - /env/mettagrid/simple
#         - /env/mettagrid/bases

#         probabilities:
#         - 0.5
#         - 0.5

#     """

#     def __init__(
#         self,
#         cfg: Union[DictConfig, ListConfig],
#         render_mode: Optional[str] = None,
#         buf=None,
#         **kwargs,
#     ):
#         """
#         Initialize a MettaGridEnvSet.

#         Args:
#             cfg: provided OmegaConf configuration
#                 - cfg.env should provide sub-configurations

#             weights: weights for selecting environments.
#                 - Will be normalized to sum to 1.
#                 - If None, uniform distribution will be used.

#             render_mode: Mode for rendering the environment
#             buf: Buffer for Pufferlib
#             **kwargs: Additional arguments passed to parent classes
#         """

#         self._original_cfg_paths = list(cfg.envs.keys())
#         weights = list(cfg.envs.values())

#         # Validate that all environments have the same agent count
#         first_env_cfg = config_from_path(self._original_cfg_paths[0])
#         num_agents = first_env_cfg.game.num_agents
#         action_space = first_env_cfg.game.actions

#         # Improve error message with specific environment information
#         for env_path in self._original_cfg_paths:
#             env_cfg = config_from_path(env_path)
#             if env_cfg.game.num_agents != num_agents:
#                 raise ValueError(
#                     "For MettaGridEnvSet, the number of agents must be the same in all environments. "
#                     f"Environment '{env_path}' has {env_cfg.game.num_agents} agents, but expected {num_agents} "
#                     f"(from first environment '{self._original_cfg_paths[0]}')"
#                 )
#             if env_cfg.game.actions != action_space:
#                 raise ValueError(
#                     "For MettaGridEnvSet, the action space must be the same in all environments. "
#                     f"Environment '{env_path}' has {env_cfg.game.actions}, but expected {action_space} "
#                     f"(from first environment '{self._original_cfg_paths[0]}')"
#                 )

#         # Handle probabilities/weights
#         if weights is None:
#             # Use uniform distribution if no probabilities provided
#             self._probabilities = [1.0 / len(self._original_cfg_paths)] * len(self._original_cfg_paths)
#         else:
#             # Check that probabilities match the number of environments
#             if len(weights) != len(self._original_cfg_paths):
#                 raise ValueError(
#                     f"Number of weights ({len(weights)}) must match "
#                     f"number of environments ({len(self._original_cfg_paths)})"
#                 )

#             if any(p < 0 for p in weights):
#                 raise ValueError("All weights must be non-negative")

#             # Normalize weights to probabilities
#             total = sum(weights)
#             if total == 0:
#                 raise ValueError("Sum of weights cannot be zero")
#             self._probabilities = [p / total for p in weights]

#         super().__init__(cfg, render_mode=render_mode, buf=buf, **kwargs)

#         # start with a random config from the set
#         self.active_cfg = self._resolve_original_cfg()

#     def _resolve_original_cfg(self):
#         """
#         Select a random configuration based on probabilities.

#         Returns:
#             A resolved environment configuration

#         Raises:
#             ValueError: If the number of agents in the selected environment
#                        doesn't match the global number of agents
#         """
#         selected_path = np.random.choice(self._original_cfg_paths, p=self._probabilities)
#         cfg = config_from_path(selected_path)
#         cfg = OmegaConf.create(cfg)

#         # Insert stats into the configuration
#         cfg = self._insert_progress_into_cfg(cfg)

#         OmegaConf.resolve(cfg)
#         return cfg
