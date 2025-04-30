"""
MettaGrid Environment implementation for Pufferlib.
This module provides environment classes for Metta Grid simulations.
"""

import copy
import time
from typing import Any, Dict, Optional, Union

import gymnasium as gym
import numpy as np
import numpy.typing as npt
import pufferlib
from omegaconf import DictConfig, ListConfig, OmegaConf

# Import with explicit comment about the pylint disable
from mettagrid.config.utils import simple_instantiate
from mettagrid.mettagrid_c import MettaGrid  # pylint: disable=E0611


def get_or_0(accessor_fn):
    try:
        return accessor_fn()
    except (AttributeError, KeyError, TypeError, IndexError):
        return 0


class MettaGridEnv(pufferlib.PufferEnv, gym.Env):
    """
    Environment class for MettaGrid simulations.

    This class wraps the C++ MettaGrid implementation and provides a gym-compatible
    interface for reinforcement learning.
    """

    # Type hints for attributes defined in the C++ extension to help Pylance
    observations: np.ndarray
    terminals: np.ndarray
    truncations: np.ndarray
    rewards: np.ndarray
    actions: np.ndarray

    def __init__(
        self,
        cfg: Union[DictConfig, ListConfig],
        render_mode: Optional[str],
        env_map: Optional[np.ndarray] = None,
        buf=None,
        **kwargs,
    ):
        """
        Initialize a MettaGridEnv.

        Args:
            cfg: provided OmegaConf configuration
            render_mode: Mode for rendering the environment
            buf: Buffer for Pufferlib
            **kwargs: Additional arguments passed to parent classes
        """

        self.last_episode_info: Dict[str, Any] = {}
        self.start_time = None

        self._original_cfg = cfg
        self._render_mode = render_mode
        self._renderer = None
        self._env_map = env_map
        self._map_builder = None

        self.active_cfg = self._resolve_original_cfg()
        self.labels = OmegaConf.select(self.active_cfg, "labels", default=None)

        self.initialize_episode()
        super().__init__(buf)

    def _insert_progress_into_cfg(self, cfg: Union[DictConfig, ListConfig]) -> Union[DictConfig, ListConfig]:
        """
        Insert values from last_episode_info into the configuration used to resolve the environment.

        Args:
            cfg: The configuration to update
        Returns:
            The updated configuration with stats inserted
        """

        progress_cfg = OmegaConf.create({})
        progress_cfg.progress = OmegaConf.create({})

        # store the last mean_reward before we overwrite it
        last_mean_reward = self.last_episode_info.get("progress/mean_reward", 0.0)
        progress_cfg.progress.last_mean_reward = float(last_mean_reward)  # Convert to Python float

        mean_reward = self.last_episode_info.get("episode/reward.mean", 0.0)
        progress_cfg.progress.mean_reward = float(mean_reward)  # Convert to Python float

        last_difficulty = self.last_episode_info.get("game/difficulty", 0.0)
        progress_cfg.progress.last_difficulty = float(last_difficulty)  # Convert to Python float

        # Set struct flag to False to allow accessing undefined fields
        OmegaConf.set_struct(cfg, False)
        cfg = OmegaConf.merge(cfg, progress_cfg)
        OmegaConf.set_struct(cfg, True)

        return cfg

    def _resolve_original_cfg(self) -> Union[DictConfig, ListConfig]:
        """
        Create a new resolved environment configuration from the template.

        This method:
        1. Creates a deep copy of the original configuration
        2. Resolves all variable interpolations, references, and custom resolvers
        - Any randomness in resolvers (e.g. ${sampling:0, 100, 50}) will be
            evaluated at this point, potentially resulting in a different environment
            each time this method is called

        Returns:
            The resolved configuration object with all references replaced by concrete values
        """

        cfg = OmegaConf.create(copy.deepcopy(self._original_cfg))
        cfg = self._insert_progress_into_cfg(cfg)
        OmegaConf.resolve(cfg)
        return cfg

    def initialize_episode(self):
        """Initialize a new episode."""

        # Instantiate map builder
        if self._env_map is None:
            self._map_builder = simple_instantiate(
                self.active_cfg.game.map_builder,
                recursive=self.active_cfg.game.get("recursive_map_builder", True),
            )
            env_map = self._map_builder.build()
        else:
            env_map = self._env_map

        # verify agent count
        map_agents = np.count_nonzero(np.char.startswith(env_map, "agent"))
        game_agents = get_or_0(lambda: self.active_cfg.game.num_agents)
        if game_agents != map_agents:
            raise ValueError(
                f"Number of agents in game ({game_agents}) does not match number of agents in map {map_agents}"
            )

        # Create C++ environment
        self._c_env = MettaGrid(self.active_cfg, env_map)
        self._grid_env = self._c_env
        self._num_agents = self._c_env.num_agents()
        self._env = self._grid_env

        # reset episode tracking
        self.start_time = time.perf_counter()
        self.episode_finished = False

        # optimizations for use in step()
        self._normalize_rewards = get_or_0(lambda: self.active_cfg.normalize_rewards)
        self._actions = np.zeros((self.num_agents, 2), dtype=np.int32)
        self._min_report_steps_interval: int = get_or_0(lambda: self.active_cfg.min_steps_per_info)

    def finalize_episode(self):
        """
        Process statistics and update progress tracking at the end of each episode.
        """

        rewards = self._c_env.get_episode_rewards()
        stats = self._c_env.get_episode_stats()
        rewards_sum = rewards.sum()
        rewards_mean = rewards_sum / self._num_agents

        # calculate the average performance for all agent stats (counters)
        agent_stats = {}
        for agent_entry in stats["agent"]:
            for name, count in agent_entry.items():
                agent_stats[name] = agent_stats.get(name, 0) + count
        for name, cumulative_count in agent_stats.items():
            agent_stats[name] = cumulative_count / self._num_agents

        # Get current timestamp and calculate duration
        current_time = time.perf_counter()
        episode_duration = current_time - self.start_time if self.start_time is not None else 0.0

        # N.B. most of these are just for plots, but we are also using this as our memory space
        # for progress tracking. Please do not remove fields that are marked for this purpose!
        update_dict = {
            "episode/reward.sum": rewards_sum,
            "episode/reward.mean": rewards_mean,  # [progress tracking]
            "episode/reward.min": rewards.min(),
            "episode/reward.max": rewards.max(),
            "episode/totals_steps": self._c_env.current_timestep(),
            "episode/duration_sec": episode_duration,
            "episode/timestamp": current_time,
            "game": stats["game"],
            "game/difficulty": get_or_0(lambda: self.active_cfg.game.difficulty),  # [progress tracking]
            "game/max_steps": get_or_0(lambda: self.active_cfg.game.max_steps),
            "game/min_size": get_or_0(lambda: self.active_cfg.game.min_size),
            "game/max_size": get_or_0(lambda: self.active_cfg.game.max_size),
            "game/width": get_or_0(lambda: self.active_cfg.game.map_builder["width"]),
            "game/height": get_or_0(lambda: self.active_cfg.game.map_builder["height"]),
            "progress/episode_count": get_or_0(lambda: self.active_cfg.progress.episode_count),
            "progress/mean_reward": get_or_0(lambda: self.active_cfg.progress.mean_reward),  # [progress tracking]
            "progress/last_mean_reward": get_or_0(lambda: self.active_cfg.progress.last_mean_reward),
            "progress/last_difficulty": get_or_0(lambda: self.active_cfg.progress.last_difficulty),
            "agent": agent_stats,
        }

        # Add map labels if they exist
        if self._map_builder is not None and self._map_builder.labels is not None:
            for label in self._map_builder.labels:
                update_dict[f"rewards/map:{label}"] = rewards_mean

        # Add environment labels if they exist
        if self.labels is not None:
            for label in self.labels:
                update_dict[f"rewards/env:{label}"] = rewards_mean

        # Do a single update with the complete dictionary
        self.last_episode_info.update(update_dict)

        # we back the property self.done with this value because
        # it is accessed thousands of times per step!
        self.episode_finished = True

    def reset(self, seed=None, options=None):
        """
        Reset the environment for a new episode.

        Args:
            seed: Random seed
            options: Additional options for resetting

        Returns:
            Tuple of (observations, info)
        """

        # Configure random seed if provided
        if seed is not None:
            np.random.seed(seed)

        # resolve a new map
        self.active_cfg = self._resolve_original_cfg()
        self.labels = OmegaConf.select(self.active_cfg, "labels", default=None)

        self.initialize_episode()
        self._c_env.set_buffers(self.observations, self.terminals, self.truncations, self.rewards)

        # Reset the environment and return initial (observations, infos)
        obs, infos = self._c_env.reset()

        return obs, infos

    def step(self, actions: npt.NDArray[np.uint8]):  # Shape: (num_agents, 2)
        """
        Take a step in the environment.

        BE VERY CAREFUL NOT TO SLOW THIS FUNCTION DOWN!

        Args:
            actions: Array of actions for each agent

            Actions have an action id and arg. So "move left" could be 1,1 while "move right" could be 1,2

        Returns:
            Tuple of (observations, rewards, terminals, truncations, infos)
        """

        np.copyto(self._actions, actions, casting="unsafe")
        self._c_env.step(self._actions)

        # Normalize rewards if needed (cache the configuration check)
        if self._normalize_rewards:
            self.rewards -= self.rewards.mean()

        # send info only at the end of each episode
        send_last_episode_info = False

        # if this step completes the episode, compute the stats
        if self.terminals.all() or self.truncations.all():
            self.finalize_episode()
            send_last_episode_info = True

        self.steps_since_episode_report: int = getattr(self, "steps_since_episode_report", 0) + 1

        # check against OmegaConf config param env.min_steps_per_info
        if send_last_episode_info and self.steps_since_episode_report < self._min_report_steps_interval:
            send_last_episode_info = False

        # Only reset if we're actually sending info
        if send_last_episode_info:
            self.steps_since_episode_report = 0

        # passing arguments here significantly slows down performance so throttle reporting
        return (
            self.observations,
            self.rewards,
            self.terminals,
            self.truncations,
            self.last_episode_info if send_last_episode_info else {},
        )

    @property
    def _max_steps(self):
        """Maximum number of steps allowed in an episode."""
        return self.active_cfg.game.max_steps

    @property
    def single_observation_space(self):
        """Observation space for a single agent."""
        return self._env.observation_space

    @property
    def single_action_space(self):
        """Action space for a single agent."""
        return self._env.action_space

    def action_names(self):
        """Get names of available actions."""
        return self._env.action_names()

    @property
    def player_count(self):
        """Number of players/agents in the environment."""
        return self._num_agents

    @property
    def num_agents(self):
        """Number of agents in the environment."""
        return self._num_agents

    def render(self):
        if self._renderer is None:
            return None

        return self._renderer.render(self._c_env.current_timestep(), self._c_env.grid_objects())

    @property
    def done(self):
        # cache this property value because it is accessed many times per step
        return self.episode_finished

    @property
    def grid_features(self):
        """Features of the grid."""
        return self._env.grid_features()

    @property
    def global_features(self):
        """Global features of the environment."""
        return []

    @property
    def render_mode(self):
        """Mode for rendering the environment."""
        return self._render_mode

    @property
    def map_width(self):
        """Width of the environment map."""
        return self._c_env.map_width()

    @property
    def map_height(self):
        """Height of the environment map."""
        return self._c_env.map_height()

    @property
    def grid_objects(self):
        """Objects in the grid."""
        return self._c_env.grid_objects()

    @property
    def max_action_args(self):
        """Maximum number of arguments for actions."""
        return self._c_env.max_action_args()

    @property
    def action_success(self):
        """Success status of the most recent actions."""
        return np.asarray(self._c_env.action_success())

    def object_type_names(self):
        """Names of object types in the environment."""
        return self._c_env.object_type_names()

    def inventory_item_names(self):
        return self._c_env.inventory_item_names()

    def close(self):
        """Close the environment."""
        pass
