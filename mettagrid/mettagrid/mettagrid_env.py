# mettagrid/mettagrid_env.py
from __future__ import annotations

import copy
import datetime
import uuid
from types import SimpleNamespace
from typing import Any, Dict, Optional

import gymnasium as gym
import numpy as np
import pufferlib
from omegaconf import DictConfig, OmegaConf
from pufferlib.utils import unroll_nested_dict

from mettagrid.config.utils import simple_instantiate
from mettagrid.core import MettaGrid
from mettagrid.replay_writer import ReplayWriter
from mettagrid.resolvers import register_resolvers
from mettagrid.stats_writer import StatsWriter
from mettagrid.util.debug import save_mettagrid_args

# Rebuild the NumPy types using the exposed function
np_observations_type = np.dtype(MettaGrid.get_numpy_type_name("observations"))
np_terminals_type = np.dtype(MettaGrid.get_numpy_type_name("terminals"))
np_truncations_type = np.dtype(MettaGrid.get_numpy_type_name("truncations"))
np_rewards_type = np.dtype(MettaGrid.get_numpy_type_name("rewards"))
np_actions_type = np.dtype(MettaGrid.get_numpy_type_name("actions"))
np_masks_type = np.dtype(MettaGrid.get_numpy_type_name("masks"))
np_success_type = np.dtype(MettaGrid.get_numpy_type_name("success"))


class MettaGridEnv(pufferlib.PufferEnv, gym.Env):
    # Type hints for attributes defined in the C++ extension to help Pylance
    observations: np.ndarray
    terminals: np.ndarray
    truncations: np.ndarray
    rewards: np.ndarray
    actions: np.ndarray

    def __init__(
        self,
        env_cfg: DictConfig,
        render_mode: Optional[str],
        env_map: Optional[np.ndarray] = None,
        buf=None,
        stats_writer: Optional[StatsWriter] = None,
        replay_writer: Optional[ReplayWriter] = None,
        **kwargs,
    ):
        self._debug = False
        self._render_mode = render_mode
        self._cfg_template = env_cfg
        self._env_cfg = self._get_new_env_cfg()
        self._env_map = env_map
        self._renderer = None
        self._map_builder = None

        self._stats_writer = stats_writer
        self._replay_writer = replay_writer
        self._episode_id: str = ""
        self._reset_at = datetime.datetime.now()
        self._current_seed = 0

        self._reset_env()
        num_agents = self._num_agents
        obs_width = self._c_env.obs_width
        obs_height = self._c_env.obs_height
        grid_features_size = len(self._c_env.grid_features())
        self._single_observation_space = self._c_env.observation_space
        self._single_action_space = self._c_env.action_space
        # force buffers to the correct size
        buf = {
            "observations": np.zeros(
                (num_agents, obs_width, obs_height, grid_features_size), dtype=np_observations_type, order="C"
            ),
            "terminals": np.zeros((num_agents,), dtype=np_terminals_type, order="C"),
            "truncations": np.zeros((num_agents,), dtype=np_truncations_type, order="C"),
            "rewards": np.zeros((num_agents,), dtype=np_rewards_type, order="C"),
            "actions": np.zeros((num_agents, 2), dtype=np_actions_type, order="C"),
            "masks": np.ones((num_agents,), dtype=np_masks_type, order="C"),
        }
        buf_obj = SimpleNamespace(**buf)
        super().__init__(buf_obj)

        self.labels = self._env_cfg.get("labels", None)
        self.should_reset = False

        # check on the buffer shapes

        # Define expected shapes
        num_agents = self._num_agents
        expected_obs_shape = (num_agents, obs_width, obs_height, grid_features_size)

        # Validate observation shape
        obs_shape_tuple = self.observations.shape
        if self.observations.ndim != 4 or obs_shape_tuple != expected_obs_shape:
            raise ValueError(f"Observations buffer has shape {obs_shape_tuple}, expected {expected_obs_shape}")

        # Validate terminal buffer shape
        term_shape = self.terminals.shape
        if self.terminals.ndim < 1 or term_shape[0] < num_agents:
            raise ValueError(f"Terminals buffer has shape {term_shape}, expected first dimension ≥ {num_agents}")

        # Validate truncation buffer shape
        trunc_shape = self.truncations.shape
        if self.truncations.ndim < 1 or trunc_shape[0] < num_agents:
            raise ValueError(f"Truncations buffer has shape {trunc_shape}, expected first dimension ≥ {num_agents}")

        # Validate rewards buffer shape
        reward_shape = self.rewards.shape
        if self.rewards.ndim < 1 or reward_shape[0] < num_agents:
            raise ValueError(f"Rewards buffer has shape {reward_shape}, expected first dimension ≥ {num_agents}")

    def _make_episode_id(self):
        return str(uuid.uuid4())

    def _get_new_env_cfg(self):
        env_cfg = OmegaConf.create(copy.deepcopy(self._cfg_template))
        OmegaConf.resolve(env_cfg)
        return env_cfg

    def _reset_env(self):
        if self._env_map is None:
            self._map_builder = simple_instantiate(
                self._env_cfg.game.map_builder,
                recursive=self._env_cfg.game.get("recursive_map_builder", True),
            )
            env_map = self._map_builder.build()
        else:
            env_map = self._env_map

        map_agents = np.count_nonzero(np.char.startswith(env_map, "agent"))
        assert self._env_cfg.game.num_agents == map_agents, (
            f"Number of agents {self._env_cfg.game.num_agents} does not match number of agents in map {map_agents}"
        )

        if self._debug:
            save_mettagrid_args(self._env_cfg, env_map)

        self._c_env = MettaGrid(self._env_cfg, env_map)
        self._num_agents = self._c_env.num_agents()

    def reset(self, seed=None, options=None):
        """
        This method overrides the reset method from PufferEnv.

        Reset the environment to an initial state and returns an initial observation.
        """
        self._env_cfg = self._get_new_env_cfg()

        self._reset_env()
        self._c_env.set_buffers(self.observations, self.terminals, self.truncations, self.rewards)

        self._episode_id = self._make_episode_id()
        self._current_seed = seed or 0
        self._reset_at = datetime.datetime.now()
        if self._replay_writer:
            self._replay_writer.start_episode(self._episode_id, self)

        obs, infos = self._c_env.reset()
        self.should_reset = False
        return obs, infos

    def step(self, actions: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict]:
        """
        Take a step in the environment with the given actions.

        Args:
            actions: A numpy array of shape (num_agents, 2) with dtype np.int32

        Returns:
            Tuple of (observations, rewards, terminals, truncations, infos)
        """
        np.copyto(self.actions, actions.astype(np_actions_type))

        if self._replay_writer:
            self._replay_writer.log_pre_step(self._episode_id, self.actions)

        self._c_env.step(self.actions)

        if self._env_cfg.normalize_rewards:
            self.rewards -= self.rewards.mean()

        if self._replay_writer:
            self._replay_writer.log_post_step(self._episode_id, self.rewards)

        infos = {}
        if self.terminals.all() or self.truncations.all():
            self.process_episode_stats(infos)
            self.should_reset = True

        return self.observations, self.rewards, self.terminals, self.truncations, infos

    def process_episode_stats(self, infos: Dict[str, Any]):
        episode_rewards = self._c_env.get_episode_rewards()
        episode_rewards_sum = episode_rewards.sum()
        episode_rewards_mean = episode_rewards_sum / self._num_agents

        infos.update(
            {
                "episode/reward.sum": episode_rewards_sum,
                "episode/reward.mean": episode_rewards_mean,
                "episode/reward.min": episode_rewards.min(),
                "episode/reward.max": episode_rewards.max(),
                "episode_length": self._c_env.current_timestep(),
            }
        )

        if self._map_builder is not None and self._map_builder.labels is not None:
            for label in self._map_builder.labels:
                infos.update(
                    {
                        f"rewards/map:{label}": episode_rewards_mean,
                    }
                )

        if self.labels is not None:
            for label in self.labels:
                infos.update(
                    {
                        f"rewards/env:{label}": episode_rewards_mean,
                    }
                )

        stats = self._c_env.get_episode_stats()

        infos["episode_rewards"] = episode_rewards
        infos["agent_raw"] = stats["agent"]
        infos["game"] = stats["game"]
        infos["agent"] = {}

        for agent_stats in stats["agent"]:
            for n, v in agent_stats.items():
                infos["agent"][n] = infos["agent"].get(n, 0) + v
        for n, v in infos["agent"].items():
            infos["agent"][n] = v / self._num_agents

        replay_url = None
        if self._replay_writer:
            assert self._episode_id != ""
            replay_url = self._replay_writer.write_replay(self._episode_id)
        infos["replay_url"] = replay_url

        if self._stats_writer:
            assert self._episode_id != ""

            attributes = {
                "seed": self._current_seed,
                "map_w": self.map_width,
                "map_h": self.map_height,
            }

            for k, v in unroll_nested_dict(OmegaConf.to_container(self._env_cfg, resolve=False)):
                attributes[f"config.{str(k).replace('/', '.')}"] = str(v)

            agent_metrics = {}
            for agent_idx, agent_stats in enumerate(stats["agent"]):
                agent_metrics[agent_idx] = {}
                agent_metrics[agent_idx]["reward"] = float(episode_rewards[agent_idx])
                for k, v in agent_stats.items():
                    agent_metrics[agent_idx][k] = float(v)

            # TODO: Add groups
            groups = []
            group_metrics = {}
            self._stats_writer.record_episode(
                self._episode_id,
                attributes,
                groups,
                agent_metrics,
                group_metrics,
                self._max_steps,
                replay_url,
                self._reset_at,
            )
        self._episode_id = ""

    def close(self):
        pass

    @property
    def _max_steps(self):
        return self._env_cfg.game.max_steps

    @property
    def single_observation_space(self):
        return self._single_observation_space

    @property
    def single_action_space(self):
        return self._single_action_space

    @property
    def player_count(self):
        return self._num_agents

    @property
    def num_agents(self):
        return self._num_agents

    def render(self):
        if self._renderer is None:
            return None

        return self._renderer.render(self._c_env.current_timestep(), self._c_env.grid_objects())

    @property
    def done(self):
        return self.should_reset

    @property
    def grid_features(self):
        return self._c_env.grid_features()

    @property
    def global_features(self):
        return []

    @property
    def render_mode(self):
        return self._render_mode

    @property
    def map_width(self):
        return self._c_env.map_width()

    @property
    def map_height(self):
        return self._c_env.map_height()

    @property
    def grid_objects(self):
        """
        Get information about all grid objects that are present in our map.

        It is important to keep in mind the difference between grid_objects, which are things
        like "walls" or "agents", and grid_features which is the encoded representation of all possible
        observations of grid_objects that is provided to the policy.

        Returns:
            A dictionary mapping object IDs to their properties.
        """
        return self._c_env.grid_objects()

    @property
    def max_action_args(self):
        return self._c_env.max_action_args()

    @property
    def action_success(self):
        # Get the char array and convert to numpy array
        # Note: We keep it as char/int8 type for consistency
        return np.asarray(self._c_env.action_success(), dtype=np_success_type)

    def action_names(self):
        return self._c_env.action_names()

    def object_type_names(self):
        return self._c_env.object_type_names()

    def inventory_item_names(self):
        return self._c_env.inventory_item_names()


# Ensure resolvers are registered when this module is imported
register_resolvers()
