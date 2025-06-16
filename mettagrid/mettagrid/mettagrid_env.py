from __future__ import annotations

import datetime
import logging
import random
import uuid
from typing import Any, Dict, Optional, cast

import gymnasium as gym
import numpy as np
import pufferlib
from hydra.utils import instantiate
from omegaconf import OmegaConf
from pufferlib import unroll_nested_dict
from pydantic import validate_call
from typing_extensions import override

from mettagrid.curriculum import Curriculum, Task
from mettagrid.level_builder import Level
from mettagrid.mettagrid_c import MettaGrid
from mettagrid.replay_writer import ReplayWriter
from mettagrid.stats_writer import StatsWriter
from mettagrid.util.diversity import calculate_diversity_bonus

# These data types must match PufferLib -- see pufferlib/vector.py
#
# Important:
#
# In PufferLib's class Multiprocessing, the data type for actions will be set to int32
# whenever the action space is Discrete or Multidiscrete. If we do not match the data type
# here in our child class, then we will experience extra data conversions in the background.
# Additionally the actions that are sent to the C environment will be int32 (because PufferEnv
# controls the type of self.actions) -- creating an opportunity for type confusion.

dtype_observations = np.dtype(np.uint8)
dtype_terminals = np.dtype(bool)
dtype_truncations = np.dtype(bool)
dtype_rewards = np.dtype(np.float32)
dtype_actions = np.dtype(np.int32)  # must be int32!
dtype_masks = np.dtype(bool)
dtype_success = np.dtype(bool)

logger = logging.getLogger("MettaGridEnv")


def required(func):
    """Marks methods that PufferEnv requires but does not implement for override."""
    return func


class MettaGridEnv(pufferlib.PufferEnv, gym.Env):
    # Type hints for attributes defined in the C++ extension to help Pylance
    observations: np.ndarray
    terminals: np.ndarray
    truncations: np.ndarray
    rewards: np.ndarray
    actions: np.ndarray

    @validate_call(config={"arbitrary_types_allowed": True})
    def __init__(
        self,
        curriculum: Curriculum,
        render_mode: Optional[str],
        level: Optional[Level] = None,
        buf=None,
        stats_writer: Optional[StatsWriter] = None,
        replay_writer: Optional[ReplayWriter] = None,
        **kwargs,
    ):
        self._render_mode = render_mode
        self._curriculum = curriculum
        self._task: Task = self._curriculum.get_task()
        self._level = level
        self._last_level_per_task = {}
        self._renderer = None
        self._map_labels = []
        self._stats_writer = stats_writer
        self._replay_writer = replay_writer
        self._episode_id: str | None = None
        self._reset_at = datetime.datetime.now()
        self._current_seed = 0

        self.labels = self._task.env_cfg().get("labels", None)
        self._should_reset = False

        self._initialize_c_env()
        super().__init__(buf)

        if self._render_mode is not None:
            if self._render_mode == "human":
                from .renderer.nethack import NethackRenderer

                self._renderer = NethackRenderer(self.object_type_names)
            elif self._render_mode == "miniscope":
                from .renderer.miniscope import MiniscopeRenderer

                self._renderer = MiniscopeRenderer(self.object_type_names)

    def _make_episode_id(self):
        return str(uuid.uuid4())

    def _initialize_c_env(self) -> None:
        """Initialize the C++ environment."""
        task = self._task
        level = self._level
        last_level = self._last_level_per_task.get(task.id(), None)
        if level is None and last_level is not None and random.random() < task.env_cfg().get("replay_level_prob", 0):
            # Replay the last level we had for this task, rather than building a new one.
            # This will be less adaptive to changes in the task config, but will save a lot
            # of CPU, and so is helpful if we're CPU bound.
            level = last_level

        if level is None:
            map_builder_config = task.env_cfg().game.map_builder
            map_builder = instantiate(map_builder_config, _recursive_=True, _convert_="all")
            level = map_builder.build()

        self._last_level_per_task[task.id()] = level

        # Validate the level
        level_agents = np.count_nonzero(np.char.startswith(level.grid, "agent"))
        assert task.env_cfg().game.num_agents == level_agents, (
            f"Number of agents {task.env_cfg().game.num_agents} does not match number of agents in map {level_agents}"
        )

        # Convert to container for C++ code with explicit casting to Dict[str, Any]
        config_dict = cast(Dict[str, Any], OmegaConf.to_container(task.env_cfg()))

        self._map_labels = level.labels

        # Convert string array to list of strings for C++ compatibility
        # TODO: push the not-numpy-array higher up the stack, and consider pushing not-a-sparse-list lower.
        self._c_env = MettaGrid(config_dict, level.grid.tolist())

        self._grid_env = self._c_env

    @override  # pufferlib.PufferEnv.reset
    def reset(self, seed: int | None = None) -> tuple[np.ndarray, dict]:
        self._task = self._curriculum.get_task()

        self._initialize_c_env()

        assert self.observations.dtype == dtype_observations
        assert self.terminals.dtype == dtype_terminals
        assert self.truncations.dtype == dtype_truncations
        assert self.rewards.dtype == dtype_rewards

        self._c_env.set_buffers(self.observations, self.terminals, self.truncations, self.rewards)

        self._episode_id = self._make_episode_id()
        self._current_seed = seed or 0
        self._reset_at = datetime.datetime.now()
        if self._replay_writer:
            self._replay_writer.start_episode(self._episode_id, self)

        obs, infos = self._c_env.reset()
        self._should_reset = False
        return obs, infos

    @override  # pufferlib.PufferEnv.step
    def step(self, actions: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict]:
        """
        Execute one timestep of the environment dynamics with the given actions.

        IMPORTANT: In training mode, the `actions` parameter and `self.actions` may be the same
        object, but in simulation mode they are independent. Always use the passed-in `actions`
        parameter to ensure correct behavior in all contexts.

        Args:
            actions: A numpy array of shape (num_agents, 2) with dtype np.int32

        Returns:
            Tuple of (observations, rewards, terminals, truncations, infos)

        """

        # Note: We explicitly allow invalid actions to be used. The environment will
        # penalize the agent for attempting invalid actions as a side effect of ActionHandler::handle_action()

        if self._replay_writer and self._episode_id:
            self._replay_writer.log_pre_step(self._episode_id, actions)

        self._c_env.step(actions)

        if self._replay_writer and self._episode_id:
            self._replay_writer.log_post_step(self._episode_id, self.rewards)

        infos = {}
        if self.terminals.all() or self.truncations.all():
            if self._task.env_cfg().game.diversity_bonus.enabled:
                self.rewards *= calculate_diversity_bonus(
                    self._c_env.get_episode_rewards(),
                    self._c_env.get_agent_groups(),
                    self._task.env_cfg().game.diversity_bonus.similarity_coef,
                    self._task.env_cfg().game.diversity_bonus.diversity_coef,
                )

            self.process_episode_stats(infos)
            self._should_reset = True
            self._task.complete(self._c_env.get_episode_rewards().mean())

        return self.observations, self.rewards, self.terminals, self.truncations, infos

    @override
    def close(self):
        pass

    def process_episode_stats(self, infos: Dict[str, Any]):
        episode_rewards = self._c_env.get_episode_rewards()
        episode_rewards_sum = episode_rewards.sum()
        episode_rewards_mean = episode_rewards_sum / self._c_env.num_agents

        infos.update(
            {
                "episode/reward.sum": episode_rewards_sum,
                "episode/reward.mean": episode_rewards_mean,
                "episode/reward.min": episode_rewards.min(),
                "episode/reward.max": episode_rewards.max(),
                "episode_length": self._c_env.current_step,
                f"task/{self._task.name()}/reward": episode_rewards_mean,
            }
        )

        for label in self._map_labels:
            infos[f"rewards/map:{label}"] = episode_rewards_mean

        if self.labels is not None:
            for label in self.labels:
                infos[f"rewards/env:{label}"] = episode_rewards_mean

        stats = self._c_env.get_episode_stats()

        infos["episode_rewards"] = episode_rewards
        # infos["agent_raw"] = stats["agent"]
        infos["game"] = stats["game"]
        infos["agent"] = {}
        for agent_stats in stats["agent"]:
            for n, v in agent_stats.items():
                infos["agent"][n] = infos["agent"].get(n, 0) + v
        for n, v in infos["agent"].items():
            infos["agent"][n] = v / self._c_env.num_agents

        replay_url = None
        if self._replay_writer:
            assert self._episode_id is not None, "Episode ID must be set before writing a replay"
            replay_url = self._replay_writer.write_replay(self._episode_id)
            infos["replay_url"] = replay_url

        if self._stats_writer:
            assert self._episode_id is not None, "Episode ID must be set before writing stats"

            attributes = {
                "seed": self._current_seed,
                "map_w": self.map_width,
                "map_h": self.map_height,
            }

            for k, v in unroll_nested_dict(OmegaConf.to_container(self._task.env_cfg(), resolve=False)):
                attributes[f"config.{k.replace('/', '.')}"] = str(v)

            agent_metrics = {}
            for agent_idx, agent_stats in enumerate(stats["agent"]):
                agent_metrics[agent_idx] = {}
                agent_metrics[agent_idx]["reward"] = float(episode_rewards[agent_idx])
                for k, v in agent_stats.items():
                    agent_metrics[agent_idx][k] = float(v)

            grid_objects: Dict[int, Any] = self._c_env.grid_objects()
            # iterate over grid_object values
            agent_groups: Dict[int, int] = {
                v["agent_id"]: v["agent:group"] for v in grid_objects.values() if v["type"] == 0
            }

            self._stats_writer.record_episode(
                self._episode_id,
                attributes,
                agent_metrics,
                agent_groups,
                self.max_steps,
                replay_url,
                self._reset_at,
            )
        self._episode_id = None

    @property
    def max_steps(self) -> int:
        return self._c_env.max_steps

    @property
    @required
    def single_observation_space(self) -> gym.spaces.Box:
        """
        Return the observation space for a single agent.
        Returns:
            Box: A Box space with shape depending on whether observation tokens are used.
                If using tokens: (num_agents, num_observation_tokens, 3)
                Otherwise: (obs_height, obs_width, num_grid_features)
        """
        return self._c_env.observation_space

    @property
    @required
    def single_action_space(self) -> gym.spaces.MultiDiscrete:
        """
        Return the action space for a single agent.
        Returns:
            MultiDiscrete: A MultiDiscrete space with shape (num_actions, max_action_arg + 1)
        """
        return self._c_env.action_space

    # obs_width and obs_height correspond to the view window size, and should indicate the grid from which
    # tokens are being computed.
    @property
    def obs_width(self):
        return self._c_env.obs_width

    @property
    def obs_height(self):
        return self._c_env.obs_height

    @property
    def action_names(self) -> list[str]:
        return self._c_env.action_names()

    @property
    @required
    def num_agents(self) -> int:
        return self._c_env.num_agents

    def render(self) -> str | None:
        if self._renderer is None:
            return None

        return self._renderer.render(self._c_env.current_step, self._c_env.grid_objects())

    @property
    @override
    def done(self):
        return self._should_reset

    @property
    def feature_normalizations(self) -> dict[int, float]:
        return self._c_env.feature_normalizations()

    @property
    def global_features(self):
        return []

    @property
    @override
    def render_mode(self):
        return self._render_mode

    @property
    def map_width(self) -> int:
        return self._c_env.map_width

    @property
    def map_height(self) -> int:
        return self._c_env.map_height

    @property
    def grid_objects(self) -> dict[int, dict[str, Any]]:
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
    def max_action_args(self) -> list[int]:
        """
        Get the maximum argument variant for each action type.
        Returns:
            List of integers representing max parameters for each action type
        """
        action_args_array = self._c_env.max_action_args()
        return [int(x) for x in action_args_array]

    @property
    def action_success(self) -> list[bool]:
        action_success_array = self._c_env.action_success()
        return [bool(x) for x in action_success_array]

    @property
    def object_type_names(self) -> list[str]:
        return self._c_env.object_type_names()

    @property
    def inventory_item_names(self) -> list[str]:
        return self._c_env.inventory_item_names()

    @property
    def config(self) -> dict[str, Any]:
        return cast(dict[str, Any], OmegaConf.to_container(self._task.env_cfg(), resolve=False))
