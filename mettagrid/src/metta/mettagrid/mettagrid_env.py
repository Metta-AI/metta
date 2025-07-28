from __future__ import annotations

import datetime
import logging
import os
import time
import uuid
from typing import Any, Dict, Optional, cast

import numpy as np
from gymnasium import Env as GymEnv
from gymnasium import spaces
from omegaconf import OmegaConf
from pufferlib import PufferEnv
from pydantic import validate_call
from typing_extensions import override

from metta.common.profiling.stopwatch import Stopwatch, with_instance_timer
from metta.mettagrid.curriculum.core import Task
from metta.mettagrid.level_builder import Level
from metta.mettagrid.mettagrid_c import MettaGrid
from metta.mettagrid.mettagrid_c_config import from_mettagrid_config
from metta.mettagrid.replay_writer import ReplayWriter
from metta.mettagrid.stats_writer import StatsWriter
from metta.mettagrid.util.dict_utils import unroll_nested_dict
from metta.rl.curriculum.curriculum_client import CurriculumClient

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


class MettaGridEnv(PufferEnv, GymEnv):
    # Type hints for attributes defined in the C++ extension to help Pylance
    observations: np.ndarray
    terminals: np.ndarray
    truncations: np.ndarray
    rewards: np.ndarray
    actions: np.ndarray

    @validate_call(config={"arbitrary_types_allowed": True})
    def __init__(
        self,
        curriculum_client: CurriculumClient,
        render_mode: Optional[str],
        level: Optional[Level] = None,
        buf=None,
        stats_writer: Optional[StatsWriter] = None,
        replay_writer: Optional[ReplayWriter] = None,
        is_training: bool = False,
        **kwargs,
    ):
        self._curriculum_client = curriculum_client
        self.timer = Stopwatch(logger)
        self.timer.start()
        self.timer.start("thread_idle")
        self._steps = 0
        self._resets = 0

        self._render_mode = render_mode
        self._level = level
        self._renderer = None
        self._map_labels: list[str] = []
        self._stats_writer = stats_writer
        self._replay_writer = replay_writer
        self._episode_id: str | None = None
        self._reset_at = datetime.datetime.now()
        self._current_seed: int = 0  # must be unsigned

        self._labels: list[str] = []
        self._should_reset = False

        self._is_training = is_training

        self._task = self.get_task()
        self._initialize_c_env()
        super().__init__(buf)

        if self._render_mode is not None:
            if self._render_mode == "human":
                from metta.mettagrid.renderer.nethack import NethackRenderer

                self._renderer = NethackRenderer(self.object_type_names)
            elif self._render_mode == "miniscope":
                from metta.mettagrid.renderer.miniscope import MiniscopeRenderer

                self._renderer = MiniscopeRenderer(self.object_type_names)
            elif self._render_mode == "raylib":
                # Only initialize raylib renderer if not in CI/Docker environment
                is_ci_environment = bool(
                    os.environ.get("CI") or os.environ.get("GITHUB_ACTIONS") or os.path.exists("/.dockerenv")
                )
                if not is_ci_environment:
                    try:
                        from metta.mettagrid.renderer.raylib import RaylibRenderer

                        self._renderer = RaylibRenderer(self.object_type_names, self.map_width, self.map_height)
                    except ImportError:
                        logger.warning("Raylib renderer requested but raylib not available")
                        self._renderer = None
                else:
                    logger.info("Raylib renderer disabled in CI/Docker environment")
                    self._renderer = None

    def _make_episode_id(self):
        return str(uuid.uuid4())

    @with_instance_timer("_initialize_c_env")
    def _initialize_c_env(self) -> None:
        """Initialize the C++ environment."""
        task_cfg = self._task.env_cfg()
        level = self._level

        if level is None:
            with self.timer("_initialize_c_env.build_map"):
                level = task_cfg.game.map_builder.build()

        # Validate the level
        level_agents = np.count_nonzero(np.char.startswith(level.grid, "agent"))
        assert task_cfg.game.num_agents == level_agents, (
            f"Number of agents {task_cfg.game.num_agents} does not match number of agents in map {level_agents}"
        )

        game_config_dict = OmegaConf.to_container(task_cfg.game)
        assert isinstance(game_config_dict, dict), "No valid game config dictionary in the environment config"

        # During training, we run a lot of envs in parallel, and it's better if they are not
        # all synced together. The desync_episodes flag is used to desync the episodes.
        # Ideally vecenv would have a way to desync the episodes, but it doesn't.
        if self._is_training and self._resets == 0:
            max_steps = game_config_dict["max_steps"]
            game_config_dict["max_steps"] = int(np.random.randint(1, max_steps + 1))

        self._map_labels = level.labels

        # Convert string array to list of strings for C++ compatibility
        # TODO: push the not-numpy-array higher up the stack, and consider pushing not-a-sparse-list lower.
        with self.timer("_initialize_c_env.make_c_env"):
            c_cfg = None
            try:
                c_cfg = from_mettagrid_config(game_config_dict)
            except Exception as e:
                logger.error(f"Error initializing C++ environment: {e}")
                logger.error(f"Game config: {game_config_dict}")
                raise e

            self._c_env = MettaGrid(c_cfg, level.grid.tolist(), self._current_seed)

        self._grid_env = self._c_env

    @override  # pufferlib.PufferEnv.reset
    @with_instance_timer("reset")
    def reset(self, seed: int | None = None) -> tuple[np.ndarray, dict]:
        self.timer.stop("thread_idle")

        self._task = self.get_task()

        self._initialize_c_env()
        self._steps = 0
        self._resets += 1

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

        self.timer.start("thread_idle")
        return obs, infos

    def get_task(self) -> Task:
        return self._curriculum_client.get_task()

    @override  # pufferlib.PufferEnv.step
    @with_instance_timer("step")
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
        self.timer.stop("thread_idle")

        # Note: We explicitly allow invalid actions to be used. The environment will
        # penalize the agent for attempting invalid actions as a side effect of ActionHandler::handle_action()

        with self.timer("_c_env.step"):
            self._c_env.step(actions)
            self._steps += 1

        if self._replay_writer and self._episode_id:
            with self.timer("_replay_writer.log_step"):
                self._replay_writer.log_step(self._episode_id, actions, self.rewards)

        infos = {}
        if self.terminals.all() or self.truncations.all():
            self.process_episode_stats(infos)
            self._should_reset = True
            self._task.complete(self._c_env.get_episode_rewards().mean())

        self.timer.start("thread_idle")
        return self.observations, self.rewards, self.terminals, self.truncations, infos

    @override
    def close(self):
        pass

    def process_episode_stats(self, infos: Dict[str, Any]):
        self.timer.start("process_episode_stats")

        infos.clear()

        episode_rewards = self._c_env.get_episode_rewards()
        episode_rewards_sum = episode_rewards.sum()
        episode_rewards_mean = episode_rewards_sum / self._c_env.num_agents

        for label in self._map_labels + self._labels:
            infos[f"map_reward/{label}"] = episode_rewards_mean

        with self.timer("_c_env.get_episode_stats"):
            stats = self._c_env.get_episode_stats()

        infos["game"] = stats["game"]
        infos["agent"] = {}
        for agent_stats in stats["agent"]:
            for n, v in agent_stats.items():
                infos["agent"][n] = infos["agent"].get(n, 0) + v
        for n, v in infos["agent"].items():
            infos["agent"][n] = v / self._c_env.num_agents

        attributes: dict[str, int] = {
            "seed": self._current_seed,
            "map_w": self.map_width,
            "map_h": self.map_height,
            "initial_grid_hash": self.initial_grid_hash,
            "steps": self._steps,
            "resets": self._resets,
            "max_steps": self.max_steps,
            "completion_time": int(time.time()),
        }
        infos["attributes"] = attributes

        replay_url = None

        with self.timer("_replay_writer"):
            if self._replay_writer:
                assert self._episode_id is not None, "Episode ID must be set before writing a replay"
                replay_url = self._replay_writer.write_replay(self._episode_id)
                infos["replay_url"] = replay_url

        with self.timer("_stats_writer"):
            if self._stats_writer:
                assert self._episode_id is not None, "Episode ID must be set before writing stats"

                env_cfg_flattened: dict[str, str] = {}
                env_cfg = OmegaConf.to_container(self._task.env_cfg(), resolve=False)
                for k, v in unroll_nested_dict(cast(dict[str, Any], env_cfg)):
                    env_cfg_flattened[f"config.{str(k).replace('/', '.')}"] = str(v)

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
                    env_cfg_flattened,
                    agent_metrics,
                    agent_groups,
                    self.max_steps,
                    replay_url,
                    self._reset_at,
                )

        self.timer.stop("process_episode_stats")

        elapsed_times = self.timer.get_all_elapsed()
        thread_idle_time = elapsed_times.pop("thread_idle", 0)

        wall_time = self.timer.get_elapsed()
        adjusted_wall_time = wall_time - thread_idle_time

        lap_times = self.timer.lap_all(exclude_global=False)
        lap_thread_idle_time = lap_times.pop("thread_idle", 0)
        wall_time_for_lap = lap_times.pop("global", 0)
        adjusted_lap_time = wall_time_for_lap - lap_thread_idle_time

        infos["timing_per_epoch"] = {
            **{
                f"active_frac/{op}": lap_elapsed / adjusted_lap_time if adjusted_lap_time > 0 else 0
                for op, lap_elapsed in lap_times.items()
            },
            **{f"msec/{op}": lap_elapsed * 1000 for op, lap_elapsed in lap_times.items()},
            "frac/thread_idle": lap_thread_idle_time / wall_time_for_lap,
        }
        infos["timing_cumulative"] = {
            **{
                f"active_frac/{op}": elapsed / adjusted_wall_time if adjusted_wall_time > 0 else 0
                for op, elapsed in elapsed_times.items()
            },
            "frac/thread_idle": thread_idle_time / wall_time,
        }

        task_init_time_msec = lap_times.get("_initialize_c_env", 0) * 1000
        infos.update(
            {
                f"task_reward/{self._task.short_name()}/rewards.mean": episode_rewards_mean,
                f"task_timing/{self._task.short_name()}/init_time_msec": task_init_time_msec,
            }
        )

        self._episode_id = None

    @property
    def max_steps(self) -> int:
        return self._c_env.max_steps

    @property
    @required
    def single_observation_space(self) -> spaces.Box:
        """
        Return the observation space for a single agent.
        Returns:
            Box: A Box space with shape (num_agents, num_observation_tokens, 3)
        """
        return self._c_env.observation_space

    @property
    @required
    def single_action_space(self) -> spaces.MultiDiscrete:
        """
        Return the action space for a single agent.
        Returns:
            MultiDiscrete: A MultiDiscrete space with shape (num_actions, max_action_arg + 1)
        """
        return self._c_env.action_space

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
        # Use the configured renderer if available
        if self._renderer is not None and hasattr(self._renderer, "render"):
            return self._renderer.render(self._steps, self.grid_objects)

        return None

    @property
    @override
    def done(self):
        return self._should_reset

    @property
    def feature_normalizations(self) -> dict[int, float]:
        return self._c_env.feature_normalizations()

    def get_observation_features(self) -> dict[str, dict[str, int | float]]:
        """
        Build the features dictionary for initialize_to_environment.

        Returns:
            Dictionary mapping feature names to their properties
        """
        # Get feature spec from C++ environment
        feature_spec = self._c_env.feature_spec()

        features = {}
        for feature_name, feature_info in feature_spec.items():
            feature_dict: dict[str, int | float] = {"id": feature_info["id"]}

            # Add normalization if present
            if "normalization" in feature_info:
                feature_dict["normalization"] = feature_info["normalization"]

            features[feature_name] = feature_dict

        return features

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
    def initial_grid_hash(self) -> int:
        """Returns the hash of the initial grid configuration."""
        return self._c_env.initial_grid_hash
