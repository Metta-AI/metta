"""
MettaGridEnv - Base Python environment class.

This class provides the common functionality for all framework-specific adapters:
- Creates new MettaGridCore instances on reset
- Manages curriculum, stats, and replay writing
- Provides common interface for all adapters
"""

from __future__ import annotations

import datetime
import logging
import time
import uuid
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple, cast

import numpy as np
from gymnasium import spaces
from omegaconf import OmegaConf
from pydantic import validate_call

from metta.common.profiling.stopwatch import Stopwatch, with_instance_timer
from metta.common.util.instantiate import instantiate
from metta.mettagrid.core import MettaGridCore
from metta.mettagrid.curriculum.core import Curriculum
from metta.mettagrid.level_builder import Level
from metta.mettagrid.mettagrid_c_config import from_mettagrid_config
from metta.mettagrid.replay_writer import ReplayWriter
from metta.mettagrid.stats_writer import StatsWriter
from metta.mettagrid.util.dict_utils import unroll_nested_dict

logger = logging.getLogger("MettaGridEnv")


class MettaGridEnv(ABC):
    """
    Base environment class for MettaGrid.

    This class provides common functionality for all framework-specific adapters:
    - Creates new MettaGridCore instances on reset
    - Manages curriculum, stats, and replay writing
    - Provides common interface for all adapters
    """

    @validate_call(config={"arbitrary_types_allowed": True})
    def __init__(
        self,
        curriculum: Curriculum,
        render_mode: Optional[str] = None,
        level: Optional[Level] = None,
        stats_writer: Optional[StatsWriter] = None,
        replay_writer: Optional[ReplayWriter] = None,
        is_training: bool = False,
        **kwargs: Any,
    ):
        """
        Initialize base MettaGridEnv.

        Args:
            curriculum: Curriculum for task management
            render_mode: Rendering mode (None, "human", "miniscope")
            level: Optional pre-built level
            stats_writer: Optional stats writer
            replay_writer: Optional replay writer
            is_training: Whether this is for training
            **kwargs: Additional arguments passed to subclasses
        """
        self.timer = Stopwatch(logger)
        self.timer.start()
        self.timer.start("thread_idle")
        self._steps = 0
        self._resets = 0

        self._render_mode = render_mode
        self._curriculum = curriculum
        self._task = self._curriculum.get_task()
        self._level = level
        self._renderer = None
        self._map_labels: List[str] = []
        self._stats_writer = stats_writer
        self._replay_writer = replay_writer
        self._episode_id: str | None = None
        self._reset_at = datetime.datetime.now()
        self._current_seed: int = 0
        self._is_training = is_training

        # Core environment instance - created on reset
        self._core_env: Optional[MettaGridCore] = None

        # Environment metadata
        self.labels: List[str] = self._task.env_cfg().get("labels", [])
        self._should_reset = False

        # Initialize renderer if needed
        if self._render_mode is not None:
            self._initialize_renderer()

    def _initialize_renderer(self) -> None:
        """Initialize renderer based on render mode."""
        if self._render_mode == "human":
            from metta.mettagrid.renderer.nethack import NethackRenderer

            # We'll set object_type_names after core env is created
            self._renderer_class = NethackRenderer
        elif self._render_mode == "miniscope":
            from metta.mettagrid.renderer.miniscope import MiniscopeRenderer

            self._renderer_class = MiniscopeRenderer

    def _make_episode_id(self) -> str:
        """Generate unique episode ID."""
        return str(uuid.uuid4())

    @with_instance_timer("_create_core_env")
    def _create_core_env(self, seed: Optional[int] = None) -> MettaGridCore:
        """
        Create a new MettaGridCore instance.

        Args:
            seed: Random seed for environment

        Returns:
            New MettaGridCore instance
        """
        task = self._task
        level = self._level

        if level is None:
            map_builder_config = task.env_cfg().game.map_builder
            with self.timer("_create_core_env.build_map"):
                map_builder = instantiate(map_builder_config, _recursive_=True)
                level = map_builder.build()

        # Validate the level
        level_agents = np.count_nonzero(np.char.startswith(level.grid, "agent"))
        assert task.env_cfg().game.num_agents == level_agents, (
            f"Number of agents {task.env_cfg().game.num_agents} does not match number of agents in map {level_agents}"
        )

        game_config_dict = OmegaConf.to_container(task.env_cfg().game)

        # Ensure we have a dict
        if not isinstance(game_config_dict, dict):
            raise ValueError(f"Expected dict for game config, got {type(game_config_dict)}")

        # Clean up config for C++ consumption
        if "map_builder" in game_config_dict:
            del game_config_dict["map_builder"]

        # Handle episode desyncing for training
        if self._is_training and self._resets == 0:
            max_steps = game_config_dict["max_steps"]
            if isinstance(max_steps, int):
                game_config_dict["max_steps"] = int(np.random.randint(1, max_steps + 1))

        self._map_labels = level.labels

        # Create C++ config
        with self.timer("_create_core_env.make_c_config"):
            try:
                c_cfg = from_mettagrid_config(game_config_dict)
            except Exception as e:
                logger.error(f"Error creating C++ config: {e}")
                logger.error(f"Game config: {game_config_dict}")
                raise e

        # Create core environment
        current_seed = seed if seed is not None else self._current_seed
        core_env = MettaGridCore(c_cfg, level.grid.tolist(), current_seed)

        # Initialize renderer if needed
        if self._render_mode is not None and self._renderer is None:
            self._renderer = self._renderer_class(core_env.object_type_names)

        return core_env

    def reset_base(self, seed: Optional[int] = None) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Base reset implementation - creates new core environment.

        Args:
            seed: Random seed

        Returns:
            Tuple of (observations, info)
        """
        self.timer.stop("thread_idle")

        # Get new task from curriculum
        self._task = self._curriculum.get_task()

        # Create new core environment
        self._core_env = self._create_core_env(seed)

        # Reset counters
        self._steps = 0
        self._resets += 1

        # Set up episode tracking
        self._episode_id = self._make_episode_id()
        self._current_seed = seed or 0
        self._reset_at = datetime.datetime.now()

        # Start replay recording if enabled
        if self._replay_writer and self._episode_id:
            self._replay_writer.start_episode(self._episode_id, self)

        # Reset flags
        self._should_reset = False

        # Get initial observations - subclasses handle buffer setup
        obs = self._get_initial_observations()

        self.timer.start("thread_idle")
        return obs, {}

    @abstractmethod
    def _get_initial_observations(self) -> np.ndarray:
        """
        Get initial observations after reset.

        This method must be implemented by subclasses to handle
        buffer allocation and setup specific to their framework.
        """
        pass

    def step_base(self, actions: np.ndarray) -> Dict[str, Any]:
        """
        Base step implementation - handles common logic.

        Args:
            actions: Action array

        Returns:
            Info dictionary with episode completion data
        """
        self.timer.stop("thread_idle")

        if self._core_env is None:
            raise RuntimeError("Environment not initialized. Call reset() first.")

        # Execute step in core environment
        with self.timer("_core_env.step"):
            self._core_env.step(actions)
            self._steps += 1

        # Record step for replay
        if self._replay_writer and self._episode_id:
            with self.timer("_replay_writer.log_step"):
                rewards = self._core_env._reward_buffer
                if rewards is not None:
                    self._replay_writer.log_step(self._episode_id, actions, rewards)

        # Check for episode completion
        infos = {}
        terminals = self._core_env._terminal_buffer
        truncations = self._core_env._truncation_buffer

        if terminals is not None and truncations is not None:
            if terminals.all() or truncations.all():
                self._process_episode_completion(infos)
                self._should_reset = True

        self.timer.start("thread_idle")
        return infos

    def _process_episode_completion(self, infos: Dict[str, Any]) -> None:
        """Process episode completion - stats, curriculum, etc."""
        if self._core_env is None:
            return

        self.timer.start("process_episode_stats")

        # Clear any existing infos
        infos.clear()

        # Get episode rewards and stats
        episode_rewards = self._core_env.get_episode_rewards()
        episode_rewards_sum = episode_rewards.sum()
        episode_rewards_mean = episode_rewards_sum / self._core_env.num_agents

        # Add map and label rewards
        for label in self._map_labels + self.labels:
            infos[f"map_reward/{label}"] = episode_rewards_mean

        # Add curriculum stats
        infos.update(self._curriculum.get_completion_rates())
        curriculum_stats = self._curriculum.get_curriculum_stats()
        for key, value in curriculum_stats.items():
            infos[f"curriculum/{key}"] = value

        # Get episode stats from core environment
        with self.timer("_core_env.get_episode_stats"):
            stats = self._core_env.get_episode_stats()

        # Process agent stats
        infos["game"] = stats["game"]
        infos["agent"] = {}
        for agent_stats in stats["agent"]:
            for n, v in agent_stats.items():
                infos["agent"][n] = infos["agent"].get(n, 0) + v
        for n, v in infos["agent"].items():
            infos["agent"][n] = v / self._core_env.num_agents

        # Add attributes
        attributes: Dict[str, Any] = {
            "seed": self._current_seed,
            "map_w": self._core_env.map_width,
            "map_h": self._core_env.map_height,
            "initial_grid_hash": self._core_env.initial_grid_hash,
            "steps": self._steps,
            "resets": self._resets,
            "max_steps": self._core_env.max_steps,
            "completion_time": time.time(),
        }
        infos["attributes"] = attributes

        # Handle replay writing
        replay_url = None
        with self.timer("_replay_writer"):
            if self._replay_writer and self._episode_id:
                replay_url = self._replay_writer.write_replay(self._episode_id)
                infos["replay_url"] = replay_url

        # Handle stats writing
        with self.timer("_stats_writer"):
            if self._stats_writer and self._episode_id:
                self._write_episode_stats(stats, episode_rewards, replay_url)

        # Update curriculum
        self._task.complete(episode_rewards_mean)

        # Add curriculum task probabilities
        infos["curriculum_task_probs"] = self._curriculum.get_task_probs()

        # Add timing information
        self._add_timing_info(infos)

        # Add task-specific info
        task_init_time_msec = self.timer.lap_all().get("_create_core_env", 0) * 1000
        infos.update(
            {
                f"task_reward/{self._task.short_name()}/rewards.mean": episode_rewards_mean,
                f"task_timing/{self._task.short_name()}/init_time_msec": task_init_time_msec,
            }
        )

        # Clear episode ID
        self._episode_id = None

        self.timer.stop("process_episode_stats")

    def _write_episode_stats(
        self, stats: Dict[str, Any], episode_rewards: np.ndarray, replay_url: Optional[str]
    ) -> None:
        """Write episode statistics to stats writer."""
        if not self._stats_writer or not self._episode_id or not self._core_env:
            return

        # Flatten environment config
        env_cfg_flattened: Dict[str, str] = {}
        env_cfg = OmegaConf.to_container(self._task.env_cfg(), resolve=False)
        for k, v in unroll_nested_dict(cast(Dict[str, Any], env_cfg)):
            env_cfg_flattened[f"config.{str(k).replace('/', '.')}"] = str(v)

        # Prepare agent metrics
        agent_metrics = {}
        for agent_idx, agent_stats in enumerate(stats["agent"]):
            agent_metrics[agent_idx] = {}
            agent_metrics[agent_idx]["reward"] = float(episode_rewards[agent_idx])
            for k, v in agent_stats.items():
                agent_metrics[agent_idx][k] = float(v)

        # Get agent groups
        grid_objects = self._core_env.grid_objects()
        agent_groups: Dict[int, int] = {
            v["agent_id"]: v["agent:group"] for v in grid_objects.values() if v["type"] == 0
        }

        # Record episode
        self._stats_writer.record_episode(
            self._episode_id,
            env_cfg_flattened,
            agent_metrics,
            agent_groups,
            self._core_env.max_steps,
            replay_url,
            self._reset_at,
        )

    def _add_timing_info(self, infos: Dict[str, Any]) -> None:
        """Add timing information to infos."""
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

    def render(self) -> Optional[str]:
        """Render the environment."""
        if self._renderer is None or self._core_env is None:
            return None

        return self._renderer.render(self._core_env.current_step, self._core_env.grid_objects())

    def close(self) -> None:
        """Close the environment."""
        if self._core_env is not None:
            # Clean up any resources if needed
            self._core_env = None

    # Properties that expose core environment functionality
    @property
    def done(self) -> bool:
        """Check if environment needs reset."""
        return self._should_reset

    @property
    def render_mode(self) -> Optional[str]:
        """Get render mode."""
        return self._render_mode

    @property
    def core_env(self) -> Optional[MettaGridCore]:
        """Get core environment instance."""
        return self._core_env

    # Properties that delegate to core environment
    @property
    def max_steps(self) -> int:
        if self._core_env is None:
            raise RuntimeError("Environment not initialized")
        return self._core_env.max_steps

    @property
    def num_agents(self) -> int:
        if self._core_env is None:
            raise RuntimeError("Environment not initialized")
        return self._core_env.num_agents

    @property
    def obs_width(self) -> int:
        if self._core_env is None:
            raise RuntimeError("Environment not initialized")
        return self._core_env.obs_width

    @property
    def obs_height(self) -> int:
        if self._core_env is None:
            raise RuntimeError("Environment not initialized")
        return self._core_env.obs_height

    @property
    def map_width(self) -> int:
        if self._core_env is None:
            raise RuntimeError("Environment not initialized")
        return self._core_env.map_width

    @property
    def map_height(self) -> int:
        if self._core_env is None:
            raise RuntimeError("Environment not initialized")
        return self._core_env.map_height

    @property
    def single_observation_space(self) -> spaces.Box:
        if self._core_env is None:
            raise RuntimeError("Environment not initialized")
        return self._core_env.observation_space

    @property
    def single_action_space(self) -> spaces.MultiDiscrete:
        if self._core_env is None:
            raise RuntimeError("Environment not initialized")
        return self._core_env.action_space

    @property
    def action_names(self) -> List[str]:
        if self._core_env is None:
            raise RuntimeError("Environment not initialized")
        return self._core_env.action_names

    @property
    def max_action_args(self) -> List[int]:
        if self._core_env is None:
            raise RuntimeError("Environment not initialized")
        return self._core_env.max_action_args

    @property
    def object_type_names(self) -> List[str]:
        if self._core_env is None:
            raise RuntimeError("Environment not initialized")
        return self._core_env.object_type_names

    @property
    def inventory_item_names(self) -> List[str]:
        if self._core_env is None:
            raise RuntimeError("Environment not initialized")
        return self._core_env.inventory_item_names

    @property
    def feature_normalizations(self) -> Dict[int, float]:
        if self._core_env is None:
            raise RuntimeError("Environment not initialized")
        return self._core_env.feature_normalizations

    @property
    def initial_grid_hash(self) -> int:
        if self._core_env is None:
            raise RuntimeError("Environment not initialized")
        return self._core_env.initial_grid_hash

    @property
    def action_success(self) -> List[bool]:
        if self._core_env is None:
            raise RuntimeError("Environment not initialized")
        return self._core_env.action_success

    @property
    def global_features(self) -> List[Any]:
        """Global features for compatibility."""
        return []

    def get_observation_features(self) -> Dict[str, Dict]:
        """Get observation features for policy initialization."""
        if self._core_env is None:
            raise RuntimeError("Environment not initialized")
        return self._core_env.get_observation_features()

    @property
    def grid_objects(self) -> Dict[int, Dict[str, Any]]:
        """Get grid objects information."""
        if self._core_env is None:
            raise RuntimeError("Environment not initialized")
        return self._core_env.grid_objects()
