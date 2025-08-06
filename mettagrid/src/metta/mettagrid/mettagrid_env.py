"""
MettaGridEnv - Training-specific Python environment class.

This class provides Metta's custom training environment, built on PufferLib
for high-performance vectorized training. Includes stats writing, replay writing,
and episode tracking functionality.
"""

from __future__ import annotations

import datetime
import logging
import time
import uuid
from typing import Any, Dict, List, Optional, Tuple, cast

import numpy as np
from omegaconf import OmegaConf
from pydantic import validate_call
from typing_extensions import override

from metta.common.profiling.stopwatch import Stopwatch, with_instance_timer
from metta.mettagrid.curriculum.core import Curriculum
from metta.mettagrid.level_builder import Level
from metta.mettagrid.mettagrid_c import MettaGrid as MettaGridCpp
from metta.mettagrid.puffer_base import MettaGridPufferBase
from metta.mettagrid.replay_writer import ReplayWriter
from metta.mettagrid.stats_writer import StatsWriter
from metta.mettagrid.util.dict_utils import unroll_nested_dict

logger = logging.getLogger("MettaGridEnv")


class MettaGridEnv(MettaGridPufferBase):
    """
    Main MettaGrid environment class for training.

    Inherits from MettaGridCore and PufferEnv, adding training-specific features like stats writing,
    replay writing, and episode tracking. This class is tightly coupled to PufferLib for
    vectorization support in the training system.
    """

    @validate_call(config={"arbitrary_types_allowed": True})
    def __init__(
        self,
        curriculum: Curriculum,
        render_mode: Optional[str] = None,
        level: Optional[Level] = None,
        buf: Optional[Any] = None,
        stats_writer: Optional[StatsWriter] = None,
        replay_writer: Optional[ReplayWriter] = None,
        is_training: bool = False,
        **kwargs: Any,
    ):
        """
        Initialize MettaGridEnv for training.

        Args:
            curriculum: Curriculum for task management
            render_mode: Rendering mode (None, "human", "miniscope")
            level: Optional pre-built level
            buf: PufferLib buffer object
            stats_writer: Optional stats writer
            replay_writer: Optional replay writer
            is_training: Whether this is for training
            **kwargs: Additional arguments
        """
        # Add training-specific attributes first (needed by MettaGridCore)
        self.timer = Stopwatch(logger)
        self.timer.start()
        self.timer.start("thread_idle")
        self._steps = 0
        self._resets = 0
        self._stats_writer = stats_writer
        self._replay_writer = replay_writer
        self._episode_id: str | None = None
        self._reset_at = datetime.datetime.now()
        self._is_training = is_training

        # Initialize with base PufferLib functionality
        super().__init__(
            curriculum=curriculum,
            render_mode=render_mode,
            level=level,
            buf=buf,
            **kwargs,
        )

        # Environment metadata (self._task is set by base class)
        self.labels: List[str] = self._task.env_cfg().get("labels", [])

    def _make_episode_id(self) -> str:
        """Generate unique episode ID."""
        return str(uuid.uuid4())

    def _reset_trial(self) -> None:
        """Reset the environment for a new trial within the same episode."""
        # Get new task from curriculum (for new trial)
        self._task = self._curriculum.get_task()
        task_cfg = self._task.env_cfg()
        game_config_dict = OmegaConf.to_container(task_cfg.game)
        assert isinstance(game_config_dict, dict), "Game config must be a dictionary"

        # Create new C++ environment for new trial
        self._c_env_instance = self._create_c_env(game_config_dict, self._current_seed)

        # Reset counters for new trial
        self._steps = 0

        # Set up new trial tracking
        self._trial_id = self._make_episode_id()
        self._reset_at = datetime.datetime.now()

        # CRITICAL: Set buffers once after C++ env creation, before any operations
        # This establishes shared memory for high-performance training (400k+ SPS)
        if hasattr(self, "observations") and self._c_env_instance:
            self._c_env_instance.set_buffers(self.observations, self.terminals, self.truncations, self.rewards)

        # Start replay recording for new trial if enabled
        if self._replay_writer and self._trial_id:
            self._replay_writer.start_episode(self._trial_id, self)

        # Get initial observations for new trial
        if self._c_env_instance is None:
            raise RuntimeError("Core environment not initialized")
        self._c_env_instance.reset()

    @override
    @with_instance_timer("reset")
    def reset(self, seed: Optional[int] = None) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Reset the environment for training.

        Args:
            seed: Random seed

        Returns:
            Tuple of (observations, info)
        """
        self.timer.stop("thread_idle")

        # Get new task from curriculum
        self._task = self._curriculum.get_task()
        task_cfg = self._task.env_cfg()
        game_config_dict = OmegaConf.to_container(task_cfg.game)
        assert isinstance(game_config_dict, dict), "Game config must be a dictionary"

        # Recreate C++ environment for new task (after first reset)
        if self._resets > 0:
            self._c_env_instance = self._create_c_env(game_config_dict, seed)

            # CRITICAL: Set buffers once after C++ env recreation
            # This establishes shared memory for high-performance training (400k+ SPS)
            if hasattr(self, "observations") and self._c_env_instance:
                self._c_env_instance.set_buffers(self.observations, self.terminals, self.truncations, self.rewards)

        # Reset counters
        self._steps = 0
        self._resets += 1

        # Set up episode tracking
        self._episode_id = self._make_episode_id()
        self._trial_id = self._episode_id  # For compatibility with trial-based logic
        self._current_seed = seed or 0
        self._reset_at = datetime.datetime.now()

        # Start replay recording if enabled
        if self._replay_writer and self._episode_id:
            self._replay_writer.start_episode(self._episode_id, self)

        # Reset flags
        self._should_reset = False

        # Create initial C++ environment if this is the first reset
        if self._resets == 1:
            self._c_env_instance = self._create_c_env(game_config_dict, seed)

            # CRITICAL: Set buffers once after C++ env creation, before any operations
            # This establishes shared memory for high-performance training (400k+ SPS)
            if hasattr(self, "observations") and self._c_env_instance:
                self._c_env_instance.set_buffers(self.observations, self.terminals, self.truncations, self.rewards)

        # Get initial observations from core environment
        if self._c_env_instance is None:
            raise RuntimeError("Core environment not initialized")
        observations, info = self._c_env_instance.reset()

        self.timer.start("thread_idle")
        return observations, info

    @override
    @with_instance_timer("step")
    def step(self, actions: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, Dict[str, Any]]:
        """
        Execute one timestep for training.

        Args:
            actions: Array of actions

        Returns:
            Tuple of (observations, rewards, terminals, truncations, infos)
        """
        self.timer.stop("thread_idle")

        if self._c_env_instance is None:
            raise RuntimeError("Environment not initialized. Call reset() first.")

        # Execute step directly on C++ environment to maintain buffer sharing
        # This is critical for PufferEnv performance - we must use the shared buffers
        with self.timer("_c_env.step"):
            self._c_env_instance.step(actions)
            self._steps += 1

        # Record step for replay (use shared PufferEnv buffers)
        if self._replay_writer and self._episode_id:
            with self.timer("_replay_writer.log_step"):
                self._replay_writer.log_step(self._trial_id, actions, self.rewards)

        # Check for episode completion (use shared PufferEnv buffers)
        infos = {}

        if self.terminals.all() or self.truncations.all():
            self._process_episode_completion(infos)
            # Note: _process_episode_completion already calls complete_trial()
            if self._task.is_complete():
                self._should_reset = True
                # Add curriculum task probabilities to infos for distributed logging
                infos["curriculum_task_probs"] = self._curriculum.get_task_probs()
            else:
                # Continue with new trial in same episode (like upstream)
                self._reset_trial()

        self.timer.start("thread_idle")
        return self.observations, self.rewards, self.terminals, self.truncations, infos

    def _create_c_env(self, game_config_dict: Dict[str, Any], seed: Optional[int] = None) -> MettaGridCpp:
        """
        Create a new MettaGridCpp instance with training-specific features.

        Args:
            game_config_dict: Game configuration dictionary
            seed: Random seed for environment

        Returns:
            New MettaGridCpp instance
        """
        # Handle episode desyncing for training
        if self._task.env_cfg().get("desync_episodes") and self._is_training and self._resets == 0:
            max_steps = game_config_dict["max_steps"]
            # Recreate with random max_steps
            game_config_dict = game_config_dict.copy()  # Don't modify original
            game_config_dict["max_steps"] = int(np.random.randint(1, max_steps + 1))

        return super()._create_c_env(game_config_dict, seed)

    def _process_episode_completion(self, infos: Dict[str, Any]) -> None:
        """Process episode completion - stats, curriculum, etc."""
        if self._c_env_instance is None:
            return

        self.timer.start("process_episode_stats")

        # Clear any existing infos
        infos.clear()

        # Get episode rewards and stats
        episode_rewards = self._c_env_instance.get_episode_rewards()
        episode_rewards_sum = episode_rewards.sum()
        episode_rewards_mean = episode_rewards_sum / self._c_env_instance.num_agents

        # Add map and label rewards
        for label in self._map_labels + self.labels:
            infos[f"map_reward/{label}"] = episode_rewards_mean

        # Add curriculum stats
        infos.update(self._curriculum.get_completion_rates())
        curriculum_stats = self._curriculum.get_curriculum_stats()
        for key, value in curriculum_stats.items():
            infos[f"curriculum/{key}"] = value

        # Get episode stats from core environment
        with self.timer("_c_env.get_episode_stats"):
            stats = self._c_env_instance.get_episode_stats()

        # Process agent stats
        infos["game"] = stats["game"]
        infos["agent"] = {}
        for agent_stats in stats["agent"]:
            for n, v in agent_stats.items():
                infos["agent"][n] = infos["agent"].get(n, 0) + v
        for n, v in infos["agent"].items():
            infos["agent"][n] = v / self._c_env_instance.num_agents

        # Add attributes
        attributes: Dict[str, Any] = {
            "seed": self._current_seed,
            "map_w": self._c_env_instance.map_width,
            "map_h": self._c_env_instance.map_height,
            "initial_grid_hash": self._c_env_instance.initial_grid_hash,
            "steps": self._steps,
            "resets": self._resets,
            "max_steps": self._c_env_instance.max_steps,
            "completion_time": time.time(),
        }
        infos["attributes"] = attributes

        # keep track of per-trial rewards
        if self._task.env_cfg().get("num_trials", 1) > 1:
            current_trial = self._task.env_cfg().current_trial
            infos[f"trial_{current_trial}_reward"] = episode_rewards_mean

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
        self._task.complete_trial(episode_rewards_mean)

        # Add curriculum task probabilities
        infos["curriculum_task_probs"] = self._curriculum.get_task_probs()

        # Add timing information
        self._add_timing_info(infos)

        # Add task-specific info
        task_init_time_msec = self.timer.lap_all().get("_create_c_env", 0) * 1000
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
        if not self._stats_writer or not self._episode_id or not self._c_env_instance:
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
        grid_objects = self._c_env_instance.grid_objects()
        agent_groups: Dict[int, int] = {
            v["agent_id"]: v["agent:group"] for v in grid_objects.values() if v["type"] == 0
        }

        # Record episode
        self._stats_writer.record_episode(
            self._episode_id,
            env_cfg_flattened,
            agent_metrics,
            agent_groups,
            self._c_env_instance.max_steps,
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

        wall_time_for_lap = sum(lap_times.values()) + lap_thread_idle_time
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

    # PufferLib compatibility properties for training
    @property
    def single_observation_space(self):
        """Single agent observation space for PufferLib."""
        return self._observation_space

    @property
    def single_action_space(self):
        """Single agent action space for PufferLib."""
        return self._action_space

    @property
    def emulated(self) -> bool:
        """Native envs do not use emulation (PufferLib compatibility)."""
        return False
