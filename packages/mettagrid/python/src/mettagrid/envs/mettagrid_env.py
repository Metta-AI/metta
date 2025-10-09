"""MettaGridEnv - Training-specific Python environment class.

This class provides Metta's custom training environment, built on PufferLib
for high-performance vectorized training. Includes stats writing, replay writing,
and episode tracking functionality."""

from __future__ import annotations

import datetime
import logging
import time
import uuid
from typing import TYPE_CHECKING, Any, Dict, Optional, Tuple, cast

import numpy as np
from pydantic import validate_call
from typing_extensions import override

from mettagrid.config.mettagrid_config import MettaGridConfig
from mettagrid.envs.puffer_base import MettaGridPufferBase
from mettagrid.profiling.stopwatch import Stopwatch, with_instance_timer
from mettagrid.util.dict_utils import unroll_nested_dict
from mettagrid.util.replay_writer import ReplayWriter
from mettagrid.util.stats_writer import StatsWriter

if TYPE_CHECKING:
    from mettagrid.mettagrid_c import EpisodeStats

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
        env_cfg: MettaGridConfig,
        render_mode: Optional[str] = None,
        stats_writer: Optional[StatsWriter] = None,
        replay_writer: Optional[ReplayWriter] = None,
        is_training: bool = False,
    ):
        """Initialize MettaGridEnv for training."""
        # Add training-specific attributes first (needed by MettaGridCore)
        self.timer = Stopwatch(log_level=logger.getEffectiveLevel())
        self.timer.start()
        self.timer.start("thread_idle")
        self._steps = 0
        self._resets = 0
        self._stats_writer = stats_writer
        self._replay_writer = replay_writer
        self._episode_id: str | None = None
        self._last_reset_ts = datetime.datetime.now()
        self._is_training = is_training
        self._label_completions = {"completed_tasks": [], "completion_rates": {}}
        self.per_label_rewards = {}
        self._visited_positions: Dict[int, set[tuple[int, int]]] = {}  # Track unique (r, c) per agent

        # DesyncEpisodes - when training we want to stagger experience. The first episode
        # will end early so that the next episode can begin at a different time on each worker.
        self._early_reset: int | None = None
        if self._is_training and env_cfg.desync_episodes:
            self._early_reset = int(np.random.randint(1, env_cfg.game.max_steps))

        # Initialize MettaGridPufferBase
        super().__init__(
            env_cfg,
            render_mode=render_mode,
        )

    def _make_episode_id(self) -> str:
        """Generate unique episode ID."""
        return str(uuid.uuid4())

    @override
    @with_instance_timer("reset")
    def reset(self, seed: Optional[int] = None) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Reset the environment for training."""
        self.timer.stop("thread_idle")

        # Reset counters
        self._steps = 0
        self._resets += 1

        # Set up episode tracking
        self._episode_id = self._make_episode_id()
        self._last_reset_ts = datetime.datetime.now()
        self._visited_positions = {}  # Reset pixels explored tracking

        # Start replay recording if enabled
        if self._replay_writer and self._episode_id:
            self._replay_writer.start_episode(self._episode_id, self)

        observations, info = super().reset(seed)

        self.timer.start("thread_idle")
        return observations, info

    @override
    @with_instance_timer("step")
    def step(self, actions: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, Dict[str, Any]]:
        """Execute one timestep for training."""
        self.timer.stop("thread_idle")

        with self.timer("_c_env.step"):
            observations, rewards, terminals, truncations, infos = super().step(actions)
            self._steps += 1

        if self._replay_writer and self._episode_id:
            with self.timer("_replay_writer.log_step"):
                self._replay_writer.log_step(self._episode_id, actions, rewards)

        # Track pixels explored
        with self.timer("_track_pixels_explored"):
            grid_objects = self.grid_objects()
            for _obj_id, obj_data in grid_objects.items():
                if obj_data.get("type") == 0:  # Agent type
                    agent_id = obj_data.get("agent_id")
                    if agent_id is not None:
                        pos = (obj_data["r"], obj_data["c"])
                        if agent_id not in self._visited_positions:
                            self._visited_positions[agent_id] = set()
                        self._visited_positions[agent_id].add(pos)

        # Handle early reset for #DesyncEpisodes
        if self._early_reset is not None and self._steps >= self._early_reset:
            truncations[:] = True
            self._early_reset = None

        infos = {}
        if terminals.all() or truncations.all():
            self._process_episode_completion(infos)

        self.timer.start("thread_idle")
        return observations, rewards, terminals, truncations, infos

    def _update_label_completions(self, moving_avg_window: int = 500) -> None:
        """Update label completions."""
        label = self.mg_config.label

        # keep track of a list of the last 500 labels
        if len(self._label_completions["completed_tasks"]) >= moving_avg_window:
            self._label_completions["completed_tasks"].pop(0)
        self._label_completions["completed_tasks"].append(label)

        # moving average of the completion rates
        self._label_completions["completion_rates"] = {t: 0 for t in set(self._label_completions["completed_tasks"])}
        for t in self._label_completions["completed_tasks"]:
            self._label_completions["completion_rates"][t] += 1
        self._label_completions["completion_rates"] = {
            t: self._label_completions["completion_rates"][t] / len(self._label_completions["completed_tasks"])
            for t in self._label_completions["completion_rates"]
        }

    def _process_episode_completion(self, infos: Dict[str, Any], moving_avg_window: int = 500, alpha=0.9) -> None:
        """Process episode completion - stats, etc."""
        self.timer.start("process_episode_stats")

        # Clear any existing infos
        infos.clear()

        # Get episode rewards and stats
        episode_rewards = self.get_episode_rewards()

        # Get episode stats from core environment
        with self.timer("_c_env.get_episode_stats"):
            stats = self.get_episode_stats()

        # Process agent stats
        infos["game"] = stats["game"]
        infos["agent"] = {}
        for agent_stats in stats["agent"]:
            for n, v in agent_stats.items():
                infos["agent"][n] = infos["agent"].get(n, 0) + v
        for n, v in infos["agent"].items():
            infos["agent"][n] = v / self.num_agents

        # Add pixels explored metric
        total_pixels_explored = sum(len(positions) for positions in self._visited_positions.values())
        avg_pixels_explored = total_pixels_explored / self.num_agents if self.num_agents > 0 else 0
        infos["agent"]["pixels_explored"] = avg_pixels_explored

        # If reward estimates are set, plot them compared to the mean reward
        if self.mg_config.game.reward_estimates:
            infos["reward_estimates"] = {}
            infos["reward_estimates"]["best_case_optimal_diff"] = (
                self.mg_config.game.reward_estimates["best_case_optimal_reward"] - episode_rewards.mean()
            )
            infos["reward_estimates"]["worst_case_optimal_diff"] = (
                self.mg_config.game.reward_estimates["worst_case_optimal_reward"] - episode_rewards.mean()
            )

        self._update_label_completions(moving_avg_window)

        # only plot label completions once we have a full moving average window, to prevent initial bias
        if len(self._label_completions["completed_tasks"]) >= 50:
            infos["label_completions"] = self._label_completions["completion_rates"]
        self.per_label_rewards[self.mg_config.label] = episode_rewards.mean()
        infos["per_label_rewards"] = self.per_label_rewards

        # Add attributes
        attributes: Dict[str, Any] = {
            "seed": self._current_seed,
            "map_w": self.map_width,
            "map_h": self.map_height,
            "initial_grid_hash": self.initial_grid_hash,
            "steps": self._steps,
            "resets": self._resets,
            "max_steps": self.max_steps,
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

        # Add timing information
        self._add_timing_info(infos)

        # Clear episode ID
        self._episode_id = None

        self.timer.stop("process_episode_stats")

    def _write_episode_stats(self, stats: EpisodeStats, episode_rewards: np.ndarray, replay_url: Optional[str]) -> None:
        """Write episode statistics to stats writer."""
        if not self._stats_writer or not self._episode_id:
            return

        # Flatten environment config
        env_cfg_flattened: Dict[str, str] = {}
        env_cfg = self.mg_config.model_dump()
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
        grid_objects = self.grid_objects()
        agent_groups: Dict[int, int] = {
            v["agent_id"]: v["agent:group"] for v in grid_objects.values() if v["type"] == 0
        }

        # Record episode
        self._stats_writer.record_episode(
            self._episode_id,
            env_cfg_flattened,
            agent_metrics,
            agent_groups,
            self.max_steps,
            replay_url,
            self._last_reset_ts,
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

    def close(self) -> None:
        """Close the environment."""
        super().close()
        if self._stats_writer:
            self._stats_writer.close()
