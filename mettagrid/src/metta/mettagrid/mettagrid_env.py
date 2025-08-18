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
from typing import Any, Dict, List, Mapping, Optional, Tuple, cast

import numpy as np
import torch
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
        # Dual-policy flags/overrides (can be set by trainer or via kwargs)
        self._dual_policy_enabled: bool = bool(kwargs.pop("dual_policy_enabled", False))
        self._dual_policy_training_agents_pct: float = float(kwargs.pop("dual_policy_training_agents_pct", 0.5))
        self._dual_policy_agent_groups: Optional[list[list[int]]] = None

        # Log dual policy configuration for debugging
        if self._dual_policy_enabled:
            logger.debug(f"Dual policy ENABLED with training_agents_pct={self._dual_policy_training_agents_pct}")

        # Initialize with base PufferLib functionality
        super().__init__(
            curriculum=curriculum,
            render_mode=render_mode,
            level=level,
            buf=buf,
            **kwargs,
        )

        # Environment metadata (self._task is set by base class)
        self._cfg_labels: List[str] = self._task.env_cfg().get("labels", [])

    def _make_episode_id(self) -> str:
        """Generate unique episode ID."""
        return str(uuid.uuid4())

    def _get_game_config_for_new_task(self) -> Dict[str, Any]:
        """Fetch a new task from the curriculum, sync level and labels, and return game config dict."""
        self._task = self._curriculum.get_task()
        task_cfg = self._task.env_cfg()
        game_config_dict = cast(Dict[str, Any], OmegaConf.to_container(task_cfg.game))
        assert isinstance(game_config_dict, dict), "Game config must be a dictionary"
        # Sync level with task config
        self._level = task_cfg.game.map_builder.build()
        self._map_labels = self._level.labels
        return game_config_dict

    def _bind_shared_buffers(self) -> None:
        """Bind shared buffers to the C++ environment if available (required for performance)."""
        if hasattr(self, "observations") and self._c_env_instance:
            self._c_env_instance.set_buffers(self.observations, self.terminals, self.truncations, self.rewards)

    def _create_and_bind_c_env(self, game_config_dict: Dict[str, Any], seed: Optional[int]) -> MettaGridCpp:
        """Create a C++ env and immediately bind shared buffers."""
        self._c_env_instance = self._create_c_env(game_config_dict, seed)
        self._bind_shared_buffers()
        return self._c_env_instance

    def _resolve_dual_policy_groups(self, ensure_each: bool) -> list[list[int]]:
        """Resolve or synthesize dual-policy agent groups.

        If grid-provided groups are missing or incomplete, generate NPC-first and trained-second groups.
        When ensure_each is True, guarantee at least one agent in each group (if possible).
        """
        # Prefer groups from grid objects
        agent_groups = self._get_agent_groups()
        if agent_groups and len(agent_groups) >= 2:
            self._dual_policy_agent_groups = agent_groups
            return agent_groups

        # Fall back to cached or synthesized groups
        if hasattr(self, "_dual_policy_agent_groups") and self._dual_policy_agent_groups:
            return self._dual_policy_agent_groups

        if self._c_env_instance is None:
            return []

        num_agents = self._c_env_instance.num_agents
        num_trained = int(round(num_agents * self._dual_policy_training_agents_pct))
        if ensure_each:
            num_trained = max(1, min(num_agents - 1, num_trained))
        else:
            num_trained = max(0, min(num_agents, num_trained))

        trained_ids = list(range(num_trained))
        npc_ids = list(range(num_trained, num_agents))
        agent_groups = [npc_ids, trained_ids]
        self._dual_policy_agent_groups = agent_groups
        return agent_groups

    def _reset_trial(self) -> None:
        """Reset the environment for a new trial within the same episode."""
        # Get new task and create new C++ environment for new trial
        game_config_dict = self._get_game_config_for_new_task()
        self._create_and_bind_c_env(game_config_dict, self._current_seed)

        # Reset counters for new trial
        self._steps = 0

        # Set up new trial tracking
        self._trial_id = self._make_episode_id()
        self._reset_at = datetime.datetime.now()

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

        # Get new task and synced level
        game_config_dict = self._get_game_config_for_new_task()

        # Recreate C++ environment for new task (after first reset)
        if self._resets > 0:
            self._create_and_bind_c_env(game_config_dict, seed)

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
            self._create_and_bind_c_env(game_config_dict, seed)

        # Get initial observations from core environment
        if self._c_env_instance is None:
            raise RuntimeError("Core environment not initialized")
        observations, info = self._c_env_instance.reset()

        # If dual-policy is enabled and no core groups are defined, synthesize groups per-env
        if self._dual_policy_enabled and not self._get_agent_groups():
            agent_groups = self._resolve_dual_policy_groups(ensure_each=False)
            if len(agent_groups) >= 2:
                npc_ids, trained_ids = agent_groups[0], agent_groups[1]
                logger.debug(
                    f"Dual policy groups set in reset: NPC={len(npc_ids)} agents, Trained={len(trained_ids)} agents"
                )

        self.timer.start("thread_idle")
        return observations, info

    def _compute_dual_policy_stats(self) -> dict:
        """Compute dual policy stats for current step (not just episode end)."""
        stats = {}
        if not self._dual_policy_enabled or self._c_env_instance is None:
            return stats

        # Resolve or synthesize agent groups
        agent_groups = self._resolve_dual_policy_groups(ensure_each=True)

        if len(agent_groups) >= 2:
            npc_group = agent_groups[0]
            trained_group = agent_groups[1]

            # Get rewards (use episode rewards as proxy for per-step tracking)
            try:
                # Use cumulative episode rewards divided by steps as a proxy for step rewards
                step_rewards = self._c_env_instance.get_episode_rewards()
                if self._steps > 0:
                    # Average reward per step so far
                    step_rewards = step_rewards / max(1, self._steps)
            except Exception as e:
                # If we can't get rewards, just return empty stats
                logger.warning(f"[MettaGridEnv] Could not get rewards for dual policy stats: {e}")
                return stats

            if step_rewards is not None and len(step_rewards) > 0:
                # Safely gather rewards for each group with bounds validation
                is_torch_tensor = torch.is_tensor(step_rewards)
                total_agents = len(step_rewards)

                def _safe_group_rewards(group: list[int]):
                    if not group:
                        if is_torch_tensor:
                            sr_t = cast(torch.Tensor, step_rewards)
                            return torch.zeros_like(sr_t[:1])
                        else:
                            sr_np = cast(np.ndarray, step_rewards)
                            return np.zeros(1, dtype=sr_np.dtype)

                    valid_indices = [
                        idx for idx in group if isinstance(idx, (int, np.integer)) and 0 <= idx < total_agents
                    ]

                    if not valid_indices:
                        if is_torch_tensor:
                            sr_t = cast(torch.Tensor, step_rewards)
                            return torch.zeros_like(sr_t[:1])
                        else:
                            sr_np = cast(np.ndarray, step_rewards)
                            return np.zeros(1, dtype=sr_np.dtype)

                    try:
                        return step_rewards[valid_indices]
                    except Exception as e:
                        logger.warning(
                            (f"[MettaGridEnv] Failed to index step_rewards with indices {valid_indices}: {e}")
                        )
                        if is_torch_tensor:
                            sr_t = cast(torch.Tensor, step_rewards)
                            return torch.zeros_like(sr_t[:1])
                        else:
                            sr_np = cast(np.ndarray, step_rewards)
                            return np.zeros(1, dtype=sr_np.dtype)

                # NPC group stats
                npc_rewards = _safe_group_rewards(npc_group)
                stats["dual_policy/npc/step_reward_mean"] = npc_rewards.mean().item()
                stats["dual_policy/npc/step_reward_sum"] = npc_rewards.sum().item()

                # Trained policy group stats
                trained_rewards = _safe_group_rewards(trained_group)
                stats["dual_policy/trained/step_reward_mean"] = trained_rewards.mean().item()
                stats["dual_policy/trained/step_reward_sum"] = trained_rewards.sum().item()

                # Combined stats
                stats["dual_policy/combined/step_reward_mean"] = step_rewards.mean().item()
                stats["dual_policy/combined/step_reward_sum"] = step_rewards.sum().item()

        return stats

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

        # Add dual policy stats for every step (not just episode completion)
        dual_policy_step_stats = self._compute_dual_policy_stats()
        infos.update(dual_policy_step_stats)

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
        if self._task.env_cfg().get("desync_episodes", True) and self._is_training and self._resets == 0:
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
        for label in self._map_labels + self._cfg_labels:
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

        # Add dual-policy specific logging if enabled
        if self._dual_policy_enabled:
            agent_groups = self._resolve_dual_policy_groups(ensure_each=True)

            if len(agent_groups) >= 2:
                npc_group = agent_groups[0]  # First group is NPC
                trained_group = agent_groups[1]  # Second group is trained policy

                # NPC group stats
                npc_rewards = episode_rewards[npc_group]
                npc_hearts = self._get_agent_hearts(npc_group)
                infos["dual_policy/npc/reward_mean"] = npc_rewards.mean().item()
                infos["dual_policy/npc/reward_sum"] = npc_rewards.sum().item()
                infos["dual_policy/npc/hearts_mean"] = npc_hearts.mean().item()
                infos["dual_policy/npc/hearts_sum"] = npc_hearts.sum().item()
                infos["dual_policy/npc/num_agents"] = len(npc_group)

                # Trained policy group stats
                trained_rewards = episode_rewards[trained_group]
                trained_hearts = self._get_agent_hearts(trained_group)
                infos["dual_policy/trained/reward_mean"] = trained_rewards.mean().item()
                infos["dual_policy/trained/reward_sum"] = trained_rewards.sum().item()
                infos["dual_policy/trained/hearts_mean"] = trained_hearts.mean().item()
                infos["dual_policy/trained/hearts_sum"] = trained_hearts.sum().item()
                infos["dual_policy/trained/num_agents"] = len(trained_group)

                # Combined stats
                infos["dual_policy/combined/reward_mean"] = episode_rewards_mean
                infos["dual_policy/combined/reward_sum"] = episode_rewards_sum
                infos["dual_policy/combined/hearts_mean"] = self._get_agent_hearts().mean().item()
                infos["dual_policy/combined/hearts_sum"] = self._get_agent_hearts().sum().item()
                infos["dual_policy/combined/num_agents"] = self._c_env_instance.num_agents

                # Aliases for external dashboards (policy_a = trained, policy_b = npc)
                # Reward totals
                infos["dual_policy/policy_a_reward_total"] = trained_rewards.sum().item()
                infos["dual_policy/policy_b_reward_total"] = npc_rewards.sum().item()
                infos["dual_policy/combined_reward_total"] = episode_rewards_sum
                # Reward means
                infos["dual_policy/policy_a_reward_mean"] = trained_rewards.mean().item()
                infos["dual_policy/policy_b_reward_mean"] = npc_rewards.mean().item()
                infos["dual_policy/combined_reward_mean"] = episode_rewards_mean
                # Hearts totals and means
                infos["dual_policy/policy_a_hearts_total"] = trained_hearts.sum().item()
                infos["dual_policy/policy_b_hearts_total"] = npc_hearts.sum().item()
                infos["dual_policy/policy_a_hearts_mean"] = trained_hearts.mean().item()
                infos["dual_policy/policy_b_hearts_mean"] = npc_hearts.mean().item()
                infos["dual_policy/combined_hearts_total"] = self._get_agent_hearts().sum().item()
                infos["dual_policy/combined_hearts_mean"] = self._get_agent_hearts().mean().item()
                # Agent counts
                infos["dual_policy/policy_a_num_agents"] = len(trained_group)
                infos["dual_policy/policy_b_num_agents"] = len(npc_group)

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
        self, stats: Mapping[str, Any], episode_rewards: np.ndarray, replay_url: Optional[str]
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

    # PufferLib compatibility properties provided by base class (no overrides here)

    # Use base class properties for observation/action spaces, dimensions, names, and feature metadata

    def _get_agent_groups(self) -> list[list[int]]:
        """Get agent groups for dual-policy logging.

        Returns:
            List of agent groups, where each group is a list of agent IDs.
            First group is assumed to be NPC, second group is trained policy.
        """
        # Get agent groups from grid objects
        grid_objects: Dict[int, Any] = self.c_env.grid_objects()
        agent_groups: Dict[int, int] = {
            v["agent_id"]: v["agent:group"] for v in grid_objects.values() if v["type"] == 0
        }

        if not agent_groups:
            return []

        # Group agents by their group ID
        group_agent_ids = {}
        for agent_id, group_id in agent_groups.items():
            if group_id not in group_agent_ids:
                group_agent_ids[group_id] = []
            group_agent_ids[group_id].append(agent_id)

        # Return groups sorted by group ID
        return [group_agent_ids[group_id] for group_id in sorted(group_agent_ids.keys())]

    def _get_agent_hearts(self, agent_ids: Optional[list[int]] = None) -> np.ndarray:
        """Get hearts for specified agents or all agents.

        Args:
            agent_ids: List of agent IDs to get hearts for. If None, gets all agents.

        Returns:
            Array of heart counts for the specified agents.
        """
        # Get agent stats
        # Pull per-agent stats from episode stats dict (agent list of dicts)
        stats = self.c_env.get_episode_stats()
        agent_stats = stats.get("agent", [])

        if agent_ids is None:
            # Get hearts for all agents
            hearts = []
            for agent_stat in agent_stats:
                hearts.append(agent_stat.get("inventory.heart", 0))
            return np.array(hearts)
        else:
            # Get hearts for specified agents
            hearts = []
            for agent_id in agent_ids:
                if agent_id < len(agent_stats):
                    hearts.append(agent_stats[agent_id].get("inventory.heart", 0))
                else:
                    hearts.append(0)
            return np.array(hearts)

    # Base class already defines `emulated` property
