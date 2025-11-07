import datetime
import time
from collections import defaultdict
from typing import Any, Dict, Sequence, cast

import numpy as np

from mettagrid.simulator import SimulatorEventHandler
from mettagrid.util.dict_utils import unroll_nested_dict
from mettagrid.util.stats_writer import StatsWriter


class StatsTracker(SimulatorEventHandler):
    """Tracker for recording statistics."""

    def __init__(self, stats_writer: StatsWriter):
        super().__init__()
        self._stats_writer = stats_writer
        self._episode_start_ts = datetime.datetime.now()
        self._episode_end_ts = None
        self._label_completions = {"completed_tasks": [], "completion_rates": {}}
        self._per_label_rewards = {}

    @staticmethod
    def _gini_coefficient(values: Sequence[float]) -> float:
        """Compute the Gini coefficient for a sequence of values."""

        arr = np.asarray(values, dtype=np.float32)
        if arr.size <= 1 or np.allclose(arr, arr[0]):
            return 0.0

        min_value = float(arr.min())
        shifted = arr - min_value if min_value < 0.0 else arr
        if np.allclose(shifted, 0.0):
            return 0.0

        mean = float(shifted.mean())
        if np.isclose(mean, 0.0):
            return 0.0

        diff_sum = np.abs(shifted[:, None] - shifted[None, :]).sum()
        n = shifted.size
        return float(diff_sum / (2.0 * n * n * mean))

    @classmethod
    def _compute_fairness_metrics(cls, reward_list: Sequence[float]) -> tuple[float, float, float]:
        """Return fairness metrics (gap, std, gini) for provided rewards."""

        rewards_arr = np.asarray(reward_list, dtype=np.float32)
        if rewards_arr.size <= 1:
            return 0.0, 0.0, 0.0

        fairness_gap = float(rewards_arr.max() - rewards_arr.min())
        fairness_std = float(rewards_arr.std(ddof=0))
        fairness_gini = cls._gini_coefficient(rewards_arr)
        return fairness_gap, fairness_std, fairness_gini

    def on_episode_start(self) -> None:
        super().on_episode_start()
        assert self._sim is not None
        self._sim._context["infos"] = {}
        self._episode_start_ts = datetime.datetime.now()

    def on_episode_end(self) -> None:
        super().on_episode_end()
        assert self._sim is not None

        # Get episode rewards and stats
        episode_rewards = self._sim.episode_rewards
        mean_reward = episode_rewards.mean()

        stats = self._sim.episode_stats
        config = self._sim.config
        num_agents = config.game.num_agents
        infos = self._sim._context["infos"]

        # Process agent stats
        infos["game"] = stats["game"]
        infos["agent"] = {}
        for agent_stats in stats["agent"]:
            for n, v in agent_stats.items():
                infos["agent"][n] = infos["agent"].get(n, 0) + v
        for n, v in infos["agent"].items():
            infos["agent"][n] = v / num_agents

        # If reward estimates are set, plot them compared to the mean reward
        if config.game.reward_estimates:
            infos["reward_estimates"] = {}
            infos["reward_estimates"]["best_case_optimal_diff"] = (
                config.game.reward_estimates["best_case_optimal_reward"] - mean_reward
            )
            infos["reward_estimates"]["worst_case_optimal_diff"] = (
                config.game.reward_estimates["worst_case_optimal_reward"] - mean_reward
            )

        self._update_label_completions()

        # only plot label completions once we have a full moving average window, to prevent initial bias
        if len(self._label_completions["completed_tasks"]) >= 50:
            infos["label_completions"] = self._label_completions["completion_rates"]
        self._per_label_rewards[config.label] = mean_reward
        infos["per_label_rewards"] = self._per_label_rewards

        # Add attributes
        attributes: Dict[str, Any] = {
            "seed": self._sim.seed,
            "map_w": self._sim.map_width,
            "map_h": self._sim.map_height,
            "steps": self._sim.current_step,
            "max_steps": self._sim.config.game.max_steps,
            "completion_time": time.time(),
        }
        infos["attributes"] = attributes

        # Add timing information
        self._add_timing_info()

        # Flatten environment config
        env_cfg_flattened: Dict[str, str] = {}
        env_cfg = self._sim.config.model_dump()
        for k, v in unroll_nested_dict(cast(Dict[str, Any], env_cfg)):
            env_cfg_flattened[f"config.{str(k).replace('/', '.')}"] = str(v)

        # Prepare agent metrics
        agent_metrics: Dict[int, Dict[str, float]] = {}
        for agent_idx, agent_stats in enumerate(stats["agent"]):
            agent_metrics[agent_idx] = {}
            agent_metrics[agent_idx]["reward"] = float(self._sim.episode_rewards[agent_idx])
            for k, v in agent_stats.items():
                agent_metrics[agent_idx][k] = float(v)

        # Get agent groups
        grid_objects = self._sim.grid_objects(ignore_types=["wall"])
        agent_groups: Dict[int, int] = {
            obj["agent_id"]: obj["agent:group"]
            for obj in grid_objects.values()
            if obj.get("type_name") == "agent" and "agent_id" in obj and "agent:group" in obj
        }

        # Compute fairness metrics within agent groups (or globally if no groups)
        if agent_metrics:
            grouped_rewards: Dict[int, list[float]] = defaultdict(list)

            if agent_groups:
                for agent_id, metrics in agent_metrics.items():
                    group_id = agent_groups.get(agent_id, -1)
                    grouped_rewards[group_id].append(metrics["reward"])
            else:
                grouped_rewards[-1].extend(metrics["reward"] for metrics in agent_metrics.values())

            for group_id, reward_list in grouped_rewards.items():
                if not reward_list:
                    continue
                fairness_gap, fairness_std, fairness_gini = self._compute_fairness_metrics(reward_list)

                if group_id == -1 and agent_groups:
                    target_agents = [agent_id for agent_id in agent_metrics.keys() if agent_id not in agent_groups]
                elif group_id == -1:
                    target_agents = list(agent_metrics.keys())
                else:
                    target_agents = [agent_id for agent_id, g_id in agent_groups.items() if g_id == group_id]

                for agent_id in target_agents:
                    agent_metrics[agent_id]["reward_fairness_gap"] = fairness_gap
                    agent_metrics[agent_id]["reward_fairness_std"] = fairness_std
                    agent_metrics[agent_id]["reward_fairness_gini"] = fairness_gini

        # Record episode
        self._stats_writer.record_episode(
            env_cfg_flattened,
            agent_metrics,
            agent_groups,
            self._sim.current_step,
            self._sim._context.get("replay_url", None),
            self._episode_start_ts,
        )

    def _add_timing_info(self) -> None:
        """Add timing information to infos."""
        assert self._sim is not None
        timer = self._sim._timer
        elapsed_times = timer.get_all_elapsed()
        thread_idle_time = elapsed_times.pop("thread_idle", 0)

        wall_time = timer.get_elapsed()
        adjusted_wall_time = wall_time - thread_idle_time

        lap_times = timer.lap_all(exclude_global=False)
        lap_thread_idle_time = lap_times.pop("thread_idle", 0)

        wall_time_for_lap = sum(lap_times.values()) + lap_thread_idle_time
        adjusted_lap_time = wall_time_for_lap - lap_thread_idle_time
        infos = self._sim._context["infos"]

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

    def on_close(self) -> None:
        self._stats_writer.close()

    def _update_label_completions(self, moving_avg_window: int = 500) -> None:
        """Update label completions."""
        assert self._sim is not None
        label = self._sim.config.label

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
