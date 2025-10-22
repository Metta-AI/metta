import datetime
from typing import Any, Dict, cast

from mettagrid.simulator import Simulator, SimulatorEventHandler
from mettagrid.util.dict_utils import unroll_nested_dict
from mettagrid.util.stats_writer import StatsWriter


class StatsTracker(SimulatorEventHandler):
    """Tracker for recording statistics."""

    def __init__(self, simulator: Simulator, stats_writer: StatsWriter):
        self._stats_writer = stats_writer
        self._episode_start_ts = datetime.datetime.now()
        self._episode_end_ts = None
        self._label_completions = {"completed_tasks": [], "completion_rates": {}}
        self._per_label_rewards = {}
        super().__init__(simulator)

    def on_episode_start(self) -> None:
        self._episode_start_ts = datetime.datetime.now()

    def on_episode_end(self, infos: Dict[str, Any]) -> None:
        # Get episode rewards and stats
        episode_rewards = self._simulator.episode_rewards

        stats = self._simulator.episode_stats
        config = self._simulator.config
        num_agents = config.game.num_agents

        # Process agent stats
        infos["game"] = stats["game"]
        infos["agent"] = {}
        for agent_stats in stats["agent"]:
            for n, v in agent_stats.items():
                infos["agent"][n] = infos["agent"].get(n, 0) + v
        for n, v in infos["agent"].items():
            infos["agent"][n] = v / num_agents

        # If reward estimates are set, plot them compared to the mean reward
        if self._simulator.config.game.reward_estimates:
            infos["reward_estimates"] = {}
            infos["reward_estimates"]["best_case_optimal_diff"] = (
                config.game.reward_estimates["best_case_optimal_reward"] - episode_rewards.mean()
            )
            infos["reward_estimates"]["worst_case_optimal_diff"] = (
                config.game.reward_estimates["worst_case_optimal_reward"] - episode_rewards.mean()
            )

        self._update_label_completions(moving_avg_window)

        # only plot label completions once we have a full moving average window, to prevent initial bias
        if len(self._label_completions["completed_tasks"]) >= 50:
            infos["label_completions"] = self._label_completions["completion_rates"]
        self._per_label_rewards[config.label] = episode_rewards.mean()
        infos["per_label_rewards"] = self._per_label_rewards

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

        # Add timing information
        self._add_timing_info(infos)

        # Flatten environment config
        env_cfg_flattened: Dict[str, str] = {}
        env_cfg = self._simulator.config.model_dump()
        for k, v in unroll_nested_dict(cast(Dict[str, Any], env_cfg)):
            env_cfg_flattened[f"config.{str(k).replace('/', '.')}"] = str(v)

        # Prepare agent metrics
        agent_metrics = {}
        for agent_idx, agent_stats in enumerate(stats["agent"]):
            agent_metrics[agent_idx] = {}
            agent_metrics[agent_idx]["reward"] = float(self._simulator.episode_rewards[agent_idx])
            for k, v in agent_stats.items():
                agent_metrics[agent_idx][k] = float(v)

        # Get agent groups
        grid_objects = self._simulator.grid_objects(ignore_types=["wall"])
        agent_groups: Dict[int, int] = {
            v["agent_id"]: v["agent:group"] for v in grid_objects.values() if v["type"] == 0
        }

        # Record episode
        self._stats_writer.record_episode(
            env_cfg_flattened,
            agent_metrics,
            agent_groups,
            self._simulator.current_step,
            infos.get("replay_url"),
            self._episode_start_ts,
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

    def on_close(self) -> None:
        self._stats_writer.close()

    def _update_label_completions(self, moving_avg_window: int = 500) -> None:
        """Update label completions."""
        label = self._simulator.config.label

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
