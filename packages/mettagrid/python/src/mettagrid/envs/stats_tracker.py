import datetime
import time
from typing import Any, Dict, cast

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
        self._per_label_rewards = {}
        self._per_label_chest_deposits = {}  # Track chest deposits per label

    def on_episode_start(self) -> None:
        super().on_episode_start()
        self._sim._context["infos"] = {}
        self._episode_start_ts = datetime.datetime.now()

    def on_episode_end(self) -> None:
        super().on_episode_end()

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

        self._per_label_rewards[config.label] = mean_reward
        infos["per_label_rewards"] = self._per_label_rewards

        # Track chest deposits per label (for per-mission analysis)
        chest_deposited = stats["game"].get("chest.heart.deposited", 0.0)
        self._per_label_chest_deposits[config.label] = chest_deposited
        infos["per_label_chest_deposits"] = self._per_label_chest_deposits

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
        agent_metrics = {}
        for agent_idx, agent_stats in enumerate(stats["agent"]):
            agent_metrics[agent_idx] = {}
            agent_metrics[agent_idx]["reward"] = float(self._sim.episode_rewards[agent_idx])
            for k, v in agent_stats.items():
                agent_metrics[agent_idx][k] = float(v)

        # Get agent groups
        grid_objects = self._sim.grid_objects(ignore_types=["wall"])
        agent_groups: Dict[int, int] = {
            v["agent_id"]: v["agent:group"] for v in grid_objects.values() if "agent_id" in v
        }

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
