"""Statistics reporting and aggregation."""

import logging
from collections import defaultdict, deque
from contextlib import nullcontext
from typing import Any, ContextManager, Optional, Protocol

import numpy as np
import torch
from pydantic import Field

from metta.common.wandb.context import WandbRun
from metta.rl.stats import accumulate_rollout_stats, compute_timing_stats, process_training_stats
from metta.rl.training.component import TrainerComponent
from mettagrid.base_config import Config

logger = logging.getLogger(__name__)


class Timer(Protocol):
    def __call__(self, name: str) -> ContextManager[Any]: ...


def _to_scalar(value: Any) -> Optional[float]:
    """Convert supported numeric types to float, skipping non-scalars."""

    if isinstance(value, (int, float, bool, np.number)):
        return float(value)
    if isinstance(value, np.ndarray):
        if value.size == 1:
            return float(value.item())
        return None
    if torch.is_tensor(value):
        if value.numel() == 1:
            return float(value.detach().cpu().item())
        return None
    return None


def build_wandb_payload(
    processed_stats: dict[str, Any],
    timing_info: dict[str, Any],
    grad_stats: dict[str, float],
    system_stats: dict[str, Any],
    memory_stats: dict[str, Any],
    hyperparameters: dict[str, Any],
    *,
    agent_step: int,
    epoch: int,
) -> dict[str, float]:
    """Create a flattened stats dictionary ready for wandb logging."""

    overview: dict[str, Any] = {
        "sps": timing_info.get("epoch_steps_per_second", 0.0),
        "steps_per_second": timing_info.get("steps_per_second", 0.0),
        "epoch_steps_per_second": timing_info.get("epoch_steps_per_second", 0.0),
        **processed_stats.get("overview", {}),
    }
    if "reward" in overview:
        overview["reward_vs_total_time"] = overview["reward"]

    payload: dict[str, float] = {
        "metric/agent_step": float(agent_step),
        "metric/epoch": float(epoch),
        "metric/total_time": float(timing_info.get("wall_time", 0.0)),
        "metric/train_time": float(timing_info.get("train_time", 0.0)),
    }

    def _update(items: dict[str, Any], *, prefix: str = "") -> None:
        for key, value in items.items():
            scalar = _to_scalar(value)
            if scalar is None:
                continue
            metric_key = f"{prefix}{key}" if prefix else key
            payload[metric_key] = scalar

    _update(overview, prefix="overview/")
    _update(processed_stats.get("graph_stats", {}))

    # Get experience stats and compute area under reward
    experience_stats = processed_stats.get("experience_stats", {})
    _update(experience_stats, prefix="experience/")

    _update(processed_stats.get("environment_stats", {}))
    _update(hyperparameters, prefix="hyperparameters/")
    _update(system_stats)
    _update({f"trainer_memory/{k}": v for k, v in memory_stats.items()})
    _update(grad_stats)
    _update(timing_info.get("timing_stats", {}))

    return payload


class StatsReporterConfig(Config):
    """Configuration for stats reporting."""

    report_to_wandb: bool = True
    report_to_console: bool = True
    grad_mean_variance_interval: int = 50
    interval: int = 1
    """How often to report stats (in epochs)"""
    rolling_window: int = Field(default=5, ge=1, description="Number of epochs for metric rolling averages")
    default_zero_metrics: tuple[str, ...] = Field(
        default_factory=lambda: ("env_game/assembler.heart.created",),
        description="Environment metrics that should be logged as 0 when missing.",
    )


class StatsReporterState(Config):
    """State for statistics tracking."""

    rollout_stats: dict = Field(default_factory=lambda: defaultdict(list))
    grad_stats: dict = Field(default_factory=dict)
    area_under_reward: float = 0.0
    """Cumulative area under the reward curve"""
    rolling_stats: dict[str, deque[float]] = Field(default_factory=dict)


class NoOpStatsReporter(TrainerComponent):
    """No-op stats reporter for when stats are disabled."""

    def __init__(self):
        """Initialize no-op stats reporter."""
        # Create a minimal config for the no-op reporter
        config = StatsReporterConfig(report_to_wandb=False, interval=999999)
        super().__init__(epoch_interval=config.interval)
        self.wandb_run = None

    def on_step(self, infos: list[dict[str, Any]]) -> None:
        pass

    def on_epoch_end(self, epoch: int) -> None:
        pass

    def on_training_complete(self) -> None:
        pass

    def on_failure(self) -> None:
        pass


class StatsReporter(TrainerComponent):
    """Aggregates and reports statistics to multiple backends."""

    @classmethod
    def from_config(
        cls,
        config: Optional[StatsReporterConfig],
        wandb_run: Optional[WandbRun] = None,
    ) -> TrainerComponent:
        """Create a StatsReporter from optional config, returning no-op if None."""
        if config is None:
            return NoOpStatsReporter()
        return cls(config=config, wandb_run=wandb_run)

    def __init__(
        self,
        config: StatsReporterConfig,
        wandb_run: Optional[WandbRun] = None,
    ):
        super().__init__(epoch_interval=config.interval)
        self._config = config
        self._wandb_run = wandb_run
        self._state = StatsReporterState()
        self._latest_payload: dict[str, float] | None = None
        self._state.rolling_stats = {}

    @property
    def wandb_run(self) -> WandbRun | None:
        return self._wandb_run

    @wandb_run.setter
    def wandb_run(self, run: WandbRun | None) -> None:
        self._wandb_run = run

    def register(self, context) -> None:  # type: ignore[override]
        super().register(context)
        context.stats_reporter = self

    @property
    def state(self) -> StatsReporterState:
        """Get the state for external access."""
        return self._state

    def process_rollout(self, raw_infos: list[dict[str, Any]]) -> None:
        if not raw_infos:
            return
        accumulate_rollout_stats(raw_infos, self._state.rollout_stats)

    def report_epoch(
        self,
        epoch: int,
        agent_step: int,
        graph_stats: dict[str, float],
        experience: Any,
        policy: Any,
        timer: Timer | None,
        trainer_cfg: Any,
        optimizer: torch.optim.Optimizer,
    ) -> None:
        timing_context = timer("_process_stats") if callable(timer) else nullcontext()

        with timing_context:
            payload = self._build_wandb_payload(
                graph_stats=graph_stats,
                experience=experience,
                trainer_cfg=trainer_cfg,
                policy=policy,
                agent_step=agent_step,
                epoch=epoch,
                timer=timer,
                optimizer=optimizer,
            )

            # Update area under reward curve
            # Uses the current reward value and accumulates it over time
            if "experience/rewards" in payload:
                # Assuming each epoch represents a fixed time interval
                self._state.area_under_reward += payload["experience/rewards"]
                payload["experience/area_under_reward"] = self._state.area_under_reward

            if self._wandb_run and self._config.report_to_wandb and payload:
                self._wandb_run.log(payload, step=agent_step)

            self._latest_payload = payload.copy() if payload else None

            # Clear stats after processing
            self.clear_rollout_stats()
            self.clear_grad_stats()

    def clear_rollout_stats(self) -> None:
        """Clear rollout statistics."""
        self._state.rollout_stats.clear()
        self._state.rollout_stats = defaultdict(list)

    def clear_grad_stats(self) -> None:
        """Clear gradient statistics."""
        self._state.grad_stats.clear()

    def update_grad_stats(self, grad_stats: dict[str, float]) -> None:
        self._state.grad_stats = grad_stats

    def on_step(self, infos: dict[str, Any] | list[dict[str, Any]]) -> None:
        """Accumulate step infos.

        Args:
            infos: Step information from environment
        """
        self.accumulate_infos(infos)

    def get_latest_payload(self) -> Optional[dict[str, float]]:
        if self._latest_payload is None:
            return None
        return self._latest_payload.copy()

    def on_epoch_end(self, epoch: int) -> None:
        """Report stats at epoch end.

        Args:
        """
        ctx = self.context

        self.report_epoch(
            epoch=ctx.epoch,
            agent_step=ctx.agent_step,
            graph_stats=ctx.latest_graph_stats,
            experience=ctx.experience,
            policy=ctx.policy,
            timer=ctx.stopwatch,
            trainer_cfg=ctx.config,
            optimizer=ctx.optimizer,
        )

    def on_training_complete(self) -> None:
        pass

    def on_failure(self) -> None:
        pass

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def accumulate_infos(self, info: dict[str, Any] | list[dict[str, Any]] | None) -> None:
        """Accumulate rollout info dictionaries for later aggregation."""
        if not info:
            return
        if isinstance(info, list):
            filtered = [i for i in info if i]
            if not filtered:
                return
            self.process_rollout(filtered)
            return

        self.process_rollout([info])

    def _build_wandb_payload(
        self,
        *,
        graph_stats: dict[str, float],
        experience: Any,
        trainer_cfg: Any,
        policy: Any,
        agent_step: int,
        epoch: int,
        timer: Any,
        optimizer: torch.optim.Optimizer,
    ) -> dict[str, float]:
        """Convert collected stats into a flat wandb payload."""

        if experience is None:
            return {}

        processed = process_training_stats(
            raw_stats=self._state.rollout_stats,
            graph_stats=graph_stats,
            experience=experience,
            trainer_config=trainer_cfg,
        )

        overview = processed.setdefault("overview", {})
        avg_reward = getattr(getattr(self.context, "state", None), "avg_reward", None)
        if avg_reward is None:
            avg_reward_tensor = torch.tensor(0.0, dtype=torch.float32)
        else:
            avg_reward_tensor = torch.as_tensor(avg_reward)
            if avg_reward_tensor.numel() == 0:
                avg_reward_tensor = torch.tensor(0.0, dtype=torch.float32)
        avg_reward_tensor = avg_reward_tensor.detach().cpu()
        overview["avg_reward_estimate"] = float(avg_reward_tensor.mean().item())
        overview["avg_reward_estimate_min"] = float(avg_reward_tensor.min().item())
        overview["avg_reward_estimate_max"] = float(avg_reward_tensor.max().item())

        # Ensure certain env metrics always exist (e.g., env_game/assembler.heart.created) so rolling
        # averages and wandb logs see zeros instead of missing keys.
        env_stats = processed.setdefault("environment_stats", {})
        if isinstance(env_stats, dict):
            for key in self._config.default_zero_metrics:
                env_stats.setdefault(key, 0.0)

        self._augment_with_rolling_averages(processed)

        timing_info = compute_timing_stats(timer=timer, agent_step=agent_step)
        self._normalize_steps_per_second(timing_info, agent_step)

        system_stats = self._collect_system_stats()
        memory_stats = self._collect_memory_stats()
        hyperparameters = self._collect_hyperparameters(optimizer=optimizer)

        return build_wandb_payload(
            processed_stats=processed,
            timing_info=timing_info,
            grad_stats=self._state.grad_stats,
            system_stats=system_stats,
            memory_stats=memory_stats,
            hyperparameters=hyperparameters,
            agent_step=agent_step,
            epoch=epoch,
        )

    def _augment_with_rolling_averages(self, processed: dict[str, Any]) -> None:
        env_stats = processed.get("environment_stats")
        if not isinstance(env_stats, dict):
            return

        tracked_keys = set(self._config.default_zero_metrics)
        for key in list(self._state.rolling_stats.keys()):
            if key not in tracked_keys:
                del self._state.rolling_stats[key]
        window = self._config.rolling_window

        for key in tracked_keys:
            value = env_stats.get(key)
            scalar = _to_scalar(value)
            history = self._state.rolling_stats.get(key)
            if history is None:
                history = deque(maxlen=window)
                self._state.rolling_stats[key] = history
            if scalar is None:
                scalar = history[-1] if history else None
            if scalar is None:
                continue
            history.append(scalar)
            env_stats.setdefault(key, scalar)
            env_stats[f"{key}.avg"] = sum(history) / len(history)

    def _normalize_steps_per_second(self, timing_info: dict[str, Any], agent_step: int) -> None:
        """Adjust SPS to account for agent steps accumulated before a resume."""

        context = self._context
        if context is None:
            return

        baseline = getattr(context, "timing_baseline", None)
        if not isinstance(baseline, dict):
            return

        baseline_steps = baseline.get("agent_step", 0)
        baseline_wall = baseline.get("wall_time", 0.0)

        effective_elapsed = timing_info.get("wall_time", 0.0) - baseline_wall
        effective_steps = agent_step - baseline_steps

        if effective_elapsed <= 0 or effective_steps <= 0:
            return

        sps = effective_steps / effective_elapsed
        timing_info["steps_per_second"] = sps

        timing_stats = timing_info.get("timing_stats")
        if isinstance(timing_stats, dict):
            timing_stats["timing_cumulative/sps"] = sps

    def _collect_system_stats(self) -> dict[str, Any]:
        system_monitor = getattr(self.context, "system_monitor", None)
        if system_monitor is None:
            return {}
        try:
            return system_monitor.stats()
        except Exception as exc:  # pragma: no cover
            logger.debug("System monitor stats failed: %s", exc, exc_info=True)
            return {}

    def _collect_memory_stats(self) -> dict[str, Any]:
        memory_monitor = getattr(self.context, "memory_monitor", None)
        if memory_monitor is None:
            return {}
        try:
            return memory_monitor.stats()
        except Exception as exc:  # pragma: no cover
            logger.debug("Memory monitor stats failed: %s", exc, exc_info=True)
            return {}

    def _collect_hyperparameters(self, *, optimizer: torch.optim.Optimizer) -> dict[str, Any]:
        hyperparameters: dict[str, Any] = {}
        param_groups = optimizer.param_groups
        if not param_groups:
            return hyperparameters
        param_group = param_groups[0]
        learning_rate = param_group.get("lr")
        if learning_rate is not None:
            hyperparameters["learning_rate"] = learning_rate
        scheduled_lr = param_group.get("scheduled_lr")
        if scheduled_lr is not None:
            hyperparameters["schedulefree_scheduled_lr"] = scheduled_lr
        lr_max = param_group.get("lr_max")
        if lr_max is not None:
            hyperparameters["schedulefree_lr_max"] = lr_max
        return hyperparameters
