"""Statistics reporting and aggregation."""

import logging
from collections import defaultdict
from contextlib import nullcontext
from numbers import Number
from typing import Any, Dict, List, Optional
from uuid import UUID

import numpy as np
import torch
from pydantic import Field

from metta.app_backend.clients.stats_client import StatsClient
from metta.common.wandb.context import WandbRun
from metta.eval.eval_request_config import EvalRewardSummary
from metta.rl.stats import (
    accumulate_rollout_stats,
    compute_timing_stats,
    process_training_stats,
)
from metta.rl.training.component import TrainerComponent
from metta.rl.utils import should_run
from mettagrid.config import Config

logger = logging.getLogger(__name__)


def _to_scalar(value: Any) -> Optional[float]:
    """Convert supported numeric types to float, skipping non-scalars."""

    if isinstance(value, Number):
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
    processed_stats: Dict[str, Any],
    timing_info: Dict[str, Any],
    weight_stats: Dict[str, Any],
    grad_stats: Dict[str, float],
    system_stats: Dict[str, Any],
    memory_stats: Dict[str, Any],
    parameters: Dict[str, Any],
    hyperparameters: Dict[str, Any],
    evals: EvalRewardSummary,
    *,
    agent_step: int,
    epoch: int,
) -> Dict[str, float]:
    """Create a flattened stats dictionary ready for wandb logging."""

    overview: Dict[str, Any] = {
        "sps": timing_info.get("epoch_steps_per_second", 0.0),
        "steps_per_second": timing_info.get("steps_per_second", 0.0),
        "epoch_steps_per_second": timing_info.get("epoch_steps_per_second", 0.0),
        **processed_stats.get("overview", {}),
    }
    for category, score in evals.category_scores.items():
        overview[f"{category}_score"] = score
    if "reward" in overview:
        overview["reward_vs_total_time"] = overview["reward"]

    payload: Dict[str, float] = {
        "metric/agent_step": float(agent_step),
        "metric/epoch": float(epoch),
        "metric/total_time": float(timing_info.get("wall_time", 0.0)),
        "metric/train_time": float(timing_info.get("train_time", 0.0)),
    }

    def _update(items: Dict[str, Any], *, prefix: str = "") -> None:
        for key, value in items.items():
            scalar = _to_scalar(value)
            if scalar is None:
                continue
            metric_key = f"{prefix}{key}" if prefix else key
            payload[metric_key] = scalar

    _update(overview, prefix="overview/")
    _update(processed_stats.get("losses_stats", {}), prefix="losses/")
    _update(processed_stats.get("experience_stats", {}), prefix="experience/")
    _update(processed_stats.get("environment_stats", {}))
    _update(parameters, prefix="parameters/")
    _update(hyperparameters, prefix="hyperparameters/")

    eval_metrics = evals.to_wandb_metrics_format()
    for key, value in eval_metrics.items():
        scalar = _to_scalar(value)
        if scalar is None:
            continue
        payload[f"eval_{key}"] = scalar

    _update(system_stats)
    _update({f"trainer_memory/{k}": v for k, v in memory_stats.items()})
    _update(weight_stats)
    _update(grad_stats)
    _update(timing_info.get("timing_stats", {}))

    return payload


class StatsReporterConfig(Config):
    """Configuration for stats reporting."""

    report_to_wandb: bool = True
    report_to_stats_client: bool = True
    report_to_console: bool = True
    grad_mean_variance_interval: int = 50
    interval: int = 1
    """How often to report stats (in epochs)"""
    analyze_weights_interval: int = 0
    """How often to compute weight metrics (0 disables)."""


class StatsReporterState(Config):
    """State for statistics tracking."""

    rollout_stats: Dict = Field(default_factory=lambda: defaultdict(list))
    grad_stats: Dict = Field(default_factory=dict)
    eval_scores: EvalRewardSummary = Field(default_factory=EvalRewardSummary)
    stats_run_id: Optional[UUID] = None


class NoOpStatsReporter(TrainerComponent):
    """No-op stats reporter for when stats are disabled."""

    def __init__(self):
        """Initialize no-op stats reporter."""
        # Create a minimal config for the no-op reporter
        config = StatsReporterConfig(report_to_wandb=False, report_to_stats_client=False, interval=999999)
        super().__init__(epoch_interval=config.interval)
        self.wandb_run = None
        self.stats_run_id = None

    def on_step(self, infos: List[Dict[str, Any]]) -> None:
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
        stats_client: Optional[StatsClient] = None,
        wandb_run: Optional[WandbRun] = None,
    ) -> "StatsReporter":
        """Create a StatsReporter from optional config, returning no-op if None.

        Args:
            config: Optional stats configuration
            stats_client: Optional stats client
            wandb_run: Optional wandb run

        Returns:
            StatsReporter instance (no-op if config is None)
        """
        if config is None:
            return NoOpStatsReporter()
        return cls(config=config, stats_client=stats_client, wandb_run=wandb_run)

    def __init__(
        self,
        config: StatsReporterConfig,
        stats_client: Optional[StatsClient] = None,
        wandb_run: Optional[WandbRun] = None,
    ):
        """Initialize stats reporter.

        Args:
            config: Statistics configuration
            stats_client: Optional stats client for reporting
            wandb_run: Optional wandb run for reporting
        """
        super().__init__(epoch_interval=config.interval)
        self._config = config
        self._stats_client = stats_client
        self._wandb_run = wandb_run
        self._state = StatsReporterState()
        self._latest_payload: Dict[str, float] | None = None

        # Initialize stats run if client is available
        if self._stats_client and self._config.report_to_stats_client:
            self._initialize_stats_run()

    def register(self, context) -> None:  # type: ignore[override]
        super().register(context)
        context.stats_reporter = self

    def _initialize_stats_run(self) -> None:
        """Initialize stats run with the stats client."""
        if not self._stats_client:
            return

        # Extract wandb attributes with defaults
        name = url = "unknown"
        description: Optional[str] = None
        tags: Optional[List[str]] = None

        if self._wandb_run:
            name = self._wandb_run.name or name
            url = self._wandb_run.url
            if self._wandb_run.tags:
                tags = list(self._wandb_run.tags)
            description = self._wandb_run.notes

        try:
            result = self._stats_client.create_training_run(name=name, url=url, description=description, tags=tags)
            self._state.stats_run_id = result.id
            logger.info(f"Created stats run with ID: {self._state.stats_run_id}")
        except Exception as e:
            logger.warning(f"Failed to create training run: {e}", exc_info=True)

    @property
    def state(self) -> StatsReporterState:
        """Get the state for external access."""
        return self._state

    def process_rollout(self, raw_infos: List[Dict[str, Any]]) -> None:
        """Process rollout information.

        Args:
            raw_infos: Raw info dictionaries from rollout
        """
        if not raw_infos:
            return
        accumulate_rollout_stats(raw_infos, self._state.rollout_stats)

    def report_epoch(
        self,
        epoch: int,
        agent_step: int,
        losses_stats: Dict[str, float],
        experience: Any,
        policy: Any,
        timer: Any,
        trainer_cfg: Any,
        optimizer: torch.optim.Optimizer | None,
    ) -> None:
        """Report statistics for an epoch.

        Args:
            epoch: Current epoch
            agent_step: Current agent step
            losses_stats: Loss statistics
            experience: Experience buffer
            policy: Current policy
            timer: Timer for profiling
            trainer_cfg: Trainer configuration
            optimizer: Optimizer
        """
        timing_context = timer("_process_stats") if callable(timer) else nullcontext()

        with timing_context:
            payload = self._build_wandb_payload(
                losses_stats=losses_stats,
                experience=experience,
                trainer_cfg=trainer_cfg,
                policy=policy,
                agent_step=agent_step,
                epoch=epoch,
                timer=timer,
                optimizer=optimizer,
            )

            if self._wandb_run and self._config.report_to_wandb and payload:
                self._wandb_run.log(payload, step=agent_step)

            self._latest_payload = payload.copy() if payload else None

            if payload and self._stats_client and self._config.report_to_stats_client:
                run_id = self._state.stats_run_id
                if run_id is not None:
                    attributes: Dict[str, Any] = {"metrics": payload, "agent_step": agent_step}
                    self.create_epoch(run_id, epoch, epoch, attributes=attributes)

            # Clear stats after processing
            self.clear_rollout_stats()
            self.clear_grad_stats()

    def update_eval_scores(self, scores: EvalRewardSummary) -> None:
        """Update evaluation scores.

        Args:
            scores: New evaluation scores
        """
        self._state.eval_scores = scores
        if self._context is not None:
            self.context.latest_eval_scores = scores

    def clear_rollout_stats(self) -> None:
        """Clear rollout statistics."""
        self._state.rollout_stats.clear()
        self._state.rollout_stats = defaultdict(list)

    def clear_grad_stats(self) -> None:
        """Clear gradient statistics."""
        self._state.grad_stats.clear()

    def update_grad_stats(self, grad_stats: Dict[str, float]) -> None:
        """Update gradient statistics.

        Args:
            grad_stats: New gradient statistics
        """
        self._state.grad_stats = grad_stats

    def create_epoch(
        self,
        run_id: UUID,
        start_epoch: int,
        end_epoch: int,
        attributes: Dict[str, Any] | None = None,
    ) -> Optional[UUID]:
        """Create a new epoch in the stats client.

        Args:
            run_id: Training run ID
            start_epoch: Starting epoch
            end_epoch: Ending epoch

        Returns:
            Epoch ID if created successfully
        """
        if not self._stats_client or not self._config.report_to_stats_client:
            return None

        try:
            result = self._stats_client.create_epoch(
                run_id=run_id,
                start_training_epoch=start_epoch,
                end_training_epoch=end_epoch,
                attributes=attributes or {},
            )
            return result.id
        except Exception as e:
            logger.warning(f"Failed to create epoch: {e}", exc_info=True)
            return None

    def finalize(self, status: str = "completed") -> None:
        """Finalize stats reporting.

        Args:
            status: Final status of the training run
        """
        if self._stats_client and self._state.stats_run_id and self._config.report_to_stats_client:
            try:
                self._stats_client.update_training_run_status(self._state.stats_run_id, status)
                logger.info(f"Training run status updated to '{status}'")
            except Exception as e:
                logger.warning(f"Failed to update training run status: {e}", exc_info=True)
        self._latest_payload = None

    def on_step(self, infos: Dict[str, Any] | List[Dict[str, Any]]) -> None:
        """Accumulate step infos.

        Args:
            infos: Step information from environment
        """
        self.accumulate_infos(infos)

    def get_latest_payload(self) -> Optional[Dict[str, float]]:
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
            losses_stats=ctx.latest_losses_stats,
            experience=ctx.experience,
            policy=ctx.policy,
            timer=ctx.stopwatch,
            trainer_cfg=ctx.config,
            optimizer=ctx.optimizer,
        )

    def on_training_complete(self) -> None:
        """Handle training completion.

        Args:
        """
        self.finalize(status="completed")

    def on_failure(self) -> None:
        """Handle training failure.

        Args:
            trainer: The trainer instance
        """
        self.finalize(status="failed")

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def accumulate_infos(self, info: Dict[str, Any] | List[Dict[str, Any]] | None) -> None:
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
        losses_stats: Dict[str, float],
        experience: Any,
        trainer_cfg: Any,
        policy: Any,
        agent_step: int,
        epoch: int,
        timer: Any,
        optimizer: torch.optim.Optimizer | None,
    ) -> Dict[str, float]:
        """Convert collected stats into a flat wandb payload."""

        if experience is None:
            return {}

        processed = process_training_stats(
            raw_stats=self._state.rollout_stats,
            losses_stats=losses_stats,
            experience=experience,
            trainer_config=trainer_cfg,
        )

        timing_info = compute_timing_stats(timer=timer, agent_step=agent_step)
        self._normalize_steps_per_second(timing_info, agent_step)

        weight_stats = self._collect_weight_stats(policy=policy, epoch=epoch)
        system_stats = self._collect_system_stats()
        memory_stats = self._collect_memory_stats()
        parameters = self._collect_parameters(
            experience=experience,
            optimizer=optimizer,
            timing_info=timing_info,
        )
        hyperparameters = self._collect_hyperparameters(trainer_cfg=trainer_cfg, parameters=parameters)

        return build_wandb_payload(
            processed_stats=processed,
            timing_info=timing_info,
            weight_stats=weight_stats,
            grad_stats=self._state.grad_stats,
            system_stats=system_stats,
            memory_stats=memory_stats,
            parameters=parameters,
            hyperparameters=hyperparameters,
            evals=self._state.eval_scores,
            agent_step=agent_step,
            epoch=epoch,
        )

    def _normalize_steps_per_second(self, timing_info: Dict[str, Any], agent_step: int) -> None:
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

    def _collect_weight_stats(self, *, policy: Any, epoch: int) -> Dict[str, float]:
        interval = self._config.analyze_weights_interval
        if not interval:
            policy_config = getattr(policy, "config", None)
            interval = getattr(policy_config, "analyze_weights_interval", 0) if policy_config else 0

        if not interval or not should_run(epoch, interval):
            return {}

        if not hasattr(policy, "compute_weight_metrics"):
            return {}

        weight_stats: Dict[str, float] = {}
        try:
            for metrics in policy.compute_weight_metrics():
                name = metrics.get("name", "unknown")
                for key, value in metrics.items():
                    if key == "name":
                        continue
                    scalar = _to_scalar(value)
                    if scalar is None:
                        continue
                    weight_stats[f"weights/{key}/{name}"] = scalar
        except Exception as exc:  # pragma: no cover - safeguard against model-specific failures
            logger.warning("Failed to compute weight metrics: %s", exc, exc_info=True)
        return weight_stats

    def _collect_system_stats(self) -> Dict[str, Any]:
        system_monitor = getattr(self.context, "system_monitor", None)
        if system_monitor is None:
            return {}
        try:
            return system_monitor.stats()
        except Exception as exc:  # pragma: no cover
            logger.debug("System monitor stats failed: %s", exc, exc_info=True)
            return {}

    def _collect_memory_stats(self) -> Dict[str, Any]:
        memory_monitor = getattr(self.context, "memory_monitor", None)
        if memory_monitor is None:
            return {}
        try:
            return memory_monitor.stats()
        except Exception as exc:  # pragma: no cover
            logger.debug("Memory monitor stats failed: %s", exc, exc_info=True)
            return {}

    def _collect_parameters(
        self,
        *,
        experience: Any,
        optimizer: torch.optim.Optimizer,
        timing_info: Dict[str, Any],
    ) -> Dict[str, Any]:
        learning_rate = getattr(self.context.config.optimizer, "learning_rate", 0)
        if optimizer and optimizer.param_groups:
            learning_rate = optimizer.param_groups[0].get("lr", learning_rate)

        parameters: Dict[str, Any] = {
            "learning_rate": learning_rate,
            "epoch_steps": timing_info.get("epoch_steps", 0),
            "num_minibatches": getattr(experience, "num_minibatches", 0),
            "latest_saved_policy_epoch": getattr(self.context.state, "latest_saved_policy_epoch", 0),
        }
        return parameters

    def _collect_hyperparameters(
        self,
        *,
        trainer_cfg: Any,
        parameters: Dict[str, Any],
    ) -> Dict[str, Any]:
        hyperparameters: Dict[str, Any] = {}
        if "learning_rate" in parameters:
            hyperparameters["learning_rate"] = parameters["learning_rate"]

        losses = getattr(trainer_cfg, "losses", None)
        loss_configs = getattr(losses, "loss_configs", {}) if losses else {}
        ppo_cfg = loss_configs.get("ppo") if isinstance(loss_configs, dict) else None
        if ppo_cfg is not None:
            for attr in (
                "clip_coef",
                "vf_clip_coef",
                "ent_coef",
                "l2_reg_loss_coef",
                "l2_init_loss_coef",
            ):
                value = getattr(ppo_cfg, attr, None)
                if value is None:
                    continue
                hyperparameters[f"ppo_{attr}"] = value
        return hyperparameters
