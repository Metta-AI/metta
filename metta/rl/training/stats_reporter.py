"""Statistics reporting and aggregation."""

from __future__ import annotations

import logging
from collections import defaultdict
from numbers import Number
from typing import Any, Dict, List, Optional
from uuid import UUID

import numpy as np
import torch
from pydantic import Field

from metta.app_backend.clients.stats_client import StatsClient
from metta.common.wandb.wandb_context import WandbRun
from metta.core.monitoring import cleanup_monitoring, setup_monitoring
from metta.eval.eval_request_config import EvalRewardSummary
from metta.mettagrid.config import Config
from metta.rl.stats import (
    accumulate_rollout_stats,
    compute_timing_stats,
    process_training_stats,
)
from metta.rl.training.component import TrainerComponent
from metta.rl.training.context import TrainerContext

logger = logging.getLogger(__name__)


class StatsConfig(Config):
    """Configuration for stats reporting."""

    report_to_wandb: bool = True
    report_to_stats_client: bool = True
    report_to_console: bool = True
    grad_mean_variance_interval: int = 50
    interval: int = 1
    """How often to report stats (in epochs)"""


class StatsState(Config):
    """State for statistics tracking."""

    rollout_stats: Dict = Field(default_factory=lambda: defaultdict(list))
    grad_stats: Dict = Field(default_factory=dict)
    eval_scores: EvalRewardSummary = Field(default_factory=EvalRewardSummary)
    stats_run_id: Optional[UUID] = None
    stats_epoch_id: Optional[UUID] = None
    stats_epoch_start: int = 0
    latest_saved_epoch: int = 0


class NoOpStatsReporter(TrainerComponent):
    """No-op stats reporter for when stats are disabled."""

    def __init__(self):
        """Initialize no-op stats reporter."""
        # Create a minimal config for the no-op reporter
        config = StatsConfig(report_to_wandb=False, report_to_stats_client=False, interval=999999)
        super().__init__(epoch_interval=config.interval)
        self.wandb_run = None
        self.stats_run_id = None
        self.stats_epoch_id = None
        self.infos_buffer = []

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
        config: Optional[StatsConfig],
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
        config: StatsConfig,
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
        self._state = StatsState()
        self._memory_monitor = None
        self._system_monitor = None

        # Initialize stats run if client is available
        if self._stats_client and self._config.report_to_stats_client:
            self._initialize_stats_run()

    def register(self, context: TrainerContext) -> None:  # type: ignore[override]
        super().register(context)
        reporting_enabled = (
            self._config.report_to_wandb or self._config.report_to_stats_client or self._config.report_to_console
        )
        if reporting_enabled:
            experience = context.core_loop.experience
            memory_monitor, system_monitor = setup_monitoring(
                policy=context.policy,
                experience=experience,
                timer=context.stopwatch,
            )
            self._memory_monitor = memory_monitor
            self._system_monitor = system_monitor

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
    def state(self) -> StatsState:
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
        optimizer: torch.optim.Optimizer,
        memory_monitor: Optional[Any] = None,
        system_monitor: Optional[Any] = None,
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
            memory_monitor: Optional memory monitor
            system_monitor: Optional system monitor
        """
        if self._wandb_run and self._config.report_to_wandb:
            payload = self._build_wandb_payload(
                losses_stats=losses_stats,
                experience=experience,
                trainer_cfg=trainer_cfg,
                agent_step=agent_step,
                epoch=epoch,
                timer=timer,
            )
            if payload:
                self._wandb_run.log(payload, step=agent_step)

        # Clear stats after processing
        self.clear_rollout_stats()
        self.clear_grad_stats()

    def update_eval_scores(self, scores: EvalRewardSummary) -> None:
        """Update evaluation scores.

        Args:
            scores: New evaluation scores
        """
        self._state.eval_scores = scores

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

    def update_latest_saved_epoch(self, epoch: int) -> None:
        """Update the latest saved epoch.

        Args:
            epoch: Latest saved epoch
        """
        self._state.latest_saved_epoch = epoch

    def create_epoch(self, run_id: UUID, start_epoch: int, end_epoch: int) -> Optional[UUID]:
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
                run_id=run_id, start_training_epoch=start_epoch, end_training_epoch=end_epoch
            )
            self._state.stats_epoch_id = result.id
            return result.id
        except Exception as e:
            logger.warning(f"Failed to create epoch: {e}", exc_info=True)
            return None

    def update_epoch_tracking(self, epoch: int) -> None:
        """Update epoch tracking for next epoch.

        Args:
            epoch: New epoch number
        """
        self._state.stats_epoch_start = epoch

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

    def on_step(self, infos: Dict[str, Any]) -> None:
        """Accumulate step infos.

        Args:
            infos: Step information from environment
        """
        self.accumulate_infos(infos)

    def on_epoch_end(self, epoch: int) -> None:
        """Report stats at epoch end.

        Args:
        """
        context = self.context
        experience = context.core_loop.experience

        self.report_epoch(
            epoch=context.epoch,
            agent_step=context.agent_step,
            losses_stats=getattr(context, "latest_losses_stats", {}),
            experience=experience,
            policy=context.policy,
            timer=context.stopwatch,
            trainer_cfg=context.cfg,
            optimizer=context.optimizer,
            memory_monitor=self._memory_monitor,
            system_monitor=self._system_monitor,
        )

    def on_training_complete(self) -> None:
        """Handle training completion.

        Args:
        """
        self.finalize(status="completed")
        cleanup_monitoring(self._memory_monitor, self._system_monitor)

    def on_failure(self) -> None:
        """Handle training failure.

        Args:
            trainer: The trainer instance
        """
        self.finalize(status="failed")
        cleanup_monitoring(self._memory_monitor, self._system_monitor)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def accumulate_infos(self, info: Dict[str, Any] | None) -> None:
        """Accumulate rollout info dictionaries for later aggregation."""
        if not info:
            return
        self.process_rollout([info])

    def _build_wandb_payload(
        self,
        *,
        losses_stats: Dict[str, float],
        experience: Any,
        trainer_cfg: Any,
        agent_step: int,
        epoch: int,
        timer: Any,
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

        payload: Dict[str, float] = {
            "metric/agent_step": float(agent_step),
            "metric/epoch": float(epoch),
            "metric/total_time": float(timing_info["wall_time"]),
            "metric/train_time": float(timing_info["train_time"]),
            "overview/steps_per_second": float(timing_info["steps_per_second"]),
            "overview/epoch_steps_per_second": float(timing_info["epoch_steps_per_second"]),
        }

        payload.update(self._prefix_and_filter(processed["overview"], prefix="overview/"))
        payload.update(self._prefix_and_filter(processed["environment_stats"], prefix=""))

        return payload

    @staticmethod
    def _prefix_and_filter(items: Dict[str, Any], *, prefix: str) -> Dict[str, float]:
        """Return a dict of numeric wandb-friendly values with an optional prefix."""

        flattened: Dict[str, float] = {}
        for key, value in items.items():
            scalar = StatsReporter._to_scalar(value)
            if scalar is None:
                continue
            metric_key = f"{prefix}{key}" if prefix else key
            flattened[metric_key] = scalar
        return flattened

    @staticmethod
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
