"""Component that logs training progress each epoch."""

import logging
from typing import Dict, Optional

from metta.rl.training.component import TrainerComponent
from metta.rl.utils import log_training_progress

logger = logging.getLogger(__name__)


class ProgressLogger(TrainerComponent):
    """Master-only component that emits concise progress logs each epoch."""

    _master_only = True

    def __init__(self) -> None:
        super().__init__(epoch_interval=1)
        self._previous_agent_step: int = 0

    def register(self, context) -> None:  # type: ignore[override]
        super().register(context)
        self._previous_agent_step = context.agent_step

    def on_epoch_end(self, epoch: int) -> None:  # type: ignore[override]
        ctx = self.context
        metrics = self._latest_metrics()

        log_training_progress(
            epoch=ctx.epoch,
            agent_step=ctx.agent_step,
            prev_agent_step=self._previous_agent_step,
            total_timesteps=ctx.config.total_timesteps,
            train_time=ctx.stopwatch.get_last_elapsed("_train"),
            rollout_time=ctx.stopwatch.get_last_elapsed("_rollout"),
            stats_time=ctx.stopwatch.get_last_elapsed("_process_stats"),
            run_name=ctx.run_name,
            metrics=metrics,
        )

        self._previous_agent_step = ctx.agent_step

    def _latest_metrics(self) -> Optional[Dict[str, float]]:
        stats_reporter = getattr(self.context, "stats_reporter", None)
        if stats_reporter is None:
            return None
        getter = getattr(stats_reporter, "get_latest_payload", None)
        if getter is None:
            return None
        return getter()
