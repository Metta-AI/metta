"""Automatic tuner for PPO-style update epochs."""

from __future__ import annotations

import logging
from collections import deque
from typing import Deque, Dict, Optional, Tuple

from metta.rl.trainer_config import UpdateEpochAutoTunerConfig
from metta.rl.training import TrainerComponent

logger = logging.getLogger(__name__)


class UpdateEpochAutoTuner(TrainerComponent):
    """Dynamically adjusts ``update_epochs`` using simple PPO health metrics."""

    def __init__(self, config: UpdateEpochAutoTunerConfig):
        super().__init__(epoch_interval=1)
        self._cfg = config
        self._history: Dict[int, Deque[float]] = {}
        self._current_update_epochs: int = config.min_update_epochs
        self._cooldown_counter: int = 0
        self._epochs_at_current: int = 0
        self._is_master: bool = False

    def register(self, context) -> None:  # type: ignore[override]
        super().register(context)
        if not self._cfg.enabled:
            logger.debug("UpdateEpochAutoTuner registered but disabled; no action will be taken.")
            return

        current = int(getattr(context.config, "update_epochs", self._cfg.min_update_epochs))
        clamped = self._clamp(current)
        if clamped != current:
            logger.info(
                "Clamping initial update_epochs from %s to %s to satisfy autotune bounds",
                current,
                clamped,
            )
        context.config.update_epochs = clamped
        self._current_update_epochs = clamped
        self._history.setdefault(clamped, deque(maxlen=self._cfg.metrics_window))

        self._is_master = bool(context.distributed.is_master())
        self._epochs_at_current = 0
        self._cooldown_counter = 0

        logger.info(
            "UpdateEpochAutoTuner enabled (min=%s max=%s step=%s) starting at update_epochs=%s",
            self._cfg.min_update_epochs,
            self._cfg.max_update_epochs,
            self._cfg.step_size,
            self._current_update_epochs,
        )

    def on_epoch_end(self, epoch: int) -> None:  # type: ignore[override]
        if not self._cfg.enabled:
            return

        ctx = self.context
        metrics = self._collect_metrics()
        if metrics is None:
            return
        approx_kl, clipfrac = metrics
        self._epochs_at_current += 1

        target_value = self._current_update_epochs
        if self._is_master:
            target_value = self._evaluate_master(epoch=epoch, approx_kl=approx_kl, clipfrac=clipfrac)

        decided_value = ctx.distributed.broadcast_from_master(target_value if self._is_master else None)
        decided_value = self._clamp(int(decided_value))

        if decided_value != self._current_update_epochs:
            self._handle_external_switch(decided_value, epoch, approx_kl, clipfrac)

        ctx.config.update_epochs = decided_value
        self._history.setdefault(decided_value, deque(maxlen=self._cfg.metrics_window))

    def _evaluate_master(self, *, epoch: int, approx_kl: float, clipfrac: float) -> int:
        if approx_kl > 0.0:
            self._record_metric(self._current_update_epochs, approx_kl)

        current_avg = self._average(self._current_update_epochs)

        if epoch < self._cfg.warmup_epochs or self._epochs_at_current < self._cfg.evaluation_epochs:
            return self._current_update_epochs

        if self._cooldown_counter > 0:
            self._cooldown_counter -= 1
            return self._current_update_epochs

        candidate = self._suggest_adjustment(avg_kl=current_avg, clipfrac=clipfrac)
        if candidate != self._current_update_epochs:
            self._switch_to(candidate, epoch, avg_kl=current_avg, clipfrac=clipfrac)

        return self._current_update_epochs

    def _collect_metrics(self) -> Optional[Tuple[float, float]]:
        stats = getattr(self.context, "latest_losses_stats", None)
        if not stats:
            return None
        approx_kl = float(stats.get("approx_kl", 0.0))
        clipfrac = float(stats.get("clipfrac", 0.0))
        if approx_kl <= 0.0 and clipfrac <= 0.0:
            return None
        return approx_kl, clipfrac

    def _record_metric(self, update_epochs: int, approx_kl: float) -> None:
        history = self._history.setdefault(update_epochs, deque(maxlen=self._cfg.metrics_window))
        history.append(approx_kl)

    def _average(self, update_epochs: int) -> float:
        history = self._history.get(update_epochs, deque([0.0], maxlen=1))
        if not history:
            return 0.0
        return sum(history) / len(history)

    def _suggest_adjustment(self, *, avg_kl: float, clipfrac: float) -> int:
        current = self._current_update_epochs
        target = self._cfg.target_kl
        tolerance = self._cfg.kl_tolerance
        max_clipfrac = self._cfg.max_clipfrac

        if clipfrac > max_clipfrac or avg_kl > target * (1.0 + tolerance):
            return self._clamp(current - self._cfg.step_size)

        if avg_kl > 0.0 and avg_kl < target * (1.0 - tolerance):
            return self._clamp(current + self._cfg.step_size)

        return current

    def _switch_to(self, new_value: int, epoch: int, *, avg_kl: float, clipfrac: float) -> None:
        previous = self._current_update_epochs
        self._current_update_epochs = new_value
        self._epochs_at_current = 0
        self._cooldown_counter = self._cfg.cooldown_epochs
        self._history.setdefault(new_value, deque(maxlen=self._cfg.metrics_window))

        logger.info(
            "Auto-tuning update_epochs from %s to %s at epoch %s (avg_kl=%.4f clipfrac=%.3f)",
            previous,
            new_value,
            epoch,
            avg_kl,
            clipfrac,
        )

    def _handle_external_switch(self, new_value: int, epoch: int, avg_kl: float, clipfrac: float) -> None:
        previous = self._current_update_epochs
        self._current_update_epochs = new_value
        self._epochs_at_current = 0
        logger.debug(
            "Replica adopting update_epochs change from %s to %s at epoch %s (avg_kl=%.4f clipfrac=%.3f)",
            previous,
            new_value,
            epoch,
            avg_kl,
            clipfrac,
        )

    def _clamp(self, value: int) -> int:
        return max(self._cfg.min_update_epochs, min(value, self._cfg.max_update_epochs))
