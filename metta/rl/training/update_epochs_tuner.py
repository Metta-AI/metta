"""Automatic tuner for PPO-style update epochs."""

from __future__ import annotations

import logging
from typing import Optional, Tuple

from metta.rl.trainer_config import UpdateEpochAutoTunerConfig
from metta.rl.training import TrainerComponent

logger = logging.getLogger(__name__)


class UpdateEpochAutoTuner(TrainerComponent):
    """Dynamically adjusts ``update_epochs`` using simple PPO health metrics."""

    def __init__(self, config: UpdateEpochAutoTunerConfig):
        super().__init__(epoch_interval=1)
        self._cfg = config
        self._current_update_epochs = config.min_update_epochs
        self._cooldown_counter = 0
        self._epochs_at_current = 0
        self._is_master = False

    def register(self, context) -> None:  # type: ignore[override]
        super().register(context)
        if not self._cfg.enabled:
            logger.debug("UpdateEpochAutoTuner registered but disabled; no action will be taken.")
            return

        clamped = self._clamp(int(getattr(context.config, "update_epochs", self._cfg.min_update_epochs)))
        if clamped != context.config.update_epochs:
            logger.info("Clamping initial update_epochs from %s to %s", context.config.update_epochs, clamped)
        context.config.update_epochs = self._current_update_epochs = clamped
        self._is_master = bool(context.distributed.is_master())
        self._epochs_at_current = self._cooldown_counter = 0

        logger.info(
            "UpdateEpochAutoTuner enabled (min=%s max=%s step=%s start=%s)",
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

        decided_value = self._clamp(int(ctx.distributed.broadcast_from_master(target_value if self._is_master else None)))
        if decided_value != self._current_update_epochs:
            self._set_value(
                decided_value,
                epoch=epoch,
                approx_kl=approx_kl,
                clipfrac=clipfrac,
                set_cooldown=False,
                log_fn=logger.debug,
                prefix="Replica adopting",
            )

        ctx.config.update_epochs = decided_value

    def _evaluate_master(self, *, epoch: int, approx_kl: float, clipfrac: float) -> int:
        if epoch < self._cfg.warmup_epochs or self._epochs_at_current < self._cfg.evaluation_epochs:
            return self._current_update_epochs

        if self._cooldown_counter > 0:
            self._cooldown_counter -= 1
            return self._current_update_epochs

        candidate = self._suggest_adjustment(approx_kl=approx_kl, clipfrac=clipfrac)
        if candidate != self._current_update_epochs:
            self._set_value(
                candidate,
                epoch=epoch,
                approx_kl=approx_kl,
                clipfrac=clipfrac,
                set_cooldown=True,
                log_fn=logger.info,
                prefix="Auto-tuning",
            )

        return self._current_update_epochs

    def _collect_metrics(self) -> Optional[Tuple[float, float]]:
        if not (stats := getattr(self.context, "latest_losses_stats", None)):
            return None
        approx_kl = float(stats.get("approx_kl", 0.0))
        clipfrac = float(stats.get("clipfrac", 0.0))
        return None if approx_kl <= 0.0 and clipfrac <= 0.0 else (approx_kl, clipfrac)

    def _suggest_adjustment(self, *, approx_kl: float, clipfrac: float) -> int:
        current = self._current_update_epochs
        target = self._cfg.target_kl
        tolerance = self._cfg.kl_tolerance
        max_clipfrac = self._cfg.max_clipfrac

        if clipfrac > max_clipfrac or approx_kl > target * (1.0 + tolerance):
            return self._clamp(current - self._cfg.step_size)

        if approx_kl > 0.0 and approx_kl < target * (1.0 - tolerance):
            return self._clamp(current + self._cfg.step_size)

        return current

    def _set_value(
        self,
        new_value: int,
        *,
        epoch: int,
        approx_kl: float,
        clipfrac: float,
        set_cooldown: bool,
        log_fn,
        prefix: str,
    ) -> None:
        previous = self._current_update_epochs
        self._current_update_epochs = new_value
        self._epochs_at_current = 0
        if set_cooldown:
            self._cooldown_counter = self._cfg.cooldown_epochs

        log_fn(
            "%s update_epochs from %s to %s at epoch %s (approx_kl=%.4f clipfrac=%.3f)",
            prefix,
            previous,
            new_value,
            epoch,
            approx_kl,
            clipfrac,
        )

    def _clamp(self, value: int) -> int:
        return max(self._cfg.min_update_epochs, min(value, self._cfg.max_update_epochs))
