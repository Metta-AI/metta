"""Automatic tuner for PPO-style update epochs."""

from __future__ import annotations

import logging

from metta.rl.trainer_config import UpdateEpochAutoTunerConfig
from metta.rl.training import TrainerComponent

logger = logging.getLogger(__name__)


class UpdateEpochAutoTuner(TrainerComponent):
    """Adjusts ``update_epochs`` online using KL and clipfrac."""

    def __init__(self, config: UpdateEpochAutoTunerConfig):
        super().__init__(epoch_interval=1)
        self._cfg = config
        self._current = config.min_update_epochs
        self._cooldown = 0
        self._epochs_here = 0
        self._is_master = False

    def register(self, context) -> None:  # type: ignore[override]
        super().register(context)
        if not self._cfg.enabled:
            return

        clamped = self._clamp(int(getattr(context.config, "update_epochs", self._cfg.min_update_epochs)))
        if clamped != context.config.update_epochs:
            logger.info("Clamping initial update_epochs from %s to %s", context.config.update_epochs, clamped)

        context.config.update_epochs = self._current = clamped
        self._is_master = bool(context.distributed.is_master())
        self._epochs_here = self._cooldown = 0

    def on_epoch_end(self, epoch: int) -> None:  # type: ignore[override]
        if not self._cfg.enabled:
            return

        stats = getattr(self.context, "latest_losses_stats", None)
        if not stats:
            return

        approx_kl = float(stats.get("approx_kl", 0.0))
        clipfrac = float(stats.get("clipfrac", 0.0))
        if approx_kl <= 0.0 and clipfrac <= 0.0:
            return

        self._epochs_here += 1

        target = self._current
        if self._is_master:
            target = self._decide(epoch, approx_kl, clipfrac)

        decided = self._clamp(int(self.context.distributed.broadcast_from_master(target if self._is_master else None)))
        if decided != self._current:
            self._apply(decided, epoch, approx_kl, clipfrac, cooldown=False, log=logger.debug)

        self.context.config.update_epochs = decided

    def _decide(self, epoch: int, approx_kl: float, clipfrac: float) -> int:
        if epoch < self._cfg.warmup_epochs or self._epochs_here < self._cfg.evaluation_epochs:
            return self._current

        if self._cooldown > 0:
            self._cooldown -= 1
            return self._current

        candidate = self._suggest(approx_kl, clipfrac)
        if candidate != self._current:
            self._apply(candidate, epoch, approx_kl, clipfrac, cooldown=True, log=logger.info)

        return self._current

    def _suggest(self, approx_kl: float, clipfrac: float) -> int:
        target = self._cfg.target_kl
        tol = self._cfg.kl_tolerance

        if clipfrac > self._cfg.max_clipfrac or approx_kl > target * (1.0 + tol):
            return self._clamp(self._current - self._cfg.step_size)

        if 0.0 < approx_kl < target * (1.0 - tol):
            return self._clamp(self._current + self._cfg.step_size)

        return self._current

    def _apply(self, new_value: int, epoch: int, approx_kl: float, clipfrac: float, *, cooldown: bool, log) -> None:
        previous = self._current
        self._current = new_value
        self._epochs_here = 0
        if cooldown:
            self._cooldown = self._cfg.cooldown_epochs

        log(
            "update_epochs %s→%s at epoch %s (approx_kl=%.4f clipfrac=%.3f)",
            previous,
            new_value,
            epoch,
            approx_kl,
            clipfrac,
        )

    def _clamp(self, value: int) -> int:
        return max(self._cfg.min_update_epochs, min(value, self._cfg.max_update_epochs))
