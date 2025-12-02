"""Automatic tuner for PPO-style update epochs."""

from __future__ import annotations

import logging

from pydantic import Field, model_validator

from metta.rl.training import TrainerComponent
from mettagrid.base_config import Config

logger = logging.getLogger(__name__)


class UpdateEpochAutoTunerConfig(Config):
    """Configuration for automatically tuning update epochs."""

    min_update_epochs: int = Field(default=1, ge=1)
    max_update_epochs: int = Field(default=8, ge=1)
    step_size: int = Field(default=1, ge=1)
    evaluation_epochs: int = Field(default=0, ge=0)
    warmup_epochs: int = Field(default=2, ge=0)
    cooldown_epochs: int = Field(default=2, ge=0)
    target_kl: float = Field(default=0.015, ge=0.0)
    kl_tolerance: float = Field(default=0.3, ge=0.0)
    max_clipfrac: float = Field(default=0.3, ge=0.0, le=1.0)

    @model_validator(mode="after")
    def validate_bounds(self) -> "UpdateEpochAutoTunerConfig":
        if self.max_update_epochs < self.min_update_epochs:
            raise ValueError("max_update_epochs must be >= min_update_epochs")
        return self

    @property
    def enabled(self) -> bool:
        return self.evaluation_epochs > 0


class UpdateEpochAutoTuner(TrainerComponent):
    """Adjusts ``update_epochs`` online using KL and clipfrac."""

    def __init__(self, config: UpdateEpochAutoTunerConfig):
        super().__init__(epoch_interval=1)
        self._cfg = config
        self._current = config.min_update_epochs
        self._cooldown = 0
        self._epochs_here = 0

    def register(self, context) -> None:  # type: ignore[override]
        super().register(context)
        if not self._cfg.enabled:
            return

        clamped = self._clamp(int(getattr(context.config, "update_epochs", self._cfg.min_update_epochs)))
        if clamped != context.config.update_epochs:
            logger.info("Clamping initial update_epochs from %s to %s", context.config.update_epochs, clamped)

        context.config.update_epochs = self._current = clamped
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

        is_master = bool(self.context.distributed.is_master())
        target = self._decide(epoch, approx_kl, clipfrac) if is_master else self._current

        decided = self._clamp(int(self.context.distributed.broadcast_from_master(target if is_master else None)))
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
