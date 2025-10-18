"""Automatic tuner for PPO-style update epochs."""

from __future__ import annotations

import logging
from collections import deque
from typing import Deque, Dict, Optional

from metta.rl.trainer_config import UpdateEpochAutoTunerConfig
from metta.rl.training import TrainerComponent

logger = logging.getLogger(__name__)


class UpdateEpochAutoTuner(TrainerComponent):
    """Dynamically adjusts ``update_epochs`` to maximize training throughput."""

    def __init__(self, config: UpdateEpochAutoTunerConfig):
        super().__init__(epoch_interval=1)
        self._cfg = config
        self._history: Dict[int, Deque[float]] = {}
        self._current_update_epochs: int = config.min_update_epochs
        self._best_value: Optional[int] = None
        self._best_throughput: float = 0.0
        self._direction: int = 1
        self._cooldown_counter: int = 0
        self._epochs_at_current: int = 0
        self._last_agent_step: Optional[int] = None
        self._last_wall_time: Optional[float] = None
        self._baseline_ready: bool = False
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
        self._best_value = clamped

        self._is_master = bool(context.distributed.is_master())
        self._last_agent_step = int(context.agent_step)
        self._last_wall_time = float(context.stopwatch.get_elapsed())
        self._baseline_ready = False
        self._epochs_at_current = 0
        self._cooldown_counter = 0
        self._direction = 1

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
        current_steps = int(ctx.agent_step)
        current_wall = float(ctx.stopwatch.get_elapsed())

        if not self._baseline_ready:
            self._baseline_ready = True
            self._last_agent_step = current_steps
            self._last_wall_time = current_wall
            return

        if self._last_agent_step is None or self._last_wall_time is None:
            self._last_agent_step = current_steps
            self._last_wall_time = current_wall
            return

        delta_steps = current_steps - self._last_agent_step
        delta_time = current_wall - self._last_wall_time

        self._last_agent_step = current_steps
        self._last_wall_time = current_wall

        if delta_steps <= 0 or delta_time <= 0:
            return

        throughput = delta_steps / delta_time
        self._epochs_at_current += 1

        target_value = self._current_update_epochs
        if self._is_master:
            target_value = self._evaluate_master(epoch=epoch, throughput=throughput)

        decided_value = ctx.distributed.broadcast_from_master(target_value if self._is_master else None)
        decided_value = self._clamp(int(decided_value))

        if decided_value != self._current_update_epochs:
            self._handle_external_switch(decided_value, epoch)

        ctx.config.update_epochs = decided_value
        self._history.setdefault(decided_value, deque(maxlen=self._cfg.metrics_window))

    def _evaluate_master(self, *, epoch: int, throughput: float) -> int:
        self._record_throughput(self._current_update_epochs, throughput)
        current_avg = self._average(self._current_update_epochs)

        if current_avg > self._best_throughput:
            self._best_throughput = current_avg
            self._best_value = self._current_update_epochs

        if epoch < self._cfg.warmup_epochs or self._epochs_at_current < self._cfg.evaluation_epochs:
            return self._current_update_epochs

        if self._cooldown_counter > 0:
            self._cooldown_counter -= 1
            return self._current_update_epochs

        candidate = self._select_candidate(current_avg)
        if candidate != self._current_update_epochs:
            self._switch_to(candidate, epoch)

        return self._current_update_epochs

    def _record_throughput(self, update_epochs: int, throughput: float) -> None:
        history = self._history.setdefault(update_epochs, deque(maxlen=self._cfg.metrics_window))
        history.append(throughput)

    def _average(self, update_epochs: int) -> float:
        history = self._history.get(update_epochs)
        if not history:
            return 0.0
        return sum(history) / len(history)

    def _select_candidate(self, current_avg: float) -> int:
        if (
            self._best_value is not None
            and self._best_value != self._current_update_epochs
            and current_avg > 0.0
            and self._best_throughput >= (1.0 + self._cfg.min_relative_improvement) * current_avg
        ):
            return self._best_value

        candidate = self._current_update_epochs + self._direction * self._cfg.step_size
        if candidate < self._cfg.min_update_epochs or candidate > self._cfg.max_update_epochs:
            self._direction *= -1
            candidate = self._current_update_epochs + self._direction * self._cfg.step_size

        candidate = self._clamp(candidate)
        if candidate == self._current_update_epochs:
            return candidate

        history = self._history.get(candidate)
        if history and len(history) >= self._cfg.evaluation_epochs:
            candidate_avg = self._average(candidate)
            if candidate_avg > self._best_throughput:
                self._best_throughput = candidate_avg
                self._best_value = candidate
            if current_avg > 0.0 and candidate_avg < (1.0 + self._cfg.min_relative_improvement) * current_avg:
                self._direction *= -1
                return self._current_update_epochs

        return candidate

    def _switch_to(self, new_value: int, epoch: int) -> None:
        previous = self._current_update_epochs
        self._current_update_epochs = new_value
        self._epochs_at_current = 0
        self._cooldown_counter = self._cfg.cooldown_epochs
        self._history.setdefault(new_value, deque(maxlen=self._cfg.metrics_window))

        if new_value > previous:
            self._direction = 1
        elif new_value < previous:
            self._direction = -1

        logger.info(
            "Auto-tuning update_epochs from %s to %s at epoch %s (best throughput=%.2f)",
            previous,
            new_value,
            epoch,
            self._best_throughput,
        )

    def _handle_external_switch(self, new_value: int, epoch: int) -> None:
        previous = self._current_update_epochs
        self._current_update_epochs = new_value
        self._epochs_at_current = 0
        if new_value > previous:
            self._direction = 1
        elif new_value < previous:
            self._direction = -1
        logger.debug(
            "Replica adopting update_epochs change from %s to %s at epoch %s",
            previous,
            new_value,
            epoch,
        )

    def _clamp(self, value: int) -> int:
        return max(self._cfg.min_update_epochs, min(value, self._cfg.max_update_epochs))
