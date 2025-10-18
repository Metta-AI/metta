from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable, Literal, Optional

import numpy as np

AnnealStyle = Literal["linear", "cosine", "sawtooth"]


def _clamp01(x: float) -> float:
    return 0.0 if x <= 0.0 else 1.0 if x >= 1.0 else x


def linear(progress: float, start: float, end: float) -> float:
    t = _clamp01(progress)
    return start + (end - start) * t


def cosine(progress: float, start: float, end: float) -> float:
    # Half cosine from start to end
    t = _clamp01(progress)
    cos_t = (1 - math.cos(math.pi * t)) * 0.5
    return start + (end - start) * cos_t


def sawtooth(progress: float, start: float, end: float) -> float:
    # Wrap progress to [0,1) and do linear within the cycle
    t = progress % 1.0
    return start + (end - start) * t


ANNEALERS: dict[AnnealStyle, Callable[[float, float, float], float]] = {
    "linear": linear,
    "cosine": cosine,
    "sawtooth": sawtooth,
}


@dataclass(slots=True)
class HyperSchedule:
    """A single hyperparameter schedule.

    Attributes:
        attr_path: Dot or attribute path on the loss config, e.g. "clip_coef" or
            "vtrace.rho_clip".
        style: Anneal style name.
        start_value: Starting value of the hyperparameter.
        end_value: Ending value of the hyperparameter.
        start_epoch: Optional start epoch for scheduling.
        end_epoch: Optional end epoch for scheduling.
        start_agent_step: Optional start step for scheduling.
        end_agent_step: Optional end step for scheduling.
    """

    attr_path: str
    style: AnnealStyle
    start_value: float
    end_value: float
    start_epoch: Optional[int] = None
    end_epoch: Optional[int] = None
    start_agent_step: Optional[int] = None
    end_agent_step: Optional[int] = None

    def compute_progress(self, *, epoch: int, agent_step: int) -> float:
        # Prefer step-based schedule if provided; otherwise epoch-based.
        if self.start_agent_step is not None and self.end_agent_step is not None:
            span = max(1, self.end_agent_step - self.start_agent_step)
            return (agent_step - self.start_agent_step) / span

        if self.start_epoch is not None and self.end_epoch is not None:
            span = max(1, self.end_epoch - self.start_epoch)
            return (epoch - self.start_epoch) / span

        # If no bounds supplied, consider full [0,1] immediately
        return 1.0

    # Unified scheduler API
    def update(self, obj: object, ctx) -> None:
        """update method for scheduler classes must have the same signature as they are all called by Loss"""
        epoch = getattr(ctx, "epoch", 0)
        agent_step = getattr(ctx, "agent_step", 0)
        progress = self.compute_progress(epoch=int(epoch), agent_step=int(agent_step))
        fn = ANNEALERS[self.style]
        new_value = fn(progress, self.start_value, self.end_value)
        _set_attr_path(obj, self.attr_path, new_value)


def _set_attr_path(obj: object, path: str, value: float) -> None:
    parts = path.split(".")
    target = obj
    for part in parts[:-1]:
        target = getattr(target, part)
    setattr(target, parts[-1], value)


# -----------------------------------------------------------------------------
# Metric-driven scheduling based on environment stats accumulated during rollout
# -----------------------------------------------------------------------------


@dataclass(slots=True)
class MetricSchedule:
    """Set a hyperparameter from an aggregated environment metric.

    Attributes:
        attr_path: Dot/attribute path on the loss config to write, e.g. "ent_coef".
        metric_key: Key within `context.stats_reporter.state.rollout_stats` to read.
        transform: Optional mapping from metric value -> hyper value (applied before EMA/clamp).
        ema_beta: Optional EMA smoothing factor in [0, 1); when set, applies EMA over metric.
        min_value: Optional lower clamp on resulting hyper value.
        max_value: Optional upper clamp on resulting hyper value.
    """

    attr_path: str
    metric_key: str
    transform: Optional[Callable[[float], float]] = None
    ema_beta: Optional[float] = None
    min_value: Optional[float] = None
    max_value: Optional[float] = None

    # Internal state for EMA smoothing
    _ema_state: Optional[float] = None

    def _get_metric_mean(self, ctx) -> Optional[float]:
        """Return the mean of the configured metric for the current epoch, if available."""

        stats_reporter = getattr(ctx, "stats_reporter", None)
        if stats_reporter is None or getattr(stats_reporter, "state", None) is None:
            return None

        values = stats_reporter.state.rollout_stats.get(self.metric_key)
        if not values:
            return None

        try:
            mean_value = float(np.mean(values))
        except Exception:
            return None
        return mean_value

    def _apply_smoothing_and_clamp(self, value: float) -> float:
        # Optional EMA smoothing
        if self.ema_beta is not None:
            if self._ema_state is None:
                self._ema_state = value
            else:
                beta = float(self.ema_beta)
                self._ema_state = beta * self._ema_state + (1.0 - beta) * value
            value = self._ema_state

        # Optional clamping
        if self.min_value is not None:
            value = max(float(self.min_value), value)
        if self.max_value is not None:
            value = min(float(self.max_value), value)
        return float(value)

    def compute_value(self, ctx) -> Optional[float]:
        metric = self._get_metric_mean(ctx)
        if metric is None:
            return None
        if self.transform is not None:
            try:
                metric = float(self.transform(metric))
            except Exception:
                # If transform fails, skip update for this step
                return None
        return self._apply_smoothing_and_clamp(metric)

    def update(self, *, obj: object, ctx) -> None:
        """update method for scheduler classes must have the same signature as they are all called by Loss"""
        value = self.compute_value(ctx)
        if value is None:
            return
        _set_attr_path(obj, self.attr_path, value)


# -----------------------------------------------------------------------------
# Loss run scheduling (enable/disable loss per phase over epochs/steps/cycles)
# Single schedule type used by configs as rollout_sched/train_sched
# -----------------------------------------------------------------------------


@dataclass(slots=True)
class PhaseRunSchedule:
    """Schedule for a single phase (e.g., rollout or train). You can supply absolute windows by epoch or agent step
    count, a cyclical gating pattern (ie run every 1st and 2nd epoch on a 3 epoch cycle), or both.

    If you provide both start and end bounds (either epoch or agent steps) AND cyclical gating then a phase runs when
    BOTH conditions hold:
      1) It is within the absolute window (prefer steps; otherwise epochs)
      2) It matches the cyclical gating (if configured)

    Notes:
      - Epoch cycle indexing is 1-based (1..cycle_length).
      - Absolute windows are half-open: [begin, end).

    Example:
      Run only the training phase for the first 100 epochs:
        train_sched = PhaseRunSchedule(begin_at_epoch=0, end_at_epoch=100)

      Run rollout every 3 epochs, only on the 1st epoch of each cycle:
        rollout_sched = PhaseRunSchedule(cycle_length=3, active_in_cycle=[1])

      Run train between steps 5M and 20M (steps override epochs when provided):
        train_sched = PhaseRunSchedule(begin_at_step=5_000_000, end_at_step=20_000_000)
    """

    # Absolute window (by epoch)
    begin_at_epoch: Optional[int] = None
    end_at_epoch: Optional[int] = None

    # Absolute window (by agent step)
    begin_at_step: Optional[int] = None
    end_at_step: Optional[int] = None

    # Cyclical schedule (by epochs)
    cycle_length: Optional[int] = None
    active_in_cycle: Optional[list[int]] = None

    def is_active(self, *, epoch: int, agent_step: int) -> bool:
        # Prefer step-based window if fully specified
        if self.begin_at_step is not None and self.end_at_step is not None:
            if not (self.begin_at_step <= agent_step < self.end_at_step):
                return False
        else:
            # Fallback to epoch window if provided
            begin = 0 if self.begin_at_epoch is None else self.begin_at_epoch
            end = float("inf") if self.end_at_epoch is None else self.end_at_epoch
            if not (begin <= epoch < end):
                return False

        # Cycle gate (optional)
        if self.cycle_length and self.active_in_cycle:
            # 1-indexed epoch within cycle
            epoch_in_cycle = (epoch % self.cycle_length) + 1
            if epoch_in_cycle not in self.active_in_cycle:
                return False

        return True
