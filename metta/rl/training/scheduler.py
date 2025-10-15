from __future__ import annotations

import math
from typing import Any, Callable, Literal, Optional

import numpy as np
from pydantic import Field

from metta.rl.training.component import TrainerComponent
from mettagrid.base_config import Config

AnnealStyle = Literal["linear", "cosine", "sawtooth"]


def _clamp01(x: float) -> float:
    return 0.0 if x <= 0.0 else 1.0 if x >= 1.0 else x


def _linear(progress: float, start: float, end: float) -> float:
    t = _clamp01(progress)
    return start + (end - start) * t


def _cosine(progress: float, start: float, end: float) -> float:
    t = _clamp01(progress)
    cos_t = (1 - math.cos(math.pi * t)) * 0.5
    return start + (end - start) * cos_t


def _sawtooth(progress: float, start: float, end: float) -> float:
    t = progress % 1.0
    return start + (end - start) * t


ANNEALERS: dict[AnnealStyle, Callable[[float, float, float], float]] = {
    "linear": _linear,
    "cosine": _cosine,
    "sawtooth": _sawtooth,
}


class HyperAnnealRule(Config):
    """Top-down hyperparameter annealing rule targeting a specific loss config attribute."""

    loss: str
    attr_path: str
    style: AnnealStyle
    start_value: float
    end_value: float
    start_epoch: Optional[int] = Field(default=None)
    end_epoch: Optional[int] = Field(default=None)
    start_agent_step: Optional[int] = Field(default=None)
    end_agent_step: Optional[int] = Field(default=None)

    def _progress(self, *, epoch: int, agent_step: int) -> float:
        if self.start_agent_step is not None and self.end_agent_step is not None:
            span = max(1, int(self.end_agent_step - self.start_agent_step))
            return (int(agent_step) - int(self.start_agent_step)) / span

        if self.start_epoch is not None and self.end_epoch is not None:
            span = max(1, int(self.end_epoch - self.start_epoch))
            return (int(epoch) - int(self.start_epoch)) / span

        return 1.0

    def apply(self, *, obj: object, epoch: int, agent_step: int) -> None:
        fn = ANNEALERS[self.style]
        value = fn(self._progress(epoch=epoch, agent_step=agent_step), self.start_value, self.end_value)
        _set_attr_path(obj, self.attr_path, float(value))


class MetricRule(Config):
    """Metric-driven hyper update rule targeting a specific loss config attribute."""

    loss: str
    attr_path: str
    metric_key: str
    transform: Optional[Callable[[float], float]] = None
    ema_beta: Optional[float] = Field(default=None, ge=0.0, lt=1.0)
    min_value: Optional[float] = None
    max_value: Optional[float] = None

    # runtime only
    _ema_state: Optional[float] = None

    def _read_metric(self, ctx) -> Optional[float]:
        stats_reporter = getattr(ctx, "stats_reporter", None)
        if stats_reporter is None or getattr(stats_reporter, "state", None) is None:
            return None
        values = stats_reporter.state.rollout_stats.get(self.metric_key)
        if not values:
            return None
        try:
            return float(np.mean(values))
        except Exception:
            return None

    def _smooth_and_clamp(self, value: float) -> float:
        if self.ema_beta is not None:
            if self._ema_state is None:
                self._ema_state = float(value)
            else:
                beta = float(self.ema_beta)
                self._ema_state = beta * float(self._ema_state) + (1.0 - beta) * float(value)
            value = float(self._ema_state)
        if self.min_value is not None:
            value = max(float(self.min_value), float(value))
        if self.max_value is not None:
            value = min(float(self.max_value), float(value))
        return float(value)

    def apply(self, *, obj: object, ctx) -> None:
        metric = self._read_metric(ctx)
        if metric is None:
            return
        if self.transform is not None:
            try:
                metric = float(self.transform(metric))
            except Exception:
                return
        _set_attr_path(obj, self.attr_path, self._smooth_and_clamp(metric))


class LossRunGate(Config):
    """Per-loss, per-phase run gating over epochs/steps/cycles."""

    loss: str
    phase: Literal["rollout", "train"]

    begin_at_epoch: Optional[int] = None
    end_at_epoch: Optional[int] = None
    begin_at_step: Optional[int] = None
    end_at_step: Optional[int] = None
    cycle_length: Optional[int] = None
    active_in_cycle: Optional[list[int]] = None

    def is_active(self, *, epoch: int, agent_step: int) -> bool:
        if self.begin_at_step is not None and self.end_at_step is not None:
            if not (int(self.begin_at_step) <= int(agent_step) < int(self.end_at_step)):
                return False
        else:
            begin = 0 if self.begin_at_epoch is None else int(self.begin_at_epoch)
            end = float("inf") if self.end_at_epoch is None else int(self.end_at_epoch)
            if not (begin <= int(epoch) < end):
                return False

        if self.cycle_length and self.active_in_cycle:
            epoch_in_cycle = (int(epoch) % int(self.cycle_length)) + 1
            if epoch_in_cycle not in self.active_in_cycle:
                return False
        return True


class SchedulerConfig(Config):
    hyper_anneal: list[HyperAnnealRule] = Field(default_factory=list)
    metric_rules: list[MetricRule] = Field(default_factory=list)
    run_gates: list[LossRunGate] = Field(default_factory=list)


class LossScheduler(TrainerComponent):
    """Trainer-level scheduler applying hyper updates and run gating per phase."""

    def __init__(self, config: SchedulerConfig) -> None:
        super().__init__(epoch_interval=1, step_interval=0)
        self.config = config

    def apply(self, *, phase: Literal["rollout", "train"]) -> None:
        ctx = self.context
        epoch = int(getattr(ctx, "epoch", 0))
        agent_step = int(getattr(ctx, "agent_step", 0))

        # 1) Apply run gates for the requested phase
        gates = getattr(ctx, "loss_run_gates", None)
        if gates is None:
            gates = {}
            ctx.loss_run_gates = gates

        for gate in self.config.run_gates:
            if gate.phase != phase:
                continue
            loss_name = gate.loss
            allowed = gate.is_active(epoch=epoch, agent_step=agent_step)
            entry = gates.get(loss_name)
            if entry is None:
                entry = {}
                gates[loss_name] = entry
            entry[phase] = bool(allowed)

        # 2) Apply hyper anneal rules and metric rules
        for rule in self.config.hyper_anneal:
            loss = ctx.losses.get(rule.loss)
            if loss is None:
                continue
            rule.apply(obj=loss.loss_cfg, epoch=epoch, agent_step=agent_step)

        for rule in self.config.metric_rules:
            loss = ctx.losses.get(rule.loss)
            if loss is None:
                continue
            rule.apply(obj=loss.loss_cfg, ctx=ctx)


def _set_attr_path(obj: object, path: str, value: Any) -> None:
    parts = path.split(".")
    target = obj
    for part in parts[:-1]:
        target = getattr(target, part)
    setattr(target, parts[-1], value)
