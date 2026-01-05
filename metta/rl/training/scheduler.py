from __future__ import annotations

import math
from typing import Any, Callable, Iterable, Literal, Optional

import numpy as np
from pydantic import Field
from pydantic.json_schema import SkipJsonSchema

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


class ScheduleRule(Config):
    """Unified rule for scheduling config updates.

    Supports two modes:
    - progress: anneal between start/end values over epoch or agent_step ranges
    - metric: derive value from a reported metric with optional transform/EMA/clamp
    """

    # Target selection (path relative to TrainerConfig)
    target_path: str

    # Mode selection (None => inferred: metric if metric_key set else progress)
    mode: Optional[Literal["progress", "metric"]] = None

    # Progress/anneal fields
    style: AnnealStyle = "linear"
    start_value: Optional[float] = None
    end_value: Optional[float] = None
    start_epoch: Optional[int] = Field(default=None)
    end_epoch: Optional[int] = Field(default=None)
    start_agent_step: Optional[int] = Field(default=None)
    end_agent_step: Optional[int] = Field(default=None)

    # Metric-derived fields
    metric_key: Optional[str] = None
    transform: Optional[SkipJsonSchema[Callable[[float], float]]] = None
    ema_beta: Optional[float] = Field(default=None, ge=0.0, lt=1.0)
    min_value: Optional[float] = None
    max_value: Optional[float] = None

    # runtime only
    _ema_state: Optional[float] = None

    # -------------- progress helpers --------------
    def _progress(self, *, epoch: int, agent_step: int) -> float:
        if self.start_agent_step is not None and self.end_agent_step is not None:
            span = max(1, int(self.end_agent_step - self.start_agent_step))
            return (int(agent_step) - int(self.start_agent_step)) / span

        if self.start_epoch is not None and self.end_epoch is not None:
            span = max(1, int(self.end_epoch - self.start_epoch))
            return (int(epoch) - int(self.start_epoch)) / span

        return 1.0

    def _apply_progress(self, *, obj: object, epoch: int, agent_step: int) -> None:
        if self.start_value is None or self.end_value is None:
            return

        # Check if current step/epoch is within the rule's range
        if self.start_agent_step is not None and self.end_agent_step is not None:
            if not (self.start_agent_step <= agent_step < self.end_agent_step):
                return
        elif self.start_epoch is not None and self.end_epoch is not None:
            if not (self.start_epoch <= epoch < self.end_epoch):
                return

        fn = ANNEALERS[self.style]
        value = fn(self._progress(epoch=epoch, agent_step=agent_step), float(self.start_value), float(self.end_value))
        _set_attr_path(obj, self.target_path, float(value))

    # -------------- metric helpers --------------
    def _read_metric(self, ctx) -> Optional[float]:
        if not self.metric_key:
            return None
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

    def _apply_metric(self, *, obj: object, ctx) -> None:
        metric = self._read_metric(ctx)
        if metric is None:
            return
        if self.transform is not None:
            try:
                metric = float(self.transform(metric))
            except Exception:
                return
        _set_attr_path(obj, self.target_path, self._smooth_and_clamp(metric))

    # -------------- main apply --------------
    def apply(self, *, obj: object, ctx) -> None:
        epoch = getattr(ctx, "epoch", 0)
        agent_step = getattr(ctx, "agent_step", 0)
        mode = self.mode or ("metric" if self.metric_key else "progress")
        if mode == "metric":
            self._apply_metric(obj=obj, ctx=ctx)
        else:
            self._apply_progress(obj=obj, epoch=int(epoch), agent_step=int(agent_step))


class LossRunGate(Config):
    """Per-loss, per-phase run gating over epochs/steps/cycles."""

    loss_instance_name: str
    phase: Literal["rollout", "train"]

    begin_at_epoch: Optional[int] = None
    end_at_epoch: Optional[int] = None
    begin_at_step: Optional[int] = None
    end_at_step: Optional[int] = None
    cycle_length: Optional[int] = None
    active_in_cycle: Optional[list[int]] = None

    def is_active(self, *, epoch: int, agent_step: int) -> bool:
        # Use step-based logic if either step field is set
        if self.begin_at_step is not None or self.end_at_step is not None:
            begin = 0 if self.begin_at_step is None else int(self.begin_at_step)
            end = float("inf") if self.end_at_step is None else int(self.end_at_step)
            if not (begin <= int(agent_step) < end):
                return False
        else:
            # Use epoch-based logic when no step fields are set
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
    # Unified rules list only
    rules: list[ScheduleRule] = Field(default_factory=list)
    run_gates: list[LossRunGate] = Field(default_factory=list)


class LossScheduler(TrainerComponent):
    """Trainer-level scheduler for both:
    1) updates to any hyperparameters and
    2) gating as to whether entire losses should run or not given the current epoch or agent step.

    Usage (from an experiments recipe):

        from metta.tools.train import TrainTool
        from metta.rl.training import TrainingEnvironmentConfig, EvaluatorConfig
        from metta.rl.training.scheduler import (
            SchedulerConfig, ScheduleRule, LossRunGate
        )

        def train(...):
            return TrainTool(
                training_env=TrainingEnvironmentConfig(curriculum=...),
                evaluator=EvaluatorConfig(simulations=...),

                # Configure the scheduler with run gates and rules
                scheduler=SchedulerConfig(
                    # Gate when a loss is allowed to run per phase
                    run_gates=[
                        # Start PPO training after epoch 5
                        LossRunGate(loss_instance_name="ppo_actor", phase="train", begin_at_epoch=5),
                        # Allow PPO rollout from the start (explicit for clarity)
                        LossRunGate(loss_instance_name="ppo_actor", phase="rollout", begin_at_epoch=0),
                    ],

                    # Hyperparameter updates (progress- or metric-driven)
                    rules=[
                        # Linearly anneal PPO actor entropy coefficient from 0.02 -> 0.0 over epochs 0..50
                        ScheduleRule(
                            target_path="losses.ppo_actor.ent_coef",
                            mode="progress",
                            style="linear",
                            start_value=0.02,
                            end_value=0.0,
                            start_epoch=0,
                            end_epoch=50,
                        ),

                        # Drive a hyperparameter from a rollout metric (with smoothing and clamping)
                        ScheduleRule(
                            target_path="losses.ppo_critic.vf_coef",
                            mode="metric",
                            metric_key="your_metric_key",  # key found in StatsReporter.state.rollout_stats
                            ema_beta=0.9,
                            min_value=0.1,
                            max_value=1.0,
                        ),
                    ],
                ),
            )

    Notes:
    - "target_path" is relative to the TrainerConfig root (e.g., "losses.ppo_actor.ent_coef").
    - For metric-driven rules, "metric_key" must exist in rollout stats accumulated by
      the StatsReporter (flattened keys from env infos).
    """

    def __init__(self, config: SchedulerConfig) -> None:
        super().__init__(epoch_interval=1, step_interval=0)
        self.config = config

    def register(self, context) -> None:
        """Attach context and initialize gates for the very first rollout."""
        super().register(context)

        # Initialize rollout gates/hypers for epoch 0 before the first rollout
        self.apply(phase="rollout")

    def apply(self, *, phase: Literal["rollout", "train"]) -> None:
        epoch = self.context.epoch
        agent_step = self.context.agent_step
        # One-time supervisor teardown after teacher phase to avoid extra forwards.
        env_obj = getattr(self.context, "env", None)
        # 1) Apply run gates for the requested phase
        gates = getattr(self.context, "loss_run_gates", None)
        if gates is None:
            gates = {}
            self.context.loss_run_gates = gates

        # OR-combine semantics across gates for the same loss/phase.
        # Re-initialize per apply call to False iff there exists at least one gate
        # for this (loss, phase) in the current pass; otherwise leave unset so
        # downstream defaults (True) still apply when no gates exist for a phase.
        seen_loss_phase: set[tuple[str, str]] = set()

        for gate in self.config.run_gates:
            if gate.phase != phase:
                continue
            loss_name = gate.loss_instance_name
            allowed = gate.is_active(epoch=epoch, agent_step=agent_step)
            entry = gates.get(loss_name)
            if entry is None:
                entry = {}
                gates[loss_name] = entry
            key = (loss_name, phase)
            if key not in seen_loss_phase:
                # Initialize to False for this apply pass; subsequent gates OR into it
                entry[phase] = False
                seen_loss_phase.add(key)
            entry[phase] = bool(entry[phase]) or bool(allowed)

        # If a teacher/supervisor rollout loss is gated OFF, disable the supervisor to avoid extra forwards.
        if phase == "rollout":
            sup_off = False
            for loss_name, entry in gates.items():
                if loss_name in {"sliced_scripted_cloner", "supervisor", "sliced_kickstarter", "logit_kickstarter"}:
                    if entry.get("rollout") is False:
                        sup_off = True
                        break
            if sup_off:
                env_obj = getattr(self.context, "env", None)
                driver = getattr(getattr(env_obj, "vecenv", None), "driver_env", None)
                if driver and hasattr(driver, "disable_supervisor"):
                    driver.disable_supervisor()
                te_cfg = getattr(getattr(self.context, "config", None), "training_env", None)
                if te_cfg:
                    te_cfg.supervisor_policy_uri = None

        # 2) Apply unified rules
        for rule in self.config.rules:
            rule.apply(obj=self.context.config, ctx=self.context)

        # Keep optimizer learning rate in sync with TrainerConfig if it changed.
        self._sync_optimizer_from_config()

        # 3) If the train phase is gated to false, restrict which experience keys
        #    must be present based on which losses are active for rollout.
        #    Check if any loss has train phase disabled (gated to False)
        train_disabled = False
        for loss_name in self.context.losses.keys():
            entry = gates.get(loss_name)
            if entry and entry.get("train") is False:
                train_disabled = True
                break
        if train_disabled:
            self._update_experience_store_keys_for_rollout()

    # ----------------- Trainer callbacks -----------------
    def on_rollout_end(self) -> None:
        """Prepare gates and hypers for the upcoming train phase."""
        self.apply(phase="train")

    def on_epoch_end(self, epoch: int) -> None:
        """Prepare gates and hypers for the next epoch's rollout."""
        self.apply(phase="rollout")

    # ----------------- Experience key management -----------------
    def _active_rollout_loss_names(self) -> Iterable[str]:
        """Return loss instance names that are active for rollout in the current epoch."""
        gates = getattr(self.context, "loss_run_gates", None) or {}
        for loss_name in self.context.losses.keys():
            entry = gates.get(loss_name)
            if not entry:
                # No gates configured for this loss; default to active.
                yield loss_name
                continue
            if bool(entry.get("rollout", True)):
                yield loss_name

    def _update_experience_store_keys_for_rollout(self) -> None:
        """Update experience buffer to only require keys for active rollout losses."""
        context = self.context
        experience = getattr(context, "experience", None)
        if experience is None:
            return

        # Always include policy experience spec keys.
        policy_spec = getattr(experience, "policy_experience_spec", None) or context.policy.get_agent_experience_spec()
        active_keys: set[Any] = set(policy_spec.keys(include_nested=True, leaves_only=True))

        # Include spec keys from losses that are active for rollout this epoch.
        for loss_name in self._active_rollout_loss_names():
            loss = context.losses.get(loss_name)
            if loss is None:
                continue
            spec = loss.get_experience_spec()
            active_keys.update(spec.keys(include_nested=True, leaves_only=True))

        active_keys.update({"reward_baseline", "slot_id", "loss_profile_id", "is_trainable_agent"})

        # If for some reason no keys were found, fall back to writing all keys.
        if not active_keys:
            experience.reset_store_keys()
            return

        experience.set_store_keys(active_keys)

    def _sync_optimizer_from_config(self) -> None:
        """Propagate TrainerConfig optimizer learning_rate into the live optimizer."""
        optimizer = self.context.optimizer
        trainer_cfg = self.context.config
        lr = float(trainer_cfg.optimizer.learning_rate)
        for group in optimizer.param_groups:
            group["lr"] = lr


def _set_attr_path(obj: object, path: str, value: Any) -> None:
    parts = path.split(".")
    target = obj
    for part in parts[:-1]:
        target = getattr(target, part)
    setattr(target, parts[-1], value)
