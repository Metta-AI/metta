"""Shared utilities for clamping training resource usage in ABES recipes."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from metta.tools.train import TrainTool


@dataclass(frozen=True)
class ClampLimits:
    batch_cap: int
    minibatch_cap: int
    bptt_cap: int
    forward_pass_cap: int
    disable_compile: bool = True
    auto_workers: Optional[bool] = None
    num_workers_cap: Optional[int] = None
    async_factor_cap: Optional[int] = None


def clamp_training_resources(tool: TrainTool, limits: ClampLimits) -> TrainTool:
    """Clamp trainer/environment settings to keep memory usage manageable."""

    trainer_updates: dict[str, object] = {}
    env_updates: dict[str, object] = {}

    trainer = tool.trainer
    env = tool.training_env

    if limits.disable_compile and getattr(trainer, "compile", False):
        trainer_updates["compile"] = False

    bptt_horizon = min(trainer.bptt_horizon, limits.bptt_cap)
    if bptt_horizon != trainer.bptt_horizon:
        trainer_updates["bptt_horizon"] = bptt_horizon
    else:
        bptt_horizon = trainer.bptt_horizon

    raw_minibatch = min(trainer.minibatch_size, limits.minibatch_cap)
    minibatch_size = max(bptt_horizon, raw_minibatch - raw_minibatch % bptt_horizon)
    if minibatch_size == 0:
        minibatch_size = bptt_horizon
    if minibatch_size != trainer.minibatch_size:
        trainer_updates["minibatch_size"] = minibatch_size
    else:
        minibatch_size = trainer.minibatch_size

    raw_batch = min(trainer.batch_size, limits.batch_cap)
    batch_size = max(minibatch_size, raw_batch - raw_batch % minibatch_size)
    if batch_size == 0:
        batch_size = minibatch_size
    if batch_size != trainer.batch_size:
        trainer_updates["batch_size"] = batch_size

    if trainer_updates:
        tool.trainer = trainer.model_copy(update=trainer_updates)

    if env.forward_pass_minibatch_target_size > limits.forward_pass_cap:
        env_updates["forward_pass_minibatch_target_size"] = limits.forward_pass_cap

    if limits.auto_workers is not None and env.auto_workers != limits.auto_workers:
        env_updates["auto_workers"] = limits.auto_workers

    if limits.num_workers_cap is not None and env.num_workers > limits.num_workers_cap:
        env_updates["num_workers"] = limits.num_workers_cap

    if (
        limits.async_factor_cap is not None
        and env.async_factor > limits.async_factor_cap
    ):
        env_updates["async_factor"] = limits.async_factor_cap

    if env_updates:
        tool.training_env = env.model_copy(update=env_updates)

    return tool
