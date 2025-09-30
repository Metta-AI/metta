"""Adaptive memory length scheduling for transformer policies."""

from __future__ import annotations

from typing import List

from pydantic import BaseModel, ConfigDict, Field, field_validator

from metta.rl.training import TrainerComponent


class MemorySchedulerConfig(BaseModel):
    enabled: bool = False
    milestones: List[int] = Field(default_factory=list)
    memory_lengths: List[int] = Field(default_factory=list)
    model_config = ConfigDict(validate_assignment=True)

    @field_validator("memory_lengths")
    @classmethod
    def _validate_lengths(cls, values: List[int], info):  # type: ignore[override]
        milestones = info.data.get("milestones", [])
        if values and len(values) != len(milestones):
            raise ValueError("memory_lengths and milestones must have the same length")
        if any(v < 0 for v in values):
            raise ValueError("memory_lengths must be non-negative")
        return values


class MemoryScheduler(TrainerComponent):
    """Adjusts transformer memory length at configured milestones."""

    _master_only = True

    def __init__(self, config: MemorySchedulerConfig) -> None:
        super().__init__(epoch_interval=1)
        self.config = config
        self._cursor = 0

    def on_epoch_end(self, epoch: int) -> None:  # type: ignore[override]
        if not self.config.enabled:
            return
        if self._cursor >= len(self.config.milestones):
            return
        milestone = self.config.milestones[self._cursor]
        if epoch < milestone:
            return
        target_memory = self.config.memory_lengths[self._cursor]
        self._cursor += 1

        policy = getattr(self.context, "policy", None)
        if policy is None or not hasattr(policy, "update_memory_len"):
            return
        try:
            policy.update_memory_len(int(target_memory))
        except Exception:  # pragma: no cover - defensive
            pass


__all__ = ["MemorySchedulerConfig", "MemoryScheduler"]
