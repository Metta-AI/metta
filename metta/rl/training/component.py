"""Base training component infrastructure."""

import logging
from enum import Enum
from typing import Any, Optional

from pydantic import Field

from metta.rl.training import ComponentContext
from metta.rl.utils import should_run

logger = logging.getLogger(__name__)


class TrainerCallback(Enum):
    """Types of callbacks that can be invoked on trainer components."""

    STEP = "step"
    ROLLOUT_END = "rollout_end"
    EPOCH_END = "epoch"
    TRAINING_COMPLETE = "training_complete"
    FAILURE = "failure"


class TrainerComponent:
    """Base class for training components."""

    _master_only: bool = False
    _epoch_interval: int = Field(default=1, ge=1)
    _step_interval: int = Field(default=1, ge=1)

    _context: ComponentContext
    _prev_epoch_for_epoch_callbacks: Optional[int]

    def __init__(self, epoch_interval: int = 1, step_interval: int = 1) -> None:
        self._epoch_interval = epoch_interval
        self._step_interval = step_interval
        self._prev_epoch_for_epoch_callbacks = None

    def register(self, context: ComponentContext) -> None:
        """Register this component with the trainer context."""

        self._context = context
        self._prev_epoch_for_epoch_callbacks = None

    # ------------------------------------------------------------------
    # Interval helpers
    # ------------------------------------------------------------------
    def should_handle_step(self, *, current_step: int, previous_step: int) -> bool:
        """Return True when this component should receive a step callback."""

        interval = getattr(self, "_step_interval", 0)
        if interval <= 0:
            return False
        return should_run(current_step, interval, previous=previous_step)

    def should_handle_epoch(self, epoch: int) -> bool:
        """Return True when this component should receive an epoch callback."""

        interval = getattr(self, "_epoch_interval", 1)
        if interval == 0:
            return True
        should_handle = should_run(epoch, interval, previous=self._prev_epoch_for_epoch_callbacks)
        self._prev_epoch_for_epoch_callbacks = epoch
        return should_handle

    @property
    def context(self) -> ComponentContext:
        """Return the trainer context associated with this component."""
        return self._context

    def on_step(self, infos: list[dict[str, Any]]) -> None:
        """Called after each environment step."""
        pass

    def on_rollout_end(self) -> None:
        """Called at the end of rollout, before training begins."""
        pass

    def on_epoch_end(self, epoch: int) -> None:
        """Called at the end of an epoch."""
        pass

    def on_training_complete(self) -> None:
        """Called when training completes successfully."""
        pass

    def on_failure(self) -> None:
        """Called when training fails."""
        pass
