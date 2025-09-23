"""Base training component infrastructure."""

import logging
from enum import Enum
from typing import Any, Optional

from pydantic import Field

from metta.rl.training import ComponentContext

logger = logging.getLogger(__name__)


class TrainerCallback(Enum):
    """Types of callbacks that can be invoked on trainer components."""

    STEP = "step"
    EPOCH_END = "epoch"
    TRAINING_COMPLETE = "training_complete"
    FAILURE = "failure"


class TrainerComponent:
    """Base class for training components."""

    _master_only: bool = False
    _epoch_interval: int = Field(default=1, ge=1)
    _step_interval: int = Field(default=1, ge=1)

    _context: Optional[ComponentContext] = None

    def __init__(self, epoch_interval: int = 1, step_interval: int = 1) -> None:
        self._epoch_interval = epoch_interval
        self._step_interval = step_interval

    def register(self, context: ComponentContext) -> None:
        """Register this component with the trainer context."""

        self._context = context

    # ------------------------------------------------------------------
    # Interval helpers
    # ------------------------------------------------------------------
    def should_handle_step(self, *, current_step: int, previous_step: int) -> bool:
        """Return True when this component should receive a step callback."""

        interval = getattr(self, "_step_interval", 0)
        if interval <= 0:
            return False
        return current_step // interval > previous_step // interval

    def should_handle_epoch(self, epoch: int) -> bool:
        """Return True when this component should receive an epoch callback."""

        interval = getattr(self, "_epoch_interval", 1)
        if interval == 0:
            return True
        return epoch % interval == 0

    @property
    def context(self) -> ComponentContext:
        """Return the trainer context associated with this component."""

        if self._context is None:
            raise RuntimeError("TrainerComponent has not been registered with a ComponentContext")
        return self._context

    def on_step(self, infos: list[dict[str, Any]]) -> None:
        """Called after each environment step."""
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
