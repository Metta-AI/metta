"""Base training component infrastructure."""

import logging
from enum import Enum
from typing import Any, Dict, Optional

from pydantic import Field

from metta.rl.training.context import TrainerContext

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

    _context: Optional[TrainerContext] = None

    def __init__(self, epoch_interval: int = 1, step_interval: int = 1) -> None:
        self._epoch_interval = epoch_interval
        self._step_interval = step_interval

    def register(self, context: TrainerContext) -> None:
        """Register this component with the trainer context."""

        self._context = context

    @property
    def context(self) -> TrainerContext:
        """Return the trainer context associated with this component."""

        if self._context is None:
            raise RuntimeError("TrainerComponent has not been registered with a TrainerContext")
        return self._context

    def on_step(self, infos: Dict[str, Any]) -> None:
        """Called after each environment step."""
        return None

    def on_epoch_end(self, epoch: int) -> None:
        """Called at the end of an epoch."""
        return None

    def on_training_complete(self) -> None:
        """Called when training completes successfully."""
        return None

    def on_failure(self) -> None:
        """Called when training fails."""
        return None
