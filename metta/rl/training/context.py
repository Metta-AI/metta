"""Context wrapper exposing trainer state to components without direct imports."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:  # pragma: no cover - import used only for type checking
    from metta.rl.trainer import Trainer


class TrainerContext:
    """Lightweight proxy that forwards attribute access to the trainer.

    Trainer components receive this context instead of the concrete ``Trainer``
    to avoid circular import issues while still being able to interact with the
    trainer's state and helper methods.
    """

    __slots__ = ("_trainer",)

    def __init__(self, trainer: "Trainer") -> None:
        object.__setattr__(self, "_trainer", trainer)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._trainer, name)

    def __setattr__(self, name: str, value: Any) -> None:
        setattr(self._trainer, name, value)

    def __delattr__(self, name: str) -> None:
        delattr(self._trainer, name)

    @property
    def trainer(self) -> "Trainer":
        """Expose the underlying trainer for the rare cases it is required."""

        return self._trainer
