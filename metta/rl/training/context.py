"""Backward-compatible re-export for the component context."""

from __future__ import annotations

from .component_context import ComponentContext, TrainerState, TrainingEnvWindow

# Maintain the historical name for callers that still import TrainerContext.
TrainerContext = ComponentContext

__all__ = [
    "ComponentContext",
    "TrainerContext",
    "TrainerState",
    "TrainingEnvWindow",
]
