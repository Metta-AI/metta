"""Compatibility shim for legacy imports.

``TrainerState`` has been merged into ``TrainerContext``. This module keeps the
old import path working while the rest of the codebase migrates to the new
context-based API.
"""

from __future__ import annotations

from metta.rl.training.context import TrainerContext

# Backwards compatibility alias
TrainerState = TrainerContext

__all__ = ["TrainerState"]
