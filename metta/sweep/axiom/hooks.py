"""Hook system for tAXIOM pipelines."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from metta.sweep.axiom.core import Ctx


class Hook(ABC):
    """Base class for hooks that run at stage membranes.

    Hooks are non-mutating observers that can:
    - Log data
    - Save checkpoints
    - Track metrics
    - Validate data
    - Perform any side effects

    They must NOT modify the data passing through.
    """

    @abstractmethod
    def run(self, ctx: Ctx, stage_name: str, data: Any) -> None:
        """Execute the hook.

        Args:
            ctx: Pipeline context
            stage_name: Name of the stage that just executed
            data: Output data from the stage
        """
        pass
