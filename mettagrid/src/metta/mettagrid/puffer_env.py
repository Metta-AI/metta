"""MettaGridPufferEnv - PufferLib adapter for MettaGrid.

This class implements the PufferLib environment interface using the base MettaGrid system.
"""

from __future__ import annotations

from typing import Any, Optional

from metta.mettagrid.curriculum.core import Curriculum
from metta.mettagrid.level_builder import Level
from metta.mettagrid.puffer_base import MettaGridPufferBase


class MettaGridPufferEnv(MettaGridPufferBase):
    """
    PufferLib adapter for MettaGrid environments.

    This class provides a clean PufferLib interface for users who want to use
    MettaGrid environments with their own PufferLib training setup.
    No training features are included - this is purely for PufferLib compatibility.

    Inherits from:
    - MettaGridPufferBase: Base PufferLib integration with shared functionality
    """

    def __init__(
        self,
        curriculum: Curriculum,
        render_mode: Optional[str] = None,
        level: Optional[Level] = None,
        buf: Optional[Any] = None,
        **kwargs: Any,
    ):
        """
        Initialize PufferLib environment.

        Args:
            curriculum: Curriculum for task management
            render_mode: Rendering mode
            level: Optional pre-built level
            buf: PufferLib buffer object
            **kwargs: Additional arguments
        """
        # Initialize with base PufferLib functionality
        super().__init__(
            curriculum=curriculum,
            render_mode=render_mode,
            level=level,
            buf=buf,
            **kwargs,
        )
