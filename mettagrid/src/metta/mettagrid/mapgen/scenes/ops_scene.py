"""Scene that renders operations."""

from typing import List

from metta.common.config import Config
from metta.mettagrid.mapgen.ops import Operation, apply_ops
from metta.mettagrid.mapgen.scene import Scene


class OpsSceneParams(Config):
    """Parameters for operations-based scene."""

    # List of operations to apply
    ops: List[Operation]

    # Initial fill material
    initial_fill: str = "empty"

    # Whether to check bounds when applying ops
    bounds_check: bool = True


class OpsScene(Scene[OpsSceneParams]):
    """Scene that renders a list of operations.

    This scene provides a bridge between the pure functional operations
    approach and the Scene system used by MapGen.
    """

    def render(self):
        """Apply operations to the scene's grid."""
        apply_ops(
            self.grid, self.params.ops, initial_fill=self.params.initial_fill, bounds_check=self.params.bounds_check
        )




