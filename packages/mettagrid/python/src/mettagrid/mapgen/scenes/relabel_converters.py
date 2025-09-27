from typing import Dict, List

import numpy as np

from mettagrid.config.config import Config
from mettagrid.mapgen.scene import Scene


class RelabelConvertersParams(Config):
    target_counts: Dict[str, int]
    # Which names to consider as converter candidates to relabel
    source_types: List[str] = [
        "generator_red",
        "generator_blue",
        "generator_green",
        "lab",
    ]


class RelabelConverters(Scene[RelabelConvertersParams]):
    """
    Reassign converter object types in-place to match target_counts across the map.

    Typical use: place a single converter type everywhere (to get symmetry via mirroring),
    then relabel to the final mix.
    """

    def render(self):
        grid = self.grid

        # Gather all candidate positions (row, col)
        ys, xs = np.where(np.isin(grid, self.params.source_types))
        positions = list(zip(ys.tolist(), xs.tolist(), strict=True))
        if not positions:
            return

        # Stable order: top-to-bottom, left-to-right
        positions.sort()

        # Create the assignment list: sequence of target types by desired counts
        assignment: List[str] = []
        for t, cnt in self.params.target_counts.items():
            assignment.extend([t] * int(cnt))

        # If we have more positions than targets, extend by repeating last type
        if len(assignment) < len(positions):
            last = assignment[-1] if assignment else self.params.source_types[0]
            assignment.extend([last] * (len(positions) - len(assignment)))
        # If fewer positions than targets, truncate
        assignment = assignment[: len(positions)]

        for (y, x), t in zip(positions, assignment, strict=True):
            grid[y, x] = t
