from typing import Dict, List, Literal

import numpy as np

from mettagrid.config.config import Config
from mettagrid.mapgen.scene import Scene


class RelabelConvertersParams(Config):
    # If target_counts omitted, types will be equalized across all positions
    target_counts: Dict[str, int] | None = None
    # Which names to consider as converter candidates to relabel
    source_types: List[str] = [
        "generator_red",
        "generator_blue",
        "generator_green",
        "lab",
    ]
    symmetry: Literal["none", "horizontal", "vertical", "both"] = "none"
    # Optional: explicit quadrant assignment when symmetry is both
    # Keys: "nw", "ne", "sw", "se"
    quadrant_types: Dict[Literal["nw", "ne", "sw", "se"], str] | None = None
    # Which types to assign to (if target_counts is None)
    target_types: List[str] = [
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

        # If requested, assign by quadrants (requires both symmetries enforced)
        if self.params.symmetry == "both" and self.params.quadrant_types:
            h, w = grid.shape
            mid_y = h // 2
            mid_x = w // 2

            def quadrant_for(y: int, x: int) -> str:
                if y < mid_y:
                    return "nw" if x < mid_x else "ne"
                return "sw" if x < mid_x else "se"

            qt = self.params.quadrant_types
            for y, x in positions:
                q = quadrant_for(y, x)
                t = qt.get(q)  # type: ignore[union-attr]
                if t is not None:
                    grid[y, x] = t
            return

        # Group by symmetry orbits so assignments keep symmetry
        h, w = grid.shape

        def key_for(y: int, x: int) -> tuple[int, int]:
            sym = self.params.symmetry
            if sym == "vertical":
                return (y, min(x, w - 1 - x))
            if sym == "horizontal":
                return (min(y, h - 1 - y), x)
            if sym == "both":
                return (min(y, h - 1 - y), min(x, w - 1 - x))
            return (y, x)

        orbits: Dict[tuple[int, int], list[tuple[int, int]]] = {}
        for y, x in positions:
            k = key_for(y, x)
            orbits.setdefault(k, []).append((y, x))

        # Sort orbits for determinism (by size desc, then key)
        orbit_items = sorted(orbits.items(), key=lambda kv: (-len(kv[1]), kv[0]))

        # Determine target counts (equalize if not provided or mismatched)
        if self.params.target_counts:
            type_names = sorted(self.params.target_counts.keys())
        else:
            type_names = sorted({t for t in self.params.target_types})
        total_positions = len(positions)
        if self.params.target_counts:
            remaining = {t: int(self.params.target_counts.get(t, 0)) for t in type_names}
            total_target = sum(remaining.values())
            if total_target != total_positions:
                # Recompute fair distribution
                per = total_positions // len(type_names)
                rem = total_positions % len(type_names)
                remaining = {t: per + (1 if i < rem else 0) for i, t in enumerate(type_names)}
        else:
            per = total_positions // len(type_names)
            rem = total_positions % len(type_names)
            remaining = {t: per + (1 if i < rem else 0) for i, t in enumerate(type_names)}

        # Greedy assignment of full orbits to types while counts permit
        for _key, coords in orbit_items:
            size = len(coords)
            # choose a type that has >= size remaining, tie-break by largest remaining then name
            candidates = [(t, c) for t, c in remaining.items() if c >= size]
            if not candidates:
                continue
            candidates.sort(key=lambda tc: (-tc[1], tc[0]))
            t = candidates[0][0]
            for y, x in coords:
                grid[y, x] = t
            remaining[t] -= size

        # If anything remains unassigned (due to counts not divisible by orbit sizes), fill leftover with best-effort
        # Collect leftover coords where still a source type
        ys2, xs2 = np.where(np.isin(grid, self.params.source_types))
        leftover = list(zip(ys2.tolist(), xs2.tolist(), strict=True))
        # Create a flat sequence of remaining type labels
        cycle: List[str] = []
        for t in type_names:
            cycle.extend([t] * max(0, remaining.get(t, 0)))
        # Interleave types to avoid large patches of a single type
        rest_labels: List[str] = []
        i = 0
        while len(rest_labels) < len(leftover) and cycle:
            rest_labels.append(cycle[i % len(cycle)])
            i += 1
        # Truncate/extend
        if len(rest_labels) < len(leftover):
            if rest_labels:
                rest_labels.extend([rest_labels[-1]] * (len(leftover) - len(rest_labels)))
            else:
                rest_labels.extend([self.params.source_types[0]] * len(leftover))
        rest_labels = rest_labels[: len(leftover)]
        for (y, x), t in zip(leftover, rest_labels, strict=True):
            grid[y, x] = t
