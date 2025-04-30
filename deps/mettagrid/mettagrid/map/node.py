import random
from dataclasses import dataclass
from typing import List, Optional

import numpy as np
import numpy.typing as npt
from omegaconf import DictConfig, ListConfig


@dataclass
class Area:
    id: int  # unique for areas in a node; not unique across nodes.
    grid: npt.NDArray[np.str_]
    tags: list[str]


# Container for a map slice, with a scene to render.
class Node:
    _areas: list[Area]

    def __init__(self, scene, grid: npt.NDArray[np.str_]):
        self.scene = scene

        # Not prefixed with `_`; scene renderers access these directly.
        self.grid = grid
        self.height = grid.shape[0]
        self.width = grid.shape[1]

        self._areas = []

        # { "lockname": [area_id1, area_id2, ...] }
        self._locks = {}
        self._full_area = Area(
            id=-1,
            grid=self.grid,
            tags=[],
        )

    def render(self):
        self.scene.render(self)

    def make_area(self, x: int, y: int, width: int, height: int, tags: Optional[List[str]] = None) -> Area:
        area = Area(
            id=len(self._areas),
            grid=self.grid[y : y + height, x : x + width],
            tags=tags or [],
        )
        self._areas.append(area)
        return area

    def select_areas(self, query) -> list[Area]:
        areas = self._areas

        selected_areas: list[Area] = []

        where = query.get("where")
        if where:
            if isinstance(where, str) and where == "full":
                selected_areas = [self._full_area]
            else:
                # Type check and handling
                if isinstance(where, (DictConfig, dict)) and "tags" in where:
                    tags = where.get("tags", [])
                    if isinstance(tags, list) or isinstance(tags, ListConfig):
                        for area in areas:
                            match = True
                            for tag in tags:
                                if tag not in area.tags:
                                    match = False
                                    break
                            if match:
                                selected_areas.append(area)
                    else:
                        raise ValueError(f"Invalid 'tags' format in 'where' clause: expected list, got {type(tags)}")
                else:
                    raise ValueError(f"Invalid 'where' structure: {where}")
        else:
            selected_areas = areas

        # Filter out locked areas.
        lock = query.get("lock")
        if lock:
            if lock not in self._locks:
                self._locks[lock] = set()

            # Remove areas that are locked.
            selected_areas = [area for area in selected_areas if area.id not in self._locks[lock]]

        limit = query.get("limit")
        if limit is not None and limit < len(selected_areas):
            order_by = query.get("order_by", "random")
            offset = query.get("offset")
            if order_by == "random":
                assert offset is None, "offset is not supported for random order"
                selected_areas = random.sample(selected_areas, k=limit)
            elif order_by == "first":
                offset = offset or 0
                selected_areas = selected_areas[offset : offset + limit]
            elif order_by == "last":
                if not offset:
                    selected_areas = selected_areas[-limit:]
                else:
                    selected_areas = selected_areas[-limit - offset : -offset]
            else:
                raise ValueError(f"Invalid order_by value: {order_by}")

        if lock:
            # Add final list of used areas to the lock.
            self._locks[lock].update([area.id for area in selected_areas])

        return selected_areas
