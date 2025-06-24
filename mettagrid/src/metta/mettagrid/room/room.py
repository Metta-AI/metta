from typing import Optional

import numpy as np
import numpy.typing as npt

from metta.mettagrid.level_builder import Level, LevelBuilder


class Room(LevelBuilder):
    def __init__(self, border_width: int = 0, border_object: str = "wall", labels: Optional[list] = None):
        self._border_width = border_width
        self._border_object = border_object
        self.labels = labels or []

    def set_size_labels(self, width: int, height: int):
        area = width * height
        if area < 4000:
            self.labels.append("small")
        elif area < 6000:
            self.labels.append("medium")
        else:
            self.labels.append("large")

    def build(self) -> Level:
        grid = self._build()
        bordered_grid = self._add_border(grid)
        return Level(bordered_grid, self.labels)

    def _add_border(self, room):
        b = self._border_width
        h, w = room.shape
        final_level = np.full((h + b * 2, w + b * 2), self._border_object, dtype="<U50")
        final_level[b : b + h, b : b + w] = room
        return final_level

    def _build(self) -> npt.NDArray[np.str_]:
        raise NotImplementedError("Subclass must implement _build method")
