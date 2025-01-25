from typing import Optional
import numpy as np
from omegaconf import DictConfig

from mettagrid.config.room.room import Room, OBJECTS, GameObject

class Random(Room):
    def __init__(
        self,
        width: int,
        height: int,
        objects: DictConfig,
        seed: Optional[int] = None,
        border_width: int = 0,
        border_object: GameObject = OBJECTS.Wall
    ):
        super().__init__(border_width=border_width, border_object=border_object)
        self._rng = np.random.default_rng(seed)
        self._width = width
        self._height = height
        self._objects = objects

    def _build(self):
        symbols = []
        area = self._width * self._height

        # Check if total objects exceed room size and halve counts if needed
        total_objects = sum(count for count in self._objects.values())
        while total_objects > 2*area / 3:
            for obj_name in self._objects:
                if obj_name != "agent":
                    self._objects[obj_name] = max(1, self._objects[obj_name] // 2)
                total_objects = sum(count for count in self._objects.values())

        # Add all objects in the proper amounts to a single large array.
        for obj_name, count in self._objects.items():
            symbols.extend([OBJECTS.named(obj_name).symbol] * count)

        assert(len(symbols) <= area), f"Too many objects in room: {len(symbols)} > {area}"
        symbols.extend([OBJECTS.Empty] * (area - len(symbols)))

        # Shuffle and reshape the array into a room.
        symbols = np.array(symbols).astype("U8")
        self._rng.shuffle(symbols)
        return symbols.reshape(self._height, self._width)
