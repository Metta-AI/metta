import numpy as np

from metta.mettagrid.char_encoder import char_to_grid_object
from metta.mettagrid.room.room import Room


class Ascii(Room):
    def __init__(self, uri: str, border_width: int = 0, border_object: str = "wall"):
        super().__init__(border_width=border_width, border_object=border_object)
        with open(uri, "r", encoding="utf-8") as f:
            ascii_map = f.read()
        lines = ascii_map.strip().splitlines()
        self._level = np.array([list(line) for line in lines], dtype="U6")
        self._level = np.vectorize(char_to_grid_object)(self._level)
        self.set_size_labels(self._level.shape[1], self._level.shape[0])

    def _build(self):
        return self._level
